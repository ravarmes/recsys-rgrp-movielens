from RecSys import RecSys
from UserFairness import GroupLossVariance
from UserFairness import RMSE


# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithms = ['RecSysALS', 'RecSysNMF', 'RecSysKNN']

for algorithm in algorithms:

    # parameters for calculating fairness measures
    l = 5
    theta = 3
    k = 3

    recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

    X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
    omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

    X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF

    # Salve o DataFrame em um arquivo Excel
    import os
    file_path_X = f'X_{algorithm}.xlsx'  # Definindo o caminho e nome do arquivo
    if not os.path.exists(file_path_X):
        X.to_excel(file_path_X, index=True)

    file_path_X_est = f'X_est_{algorithm}.xlsx'  # Definindo o caminho e nome do arquivo
    X_est.to_excel(file_path_X_est, index=True)  # Altere index=False se não quiser incluir o índice

    # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
    # The loss of group i as the mean squared estimation error over all known ratings in group i

    # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # advantaged group: 5% users with the highest number of item ratings
    # disadvantaged group: 95% users with the lowest number of item ratings
    list_users = X_est.index.tolist()
    advantaged_group = list_users[0:15]
    disadvantaged_group = list_users[15:300]
    G = {1: advantaged_group, 2: disadvantaged_group}

    # # Calculando a quantidade de elementos em cada grupo
    # quantidades = {key: len(value) for key, value in G.items()}
    # # Exibindo os resultados
    # for key, quantidade in quantidades.items():
    #     print(f"Grupo {key}: {quantidade} elementos")

    glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpActivity = glv.evaluate(X_est)
    losses_RgrpActivity = glv.get_losses(X_est)

    print("\n\n------------------------------------------")
    print(f'Algorithm: {algorithm}')
    print(f'Group (Rgrp Activity): {RgrpActivity:.7f}')
    print(f'RgrpActivity (advantaged_group)   : {losses_RgrpActivity[1]:.7f}')
    print(f'RgrpActivity (disadvantaged_group): {losses_RgrpActivity[2]:.7f}')

    rmse = RMSE(X, omega)
    rmse_result = rmse.evaluate(X_est)
    print(f'RMSE: {rmse_result:.7f}')