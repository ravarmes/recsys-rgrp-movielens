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
    k = 5

    recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

    X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
    omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

    X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF

    list_users = X_est.index.tolist()

    masculine = users_info[users_info['Gender'] == 1].index.intersection(list_users).tolist()
    feminine = users_info[users_info['Gender'] == 2].index.intersection(list_users).tolist()

    G = {1: masculine, 2: feminine}

    # # Calculando a quantidade de elementos em cada grupo
    # quantidades = {key: len(value) for key, value in G.items()}
    # # Exibindo os resultados
    # for key, quantidade in quantidades.items():
    #     print(f"Grupo {key}: {quantidade} elementos")

    glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpGender = glv.evaluate(X_est)
    losses_RgrpGender = glv.get_losses(X_est)

    print("\n\n------------------------------------------")
    print(f'Algorithm: {algorithm}')
    print(f'Group (Rgrp Gender): {RgrpGender:.7f}')
    print(f'RgrpGender (masculine): {losses_RgrpGender[1]:.7f}')
    print(f'RgrpGender (feminine) : {losses_RgrpGender[2]:.7f}')

    rmse = RMSE(X, omega)
    rmse_result = rmse.evaluate(X_est)
    print(f'RMSE: {rmse_result:.7f}')