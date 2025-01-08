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

    # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
    # The loss of group i as the mean squared estimation error over all known ratings in group i

    # G group: identifying the groups (Age: users grouped by age)
    #print(users_info)

    list_users = X_est.index.tolist()

    age_00_17 = users_info[users_info['Age'] ==  1].index.intersection(list_users).tolist()
    age_18_24 = users_info[users_info['Age'] == 18].index.intersection(list_users).tolist()
    age_25_34 = users_info[users_info['Age'] == 25].index.intersection(list_users).tolist()
    age_35_44 = users_info[users_info['Age'] == 35].index.intersection(list_users).tolist()
    age_45_49 = users_info[users_info['Age'] == 45].index.intersection(list_users).tolist()
    age_50_55 = users_info[users_info['Age'] == 50].index.intersection(list_users).tolist()
    age_56_00 = users_info[users_info['Age'] == 56].index.intersection(list_users).tolist()

    G = {1: age_00_17, 2: age_18_24, 3: age_25_34, 4: age_35_44, 5: age_45_49, 6: age_50_55, 7: age_56_00}

    # # Calculando a quantidade de elementos em cada grupo
    # quantidades = {key: len(value) for key, value in G.items()}
    # # Exibindo os resultados
    # for key, quantidade in quantidades.items():
    #     print(f"Grupo {key}: {quantidade} elementos")

    glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpAge = glv.evaluate(X_est)
    losses_RgrpAge = glv.get_losses(X_est)

    print("\n\n------------------------------------------")
    print(f'Algorithm: {algorithm}')
    print(f'Group (Rgrp Age): {RgrpAge:.7f}')
    print(f'RgrpAge (age_00_17) : {losses_RgrpAge[1]:.7f}')
    print(f'RgrpAge (age_18_24) : {losses_RgrpAge[2]:.7f}')
    print(f'RgrpAge (age_25_34) : {losses_RgrpAge[3]:.7f}')
    print(f'RgrpAge (age_35_44) : {losses_RgrpAge[4]:.7f}')
    print(f'RgrpAge (age_45_49) : {losses_RgrpAge[5]:.7f}')
    print(f'RgrpAge (age_50_55) : {losses_RgrpAge[6]:.7f}')
    print(f'RgrpAge (age_56_00) : {losses_RgrpAge[7]:.7f}')

    rmse = RMSE(X, omega)
    rmse_result = rmse.evaluate(X_est)
    print(f'RMSE: {rmse_result:.7f}')
