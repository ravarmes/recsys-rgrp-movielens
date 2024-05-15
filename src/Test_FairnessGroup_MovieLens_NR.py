from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance
import matplotlib.pyplot as plt


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

    X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysSVD

    # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
    # The loss of group i as the mean squared estimation error over all known ratings in group i

    # G group: identifying the groups (NR: users grouped by number of ratings for available items)
    # advantaged group: 5% users with the highest number of item ratings
    # disadvantaged group: 95% users with the lowest number of item ratings
    list_users = X_est.index.tolist()
    advantaged_group = list_users[0:15]
    disadvantaged_group = list_users[15:300]
    G = {1: advantaged_group, 2: disadvantaged_group}

    # Calculando a quantidade de elementos em cada grupo
    quantidades = {key: len(value) for key, value in G.items()}
    # Exibindo os resultados
    for key, quantidade in quantidades.items():
        print(f"Grupo {key}: {quantidade} elementos")

    glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpNR = glv.evaluate(X_est)
    losses_RgrpNR = glv.get_losses(X_est)

    print("\n\n------------------------------------------")
    print(f'Algorithm: {algorithm}')
    print(f'Group (Rgrp NR): {RgrpNR:.7f}')
    print(f'RgrpNR (advantaged_group)   : {losses_RgrpNR[1]:.7f}')
    print(f'RgrpNR (disadvantaged_group): {losses_RgrpNR[2]:.7f}')

    RgrpNR_groups = ['Favorecidos', 'Desfavorecidos']
    plt.bar(RgrpNR_groups, losses_RgrpNR)
    plt.title(f'Rgrp (Avaliações) ({algorithm}): {RgrpNR:.7f}')
    plt.savefig(f'plots/RgrpNR-{algorithm}')
    plt.clf()
    # plt.show()