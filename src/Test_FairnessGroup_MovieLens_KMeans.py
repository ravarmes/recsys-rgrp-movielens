from RecSys import RecSys
from UserFairness import GroupLossVariance
from UserFairness import IndividualLossVariance
from UserFairness import RMSE

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users = 300
n_items = 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithms = ['RecSysALS', 'RecSysNMF', 'RecSysKNN']

resultados = []

for algorithm in algorithms:
    print(f"\n\nProcessing algorithm: {algorithm}")
    # parameters for calculating fairness measures
    l = 5
    theta = 3
    k = 5

    recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

    X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
    omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

    X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF

    ilv = IndividualLossVariance(X, omega, 1)
    losses = ilv.get_losses(X_est)

    # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
    # The loss of group i as the mean squared estimation error over all known ratings in group i

    # G group: identifying the groups (Age, Gender, NR of users)
    # Using K-Means clustering to identify groups based on multiple attributes

    df = pd.DataFrame(columns=['Gender', 'Age'])
    df['Gender'] = users_info['Gender']
    df['Age'] = users_info['Age']
    df['NR'] = users_info['NR']
    df['Loss'] = losses

    df.dropna(subset=['Loss'], inplace=True) # eliminating rows with empty values in the 'Loss' column

    df = df.drop(columns=['Loss'])

    # Standardize the data for K-Means
    df_scaled = df.copy()
    df_scaled.iloc[:, :] = StandardScaler().fit_transform(df)

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_scaled['cluster_kmeans'] = kmeans.fit_predict(df_scaled)

    users = list(df_scaled.index)
    groups = df_scaled['cluster_kmeans']

    grouped_users = {i: [] for i in range(n_clusters)}
    for user, group in zip(users, groups):
        grouped_users[group].append(user)

    G = {1: grouped_users[0], 2: grouped_users[1], 3: grouped_users[2], 4: grouped_users[3], 5: grouped_users[4]}

    glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpKMeans = glv.evaluate(X_est)
    losses_RgrpKMeans = glv.get_losses(X_est)

    print("\n------------------------------------------")
    print(f'Algorithm: {algorithm}')
    print(f'Group (Rgrp KMeans): {RgrpKMeans:.7f}')
    print(f'RgrpKMeans (1): {losses_RgrpKMeans[1]:.7f}')
    print(f'RgrpKMeans (2): {losses_RgrpKMeans[2]:.7f}')
    print(f'RgrpKMeans (3): {losses_RgrpKMeans[3]:.7f}')
    print(f'RgrpKMeans (4): {losses_RgrpKMeans[4]:.7f}')
    print(f'RgrpKMeans (5): {losses_RgrpKMeans[5]:.7f}')

    resultados.append(f'{RgrpKMeans:.7f}')

    rmse = RMSE(X, omega)
    rmse_result = rmse.evaluate(X_est)
    print(f'RMSE: {rmse_result:.7f}')

print("\nFinal results for all algorithms:")
print(resultados)