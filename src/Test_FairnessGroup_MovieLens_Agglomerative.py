from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy


# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithms = ['RecSysALS', 'RecSysNMF', 'RecSysKNN']
resultados = []

for algorithm in algorithms:

    # parameters for calculating fairness measures
    l = 5
    theta = 3
    k = 3

    recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

    X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
    omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

    X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysSVD

    ilv = IndividualLossVariance(X, omega, 1)
    losses = ilv.get_losses(X_est)

    # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
    # The loss of group i as the mean squared estimation error over all known ratings in group i

    # G group: identifying the groups (Age, Gender, NR of users)
    # The configuration of groups was based in the hierarchical clustering (tree clustering - dendrogram)
    # Clusters 1, 2 and 3

    df = pd.DataFrame(columns=['Gender', 'Age'])
    df['Gender'] = users_info['Gender']
    df['Age'] = users_info['Age']
    df['NR'] = users_info['NR']
    df['Loss'] = losses

    df.dropna(subset=['Loss'], inplace=True) # eliminating rows with empty values in the 'Loss' column

    df = df.drop(columns=['Loss'])

    # # Gráfico da Correlação
    # plt.figure(figsize=(7, 7))
    # corr = np.corrcoef(df.values, rowvar=False)
    # sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', cbar=False, xticklabels=df.columns, yticklabels=df.columns)
    # plt.show()

    # # Gráfico pairplot
    # print(df.head())
    # sns.pairplot(df)
    # plt.title('Pairplot of DataFrame')
    # plt.show()

    df_scaled = df.copy()
    df_scaled.iloc[:, :] = StandardScaler().fit_transform(df)

    Z = hierarchy.linkage(df_scaled, 'ward')
    
    # # Plotagem do dendrograma
    # plt.clf()  # Limpa o gráfico anterior
    # plt.figure(figsize=(22, 10))
    # plt.grid(axis='y')
    # dn = hierarchy.dendrogram(Z, labels=list(df.index), leaf_font_size=8)
    # plt.title('Dendrogram')
    # plt.show()

    n_clusters = 5
    cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward') # ['0.0061508', '0.0048103', '0.0030414'] ***

    df_scaled['cluster_agglomerative'] = cluster.fit_predict(df_scaled)

    users = list(df_scaled.index)
    groups = df_scaled['cluster_agglomerative']

    grouped_users = {i: [] for i in range(n_clusters)}
    for user, group in zip(users, groups):
        grouped_users[group].append(user)

    # for gp, ctr in grouped_users.items():
    #     print(f'Cluster {gp}: {ctr}\n')

    G = {1: grouped_users[0], 2: grouped_users[1], 3: grouped_users[2], 4: grouped_users[3], 5: grouped_users[4]}

    # # Calculando a quantidade de elementos em cada grupo
    # quantidades = {key: len(value) for key, value in G.items()}
    # # Exibindo os resultados
    # for key, quantidade in quantidades.items():
    #     print(f"Grupo {key}: {quantidade} elementos")

    glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
    RgrpAll = glv.evaluate(X_est)
    losses_RgrpAll = glv.get_losses(X_est)

    print("\n\n------------------------------------------")
    print(f'Algorithm: {algorithm}')
    print(f'Group (Rgrp All): {RgrpAll:.7f}')
    print(f'RgrpAll (1) : {losses_RgrpAll[1]:.7f}')
    print(f'RgrpAll (2) : {losses_RgrpAll[2]:.7f}')
    print(f'RgrpAll (3) : {losses_RgrpAll[3]:.7f}')
    print(f'RgrpAll (4) : {losses_RgrpAll[4]:.7f}')
    print(f'RgrpAll (5) : {losses_RgrpAll[5]:.7f}')

    RgrpAll_groups = ['G1', 'G2', 'G3', 'G4', 'G5']
    plt.bar(RgrpAll_groups, losses_RgrpAll)
    plt.title(f'Rgrp (Aglomerativo) ({algorithm}): {RgrpAll:.7f}')
    plt.savefig(f'plots/RgrpAgglomerative-{algorithm}')
    plt.clf()
    #plt.show()

    resultados.append(f'{RgrpAll:.7f}')

print(resultados)
