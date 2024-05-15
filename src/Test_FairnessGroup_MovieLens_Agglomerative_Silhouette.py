from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score


# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use movies with more ratings; False: otherwise


# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, 'RecSysALS') # RecSysALS or RecSysKNN or RecSysNMF

ilv = IndividualLossVariance(X, omega, 1)
losses = ilv.get_losses(X_est)

df = pd.DataFrame(columns=['Gender', 'Age'])
df['Gender'] = users_info['Gender']
df['Age'] = users_info['Age']
df['NR'] = users_info['NR']
df['Loss'] = losses

df.dropna(subset=['Loss'], inplace=True) # eliminating rows with empty values in the 'Loss' column
df = df.drop(columns=['Loss'])

df_scaled = df.copy()
df_scaled.iloc[:, :] = StandardScaler().fit_transform(df)

Z = hierarchy.linkage(df_scaled, 'ward')

# Inicialize uma lista para armazenar os silhouette scores
silhouette_scores = []

# Experimente diferentes números de clusters
for n_clusters in range(2, 11):  # Tente de 2 a 10 clusters
    cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = cluster.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plote os silhouette scores em relação ao número de clusters
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score para Determinação do Número de Clusters')
plt.grid(True)
plt.show()

    
    
