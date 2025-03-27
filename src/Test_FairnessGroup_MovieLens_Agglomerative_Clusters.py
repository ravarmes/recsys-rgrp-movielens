from RecSys import RecSys
from UserFairness import GroupLossVariance
from UserFairness import IndividualLossVariance
from UserFairness import RMSE

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import numpy as np

# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users = 300
n_items = 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithms = ['RecSysALS', 'RecSysNMF', 'RecSysKNN']

# Parameters for sensitivity analysis
n_clusters_values = [3, 4, 5, 6, 7]  # Different numbers of clusters to test
l = 5
theta = 3
k = 5

# Dictionary to store results for each algorithm and number of clusters
results = {algorithm: {} for algorithm in algorithms}

for algorithm in algorithms:
    print(f"\n\nProcessing algorithm: {algorithm}")
    
    # For each algorithm, test different numbers of clusters
    for n_clusters in n_clusters_values:
        print(f"\nTesting with {n_clusters} clusters")
        recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)
        
        X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path)
        omega = ~X.isnull()
        
        X_est = recsys.compute_X_est(X, algorithm)
        
        # Save matrices for analysis
        import os
        file_path_X = f'X_{algorithm}_clusters{n_clusters}.xlsx'
        if not os.path.exists(file_path_X):
            X.to_excel(file_path_X, index=True)
        
        file_path_X_est = f'X_est_{algorithm}_clusters{n_clusters}.xlsx'
        X_est.to_excel(file_path_X_est, index=True)
        
        # Calculate individual losses
        ilv = IndividualLossVariance(X, omega, 1)
        losses = ilv.get_losses(X_est)
        
        # Prepare data for clustering
        df = pd.DataFrame(columns=['Gender', 'Age'])
        df['Gender'] = users_info['Gender']
        df['Age'] = users_info['Age']
        df['NR'] = users_info['NR']
        df['Loss'] = losses
        
        df.dropna(subset=['Loss'], inplace=True)  # eliminating rows with empty values in the 'Loss' column
        df = df.drop(columns=['Loss'])
        
        # Standardize the data
        df_scaled = df.copy()
        df_scaled.iloc[:, :] = StandardScaler().fit_transform(df)
        
        # Perform hierarchical clustering
        cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        df_scaled['cluster_agglomerative'] = cluster.fit_predict(df_scaled)
        
        users = list(df_scaled.index)
        groups = df_scaled['cluster_agglomerative']
        
        grouped_users = {i: [] for i in range(n_clusters)}
        for user, group in zip(users, groups):
            grouped_users[group].append(user)
        
        G = {i+1: grouped_users[i] for i in range(n_clusters)}
        
        glv = GroupLossVariance(X, omega, G, 1)
        RgrpAgglomerative = glv.evaluate(X_est)
        losses_RgrpAgglomerative = glv.get_losses(X_est)
        
        print("\n------------------------------------------")
        print(f'Algorithm: {algorithm} (n_clusters={n_clusters})')
        print(f'Group (Rgrp Agglomerative): {RgrpAgglomerative:.7f}')
        for i in range(1, n_clusters + 1):
            print(f'RgrpAgglomerative ({i}): {losses_RgrpAgglomerative[i]:.7f}')
        
        rmse = RMSE(X, omega)
        rmse_result = rmse.evaluate(X_est)
        print(f'RMSE: {rmse_result:.7f}')
        
        # Store results for this algorithm and number of clusters
        results[algorithm][n_clusters] = {
            'RgrpAgglomerative': RgrpAgglomerative,
            'group_losses': losses_RgrpAgglomerative,
            'RMSE': rmse_result
        }

# Generate sensitivity analysis report for each algorithm
for algorithm in algorithms:
    print(f"\n\nSensitivity Analysis Report for {algorithm}")
    print("=" * 80)
    print("n_clusters | RgrpAgglomerative | Group Losses | RMSE")
    print("-" * 80)
    for n_clusters in n_clusters_values:
        algorithm_results = results[algorithm][n_clusters]
        group_losses_str = " | ".join([f"{algorithm_results['group_losses'][i]:.7f}" for i in range(1, n_clusters + 1)])
        print(f"{n_clusters:2d} | {algorithm_results['RgrpAgglomerative']:.7f} | {group_losses_str} | {algorithm_results['RMSE']:.7f}")

# Save results to a file
with open('clusters_sensitivity_analysis.txt', 'w') as f:
    for algorithm in algorithms:
        f.write(f"\n\nSensitivity Analysis Report for {algorithm}\n")
        f.write("=" * 80 + "\n")
        f.write("n_clusters | RgrpAgglomerative | Group Losses | RMSE\n")
        f.write("-" * 80 + "\n")
        for n_clusters in n_clusters_values:
            algorithm_results = results[algorithm][n_clusters]
            group_losses_str = " | ".join([f"{algorithm_results['group_losses'][i]:.7f}" for i in range(1, n_clusters + 1)])
            f.write(f"{n_clusters:2d} | {algorithm_results['RgrpAgglomerative']:.7f} | {group_losses_str} | {algorithm_results['RMSE']:.7f}\n")