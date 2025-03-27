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
k_values = [3, 5, 7, 10, 15]  # Different k values to test for KNN
l = 5
theta = 3

# Dictionary to store results for each k value
knn_results = {}

for algorithm in algorithms:
    print(f"\n\nProcessing algorithm: {algorithm}")
    
    # For KNN, we'll test different k values
    if algorithm == 'RecSysKNN':
        for k in k_values:
            print(f"\nTesting KNN with k={k}")
            recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)
            
            X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path)
            omega = ~X.isnull()
            
            X_est = recsys.compute_X_est(X, algorithm)
            
            # Save matrices for analysis
            import os
            file_path_X = f'X_{algorithm}_k{k}.xlsx'
            if not os.path.exists(file_path_X):
                X.to_excel(file_path_X, index=True)
            
            file_path_X_est = f'X_est_{algorithm}_k{k}.xlsx'
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
            n_clusters = 5
            cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            df_scaled['cluster_agglomerative'] = cluster.fit_predict(df_scaled)
            
            users = list(df_scaled.index)
            groups = df_scaled['cluster_agglomerative']
            
            grouped_users = {i: [] for i in range(n_clusters)}
            for user, group in zip(users, groups):
                grouped_users[group].append(user)
            
            G = {1: grouped_users[0], 2: grouped_users[1], 3: grouped_users[2], 4: grouped_users[3], 5: grouped_users[4]}
            
            glv = GroupLossVariance(X, omega, G, 1)
            RgrpAgglomerative = glv.evaluate(X_est)
            losses_RgrpAgglomerative = glv.get_losses(X_est)
            
            print("\n------------------------------------------")
            print(f'Algorithm: {algorithm} (k={k})')
            print(f'Group (Rgrp Agglomerative): {RgrpAgglomerative:.7f}')
            print(f'RgrpAgglomerative (1): {losses_RgrpAgglomerative[1]:.7f}')
            print(f'RgrpAgglomerative (2): {losses_RgrpAgglomerative[2]:.7f}')
            print(f'RgrpAgglomerative (3): {losses_RgrpAgglomerative[3]:.7f}')
            print(f'RgrpAgglomerative (4): {losses_RgrpAgglomerative[4]:.7f}')
            print(f'RgrpAgglomerative (5): {losses_RgrpAgglomerative[5]:.7f}')
            
            rmse = RMSE(X, omega)
            rmse_result = rmse.evaluate(X_est)
            print(f'RMSE: {rmse_result:.7f}')
            
            # Store results
            knn_results[k] = {
                'RgrpAgglomerative': RgrpAgglomerative,
                'group_losses': losses_RgrpAgglomerative,
                'RMSE': rmse_result
            }
    else:
        # For other algorithms, use default k=3
        k = 3
        recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)
        
        X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir = Data_path)
        omega = ~X.isnull()
        
        X_est = recsys.compute_X_est(X, algorithm)
        
        # Save matrices for analysis
        import os
        file_path_X = f'X_{algorithm}.xlsx'
        if not os.path.exists(file_path_X):
            X.to_excel(file_path_X, index=True)
        
        file_path_X_est = f'X_est_{algorithm}.xlsx'
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
        n_clusters = 5
        cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        df_scaled['cluster_agglomerative'] = cluster.fit_predict(df_scaled)
        
        users = list(df_scaled.index)
        groups = df_scaled['cluster_agglomerative']
        
        grouped_users = {i: [] for i in range(n_clusters)}
        for user, group in zip(users, groups):
            grouped_users[group].append(user)
        
        G = {1: grouped_users[0], 2: grouped_users[1], 3: grouped_users[2], 4: grouped_users[3], 5: grouped_users[4]}
        
        glv = GroupLossVariance(X, omega, G, 1)
        RgrpAgglomerative = glv.evaluate(X_est)
        losses_RgrpAgglomerative = glv.get_losses(X_est)
        
        print("\n------------------------------------------")
        print(f'Algorithm: {algorithm}')
        print(f'Group (Rgrp Agglomerative): {RgrpAgglomerative:.7f}')
        print(f'RgrpAgglomerative (1): {losses_RgrpAgglomerative[1]:.7f}')
        print(f'RgrpAgglomerative (2): {losses_RgrpAgglomerative[2]:.7f}')
        print(f'RgrpAgglomerative (3): {losses_RgrpAgglomerative[3]:.7f}')
        print(f'RgrpAgglomerative (4): {losses_RgrpAgglomerative[4]:.7f}')
        print(f'RgrpAgglomerative (5): {losses_RgrpAgglomerative[5]:.7f}')
        
        rmse = RMSE(X, omega)
        rmse_result = rmse.evaluate(X_est)
        print(f'RMSE: {rmse_result:.7f}')

# Generate sensitivity analysis report
print("\n\nSensitivity Analysis Report for KNN")
print("=====================================")
print("k | RgrpAgglomerative | Group1 Loss | Group2 Loss | Group3 Loss | Group4 Loss | Group5 Loss | RMSE")
print("-" * 100)
for k in k_values:
    results = knn_results[k]
    print(f"{k:2d} | {results['RgrpAgglomerative']:.7f} | {results['group_losses'][1]:.7f} | {results['group_losses'][2]:.7f} | {results['group_losses'][3]:.7f} | {results['group_losses'][4]:.7f} | {results['group_losses'][5]:.7f} | {results['RMSE']:.7f}")

# Save results to a file
with open('knn_sensitivity_analysis_agglomerative.txt', 'w') as f:
    f.write("Sensitivity Analysis Report for KNN\n")
    f.write("=====================================\n")
    f.write("k | RgrpAgglomerative | Group1 Loss | Group2 Loss | Group3 Loss | Group4 Loss | Group5 Loss | RMSE\n")
    f.write("-" * 100 + "\n")
    for k in k_values:
        results = knn_results[k]
        f.write(f"{k:2d} | {results['RgrpAgglomerative']:.7f} | {results['group_losses'][1]:.7f} | {results['group_losses'][2]:.7f} | {results['group_losses'][3]:.7f} | {results['group_losses'][4]:.7f} | {results['group_losses'][5]:.7f} | {results['RMSE']:.7f}\n")