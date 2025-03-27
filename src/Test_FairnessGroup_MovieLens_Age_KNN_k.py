from RecSys import RecSys
from UserFairness import GroupLossVariance
from UserFairness import RMSE

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
            
            # Group fairness analysis by Age
            list_users = X_est.index.tolist()
            
            age_00_17 = users_info[users_info['Age'] == 1].index.intersection(list_users).tolist()
            age_18_24 = users_info[users_info['Age'] == 18].index.intersection(list_users).tolist()
            age_25_34 = users_info[users_info['Age'] == 25].index.intersection(list_users).tolist()
            age_35_44 = users_info[users_info['Age'] == 35].index.intersection(list_users).tolist()
            age_45_49 = users_info[users_info['Age'] == 45].index.intersection(list_users).tolist()
            age_50_55 = users_info[users_info['Age'] == 50].index.intersection(list_users).tolist()
            age_56_00 = users_info[users_info['Age'] == 56].index.intersection(list_users).tolist()
            
            G = {1: age_00_17, 2: age_18_24, 3: age_25_34, 4: age_35_44, 5: age_45_49, 6: age_50_55, 7: age_56_00}
            
            glv = GroupLossVariance(X, omega, G, 1)
            RgrpAge = glv.evaluate(X_est)
            losses_RgrpAge = glv.get_losses(X_est)
            
            print("\n------------------------------------------")
            print(f'Algorithm: {algorithm} (k={k})')
            print(f'Group (Rgrp Age): {RgrpAge:.7f}')
            for i in range(1, 8):
                print(f'RgrpAge (Group {i}): {losses_RgrpAge[i]:.7f}')
            
            rmse = RMSE(X, omega)
            rmse_result = rmse.evaluate(X_est)
            print(f'RMSE: {rmse_result:.7f}')
            
            # Store results
            knn_results[k] = {
                'RgrpAge': RgrpAge,
                'group_losses': losses_RgrpAge,
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
        
        # Group fairness analysis by Age
        list_users = X_est.index.tolist()
        
        age_00_17 = users_info[users_info['Age'] == 1].index.intersection(list_users).tolist()
        age_18_24 = users_info[users_info['Age'] == 18].index.intersection(list_users).tolist()
        age_25_34 = users_info[users_info['Age'] == 25].index.intersection(list_users).tolist()
        age_35_44 = users_info[users_info['Age'] == 35].index.intersection(list_users).tolist()
        age_45_49 = users_info[users_info['Age'] == 45].index.intersection(list_users).tolist()
        age_50_55 = users_info[users_info['Age'] == 50].index.intersection(list_users).tolist()
        age_56_00 = users_info[users_info['Age'] == 56].index.intersection(list_users).tolist()
        
        G = {1: age_00_17, 2: age_18_24, 3: age_25_34, 4: age_35_44, 5: age_45_49, 6: age_50_55, 7: age_56_00}
        
        glv = GroupLossVariance(X, omega, G, 1)
        RgrpAge = glv.evaluate(X_est)
        losses_RgrpAge = glv.get_losses(X_est)
        
        print("\n------------------------------------------")
        print(f'Algorithm: {algorithm}')
        print(f'Group (Rgrp Age): {RgrpAge:.7f}')
        for i in range(1, 8):
            print(f'RgrpAge (Group {i}): {losses_RgrpAge[i]:.7f}')
        
        rmse = RMSE(X, omega)
        rmse_result = rmse.evaluate(X_est)
        print(f'RMSE: {rmse_result:.7f}')

# Generate sensitivity analysis report
print("\n\nSensitivity Analysis Report for KNN")
print("=====================================")
print("k | RgrpAge | Group1 Loss | Group2 Loss | Group3 Loss | Group4 Loss | Group5 Loss | Group6 Loss | Group7 Loss | RMSE")
print("-" * 120)
for k in k_values:
    results = knn_results[k]
    print(f"{k:2d} | {results['RgrpAge']:.7f} | {results['group_losses'][1]:.7f} | {results['group_losses'][2]:.7f} | {results['group_losses'][3]:.7f} | {results['group_losses'][4]:.7f} | {results['group_losses'][5]:.7f} | {results['group_losses'][6]:.7f} | {results['group_losses'][7]:.7f} | {results['RMSE']:.7f}")

# Save results to a file
with open('knn_sensitivity_analysis_age.txt', 'w') as f:
    f.write("Sensitivity Analysis Report for KNN\n")
    f.write("=====================================\n")
    f.write("k | RgrpAge | Group1 Loss | Group2 Loss | Group3 Loss | Group4 Loss | Group5 Loss | Group6 Loss | Group7 Loss | RMSE\n")
    f.write("-" * 120 + "\n")
    for k in k_values:
        results = knn_results[k]
        f.write(f"{k:2d} | {results['RgrpAge']:.7f} | {results['group_losses'][1]:.7f} | {results['group_losses'][2]:.7f} | {results['group_losses'][3]:.7f} | {results['group_losses'][4]:.7f} | {results['group_losses'][5]:.7f} | {results['group_losses'][6]:.7f} | {results['group_losses'][7]:.7f} | {results['RMSE']:.7f}\n")
