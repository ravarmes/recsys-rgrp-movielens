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
            
            # Group fairness analysis by Gender
            list_users = X_est.index.tolist()
            
            masculine = users_info[users_info['Gender'] == 1].index.intersection(list_users).tolist()
            feminine = users_info[users_info['Gender'] == 2].index.intersection(list_users).tolist()
            
            G = {1: masculine, 2: feminine}
            
            glv = GroupLossVariance(X, omega, G, 1)
            RgrpGender = glv.evaluate(X_est)
            losses_RgrpGender = glv.get_losses(X_est)
            
            print("\n------------------------------------------")
            print(f'Algorithm: {algorithm} (k={k})')
            print(f'Group (Rgrp Gender): {RgrpGender:.7f}')
            print(f'RgrpGender (masculine): {losses_RgrpGender[1]:.7f}')
            print(f'RgrpGender (feminine): {losses_RgrpGender[2]:.7f}')
            
            rmse = RMSE(X, omega)
            rmse_result = rmse.evaluate(X_est)
            print(f'RMSE: {rmse_result:.7f}')
            
            # Store results
            knn_results[k] = {
                'RgrpGender': RgrpGender,
                'masculine_loss': losses_RgrpGender[1],
                'feminine_loss': losses_RgrpGender[2],
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
        
        # Group fairness analysis by Gender
        list_users = X_est.index.tolist()
        
        masculine = users_info[users_info['Gender'] == 1].index.intersection(list_users).tolist()
        feminine = users_info[users_info['Gender'] == 2].index.intersection(list_users).tolist()
        
        G = {1: masculine, 2: feminine}
        
        glv = GroupLossVariance(X, omega, G, 1)
        RgrpGender = glv.evaluate(X_est)
        losses_RgrpGender = glv.get_losses(X_est)
        
        print("\n------------------------------------------")
        print(f'Algorithm: {algorithm}')
        print(f'Group (Rgrp Gender): {RgrpGender:.7f}')
        print(f'RgrpGender (masculine): {losses_RgrpGender[1]:.7f}')
        print(f'RgrpGender (feminine): {losses_RgrpGender[2]:.7f}')
        
        rmse = RMSE(X, omega)
        rmse_result = rmse.evaluate(X_est)
        print(f'RMSE: {rmse_result:.7f}')

# Generate sensitivity analysis report
print("\n\nSensitivity Analysis Report for KNN")
print("=====================================")
print("k | RgrpGender | Masculine Loss | Feminine Loss | RMSE")
print("-" * 70)
for k in k_values:
    results = knn_results[k]
    print(f"{k:2d} | {results['RgrpGender']:.7f} | {results['masculine_loss']:.7f} | {results['feminine_loss']:.7f} | {results['RMSE']:.7f}")

# Save results to a file
with open('knn_sensitivity_analysis_gender.txt', 'w') as f:
    f.write("Sensitivity Analysis Report for KNN\n")
    f.write("=====================================\n")
    f.write("k | RgrpGender | Masculine Loss | Feminine Loss | RMSE\n")
    f.write("-" * 70 + "\n")
    for k in k_values:
        results = knn_results[k]
        f.write(f"{k:2d} | {results['RgrpGender']:.7f} | {results['masculine_loss']:.7f} | {results['feminine_loss']:.7f} | {results['RMSE']:.7f}\n")