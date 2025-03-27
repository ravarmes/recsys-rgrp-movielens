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
            
            # Group fairness analysis
            list_users = X_est.index.tolist()
            advantaged_group = list_users[0:15]
            disadvantaged_group = list_users[15:300]
            G = {1: advantaged_group, 2: disadvantaged_group}
            
            glv = GroupLossVariance(X, omega, G, 1)
            RgrpActivity = glv.evaluate(X_est)
            losses_RgrpActivity = glv.get_losses(X_est)
            
            print("\n------------------------------------------")
            print(f'Algorithm: {algorithm} (k={k})')
            print(f'Group (Rgrp Activity): {RgrpActivity:.7f}')
            print(f'RgrpActivity (advantaged_group): {losses_RgrpActivity[1]:.7f}')
            print(f'RgrpActivity (disadvantaged_group): {losses_RgrpActivity[2]:.7f}')
            
            rmse = RMSE(X, omega)
            rmse_result = rmse.evaluate(X_est)
            print(f'RMSE: {rmse_result:.7f}')
            
            # Store results
            knn_results[k] = {
                'RgrpActivity': RgrpActivity,
                'advantaged_loss': losses_RgrpActivity[1],
                'disadvantaged_loss': losses_RgrpActivity[2],
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
        
        # Group fairness analysis
        list_users = X_est.index.tolist()
        advantaged_group = list_users[0:15]
        disadvantaged_group = list_users[15:300]
        G = {1: advantaged_group, 2: disadvantaged_group}
        
        glv = GroupLossVariance(X, omega, G, 1)
        RgrpActivity = glv.evaluate(X_est)
        losses_RgrpActivity = glv.get_losses(X_est)
        
        print("\n------------------------------------------")
        print(f'Algorithm: {algorithm}')
        print(f'Group (Rgrp Activity): {RgrpActivity:.7f}')
        print(f'RgrpActivity (advantaged_group): {losses_RgrpActivity[1]:.7f}')
        print(f'RgrpActivity (disadvantaged_group): {losses_RgrpActivity[2]:.7f}')
        
        rmse = RMSE(X, omega)
        rmse_result = rmse.evaluate(X_est)
        print(f'RMSE: {rmse_result:.7f}')

# Generate sensitivity analysis report
print("\n\nSensitivity Analysis Report for KNN")
print("=====================================")
print("k | RgrpActivity | Adv. Loss | Disadv. Loss | RMSE")
print("-" * 60)
for k in k_values:
    results = knn_results[k]
    print(f"{k:2d} | {results['RgrpActivity']:.7f} | {results['advantaged_loss']:.7f} | {results['disadvantaged_loss']:.7f} | {results['RMSE']:.7f}")

# Save results to a file
with open('knn_sensitivity_analysis_activity.txt', 'w') as f:
    f.write("Sensitivity Analysis Report for KNN\n")
    f.write("=====================================\n")
    f.write("k | RgrpActivity | Adv. Loss | Disadv. Loss | RMSE\n")
    f.write("-" * 60 + "\n")
    for k in k_values:
        results = knn_results[k]
        f.write(f"{k:2d} | {results['RgrpActivity']:.7f} | {results['advantaged_loss']:.7f} | {results['disadvantaged_loss']:.7f} | {results['RMSE']:.7f}\n")