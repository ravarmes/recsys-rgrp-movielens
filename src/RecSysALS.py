import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
import numpy.ma as ma

class RecSysALS():
    __metaclass__ = ABCMeta
    
    def __init__(self, rank, lambda_=1e-6, ratings=None):
        self.rank = rank
        self.lambda_ = lambda_
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)

    def get_U(self):
        return pd.DataFrame(self.U, index=self.ratings.index)
    
    def get_V(self):
        return pd.DataFrame(self.V, columns=self.ratings.columns)
    
    @abstractmethod
    def fit_model(self):
        pass

    def als(self, X, k, lambda_, max_iter, threshold):
        def solve_V(X, W, U):
            X = X.values
            n, d = X.shape
            V = np.zeros((d, k))
            X = X.T
            W = W.T.values
            I = lambda_ * np.eye(k)
            for j, x_j in enumerate(X):
                v_j = np.linalg.solve(U[W[j]].T.dot(U[W[j]]) + I, U[W[j]].T.dot(x_j[W[j]]))
                V[j] = v_j
            return V

        def solve_U(X, W, V):
            X = X.values
            W = W.values
            n, d = X.shape
            U = np.zeros((n, k))
            I = lambda_ * np.eye(k)
            for i, x_i in enumerate(X):
                u_i = np.linalg.solve(V[W[i]].T.dot(V[W[i]]) + I, V[W[i]].T.dot(x_i[W[i]]))
                U[i] = u_i
            return U

        W = ~X.isnull()
        n, d = X.shape
        U = np.ones((n, k))
        V = solve_V(X, W, U)
        n_known = float(W.sum().sum())
        RMSE = np.sqrt((X - pd.DataFrame(U.dot(V.T), index=X.index, columns=X.columns)).pow(2).sum().sum() / n_known)
        RMSEs = [RMSE]
        for i in range(max_iter):
            U_new = solve_U(X, W, V)
            V_new = solve_V(X, W, U_new)
            RMSE_new = np.sqrt((X - pd.DataFrame(U_new.dot(V_new.T), index=X.index, columns=X.columns)).pow(2).sum().sum() / n_known)
            if (RMSE - RMSE_new) < threshold:
                RMSEs.append(RMSE_new)
                break
            else:
                RMSEs.append(RMSE_new)
                RMSE = RMSE_new
                U = U_new
                V = V_new
        return U, V.T

class als_RecSysALS(RecSysALS):
    def fit_model(self, ratings=None, max_iter=50, threshold=1e-5):
        X = self.ratings if ratings is None else ratings
        self.U, self.V = self.als(X, self.rank, self.lambda_, max_iter, threshold)
        self.pred = pd.DataFrame(self.U.dot(self.V), index=X.index, columns=X.columns)
        self.error = ma.power(ma.masked_invalid(X - self.pred), 2).sum()
        return self.pred, self.error
