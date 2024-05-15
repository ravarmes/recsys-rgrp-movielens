import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod

class RecSysSVD:
    __metaclass__ = ABCMeta

    def __init__(self, k, ratings=None):
        self.k = k
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        self.ratings = ratings

    def fit_model(self):
        # Preenche os valores NaN com a média de cada item
        ratings_filled = self.ratings.fillna(self.ratings.mean(axis=0))
        # Centraliza os dados subtraindo a média do usuário
        user_ratings_mean = np.mean(ratings_filled, axis=1)
        ratings_demeaned = ratings_filled.sub(user_ratings_mean, axis='index')
        
        # Realiza a SVD
        U, sigma, Vt = svds(ratings_demeaned, k=self.k)
        
        # A matriz sigma retornada é apenas a diagonal, então a transformamos em uma matriz diagonal completa
        sigma = np.diag(sigma)
        
        # Calcula a matriz de predição
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
        
        # Converte a matriz de predição em um DataFrame
        preds_df = pd.DataFrame(all_user_predicted_ratings, index=self.ratings.index, columns=self.ratings.columns)
        
        return preds_df

# Exemplo de uso
# ratings_df = pd.DataFrame(...)  # Substitua por sua matriz de classificações
# rec_sys_svd = RecSysSVD(k=50, ratings=ratings_df)
# predictions = rec_sys_svd.fit_model()
