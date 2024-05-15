import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from abc import ABCMeta, abstractmethod

class RecSysNMF:
    __metaclass__ = ABCMeta

    def __init__(self, n_components, ratings=None):
        self.n_components = n_components
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        self.ratings = ratings

    def fit_model(self, max_iter=200, tol=1e-4):
        # Inicializa o modelo NMF
        model = NMF(n_components=self.n_components, max_iter=max_iter, tol=tol, init='random')
        
        # A matriz de classificações pode conter NaNs, que não podem ser processados diretamente pelo NMF.
        # Uma abordagem comum é preencher esses NaNs com a média das classificações conhecidas.
        ratings_filled = self.ratings.fillna(self.ratings.mean())
        
        # Ajusta o modelo NMF à matriz de classificações
        W = model.fit_transform(ratings_filled)
        H = model.components_
        
        # Calcula a matriz de predição como o produto das matrizes fatoradas
        pred = pd.DataFrame(np.dot(W, H), index=self.ratings.index, columns=self.ratings.columns)
        
        # Armazena os componentes para uso posterior
        self.W = W
        self.H = H
        
        # Retorna a matriz de predição
        return pred

# Exemplo de uso
# ratings_df = pd.DataFrame(...)  # Substitua por sua matriz de classificações
# rec_sys_nmf = RecSysNMF(n_components=5, ratings=ratings_df)
# predictions = rec_sys_nmf.fit_model()
