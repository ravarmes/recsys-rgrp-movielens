import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer


class RecSysCBF:
    def __init__(self, k, ratings=None, user_based=True, movie_file=None, regularization=1e-4, alpha=1.0):
        self.k = k
        self.user_based = user_based
        self.movie_file = movie_file
        self.regularization = regularization  # Regularização padrão
        self.alpha = alpha  # Exponente para ajustar a similaridade
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        # Imputação baseada em KNN para valores ausentes
        imputer = KNNImputer(n_neighbors=self.k)
        self.ratings = pd.DataFrame(
            imputer.fit_transform(ratings),
            index=ratings.index,
            columns=ratings.columns
        )
        self.num_of_known_ratings_per_user = (~ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~ratings.isnull()).sum(axis=0)

    def load_movie_genres(self):
        movies = pd.read_csv(
            self.movie_file,
            sep='::',
            header=None,
            names=['Title', 'Genres'],
            engine='python',
            encoding='latin1'
        )
        movies['Genres'] = movies['Genres'].str.split('|')
        genres = list(set(genre for sublist in movies['Genres'] for genre in sublist))
        genre_matrix = pd.DataFrame(0, index=movies['Title'], columns=genres)
        for _, row in movies.iterrows():
            genre_matrix.loc[row['Title'], row['Genres']] = 1
        self.genre_matrix = genre_matrix

    # def get_similarity_matrix(self):
    #     if self.genre_matrix is None:
    #         raise ValueError("A matriz de gêneros não foi carregada.")
        
    #     # Preenche valores ausentes com 0
    #     genre_matrix_filled = self.genre_matrix.fillna(0)
        
    #     # print(f"Genre Matrix (Filled):\n{genre_matrix_filled.head()}")  # Verifique os dados
        
    #     # Aplicando a regularização na matriz de gêneros
    #     # regularized_genre_matrix = genre_matrix_filled + self.regularization
    #     regularized_genre_matrix = self.genre_matrix.fillna(0) * (1 + self.regularization)

        
    #     # print(f"Regularized Genre Matrix:\n{regularized_genre_matrix.head()}")  # Verifique os dados
        
    #     # # Calculando a similaridade com o parâmetro alpha
    #     # similarity = pd.DataFrame(
    #     #     cosine_similarity(regularized_genre_matrix) ** self.alpha,
    #     #     index=self.genre_matrix.index,
    #     #     columns=self.genre_matrix.index
    #     # )
    #     similarity = pd.DataFrame(
    #         cosine_similarity(regularized_genre_matrix) * self.alpha,  # Multiplica por alpha
    #         index=self.genre_matrix.index,
    #         columns=self.genre_matrix.index
    #     )
    #     return similarity


    def get_similarity_matrix(self):
        if self.genre_matrix is None:
            raise ValueError("A matriz de gêneros não foi carregada.")
        
        # Preenche valores ausentes com 0
        genre_matrix_filled = self.genre_matrix.fillna(0)
        # regularized_genre_matrix = genre_matrix_filled + self.regularization
        
        regularized_genre_matrix = (1 - self.alpha) * self.genre_matrix + self.alpha * np.random.rand(*self.genre_matrix.shape)
        
        # random_values = np.random.uniform(low=-1, high=1, size=self.genre_matrix.shape)
        # regularized_genre_matrix = (1 - self.alpha) * self.genre_matrix + self.alpha * random_values
        
        # random1 = np.random.rand(*self.genre_matrix.shape)
        # random2 = np.random.normal(loc=0, scale=1, size=self.genre_matrix.shape)
        # combined_random = (random1 + random2) / 2
        # regularized_genre_matrix = (1 - self.alpha) * self.genre_matrix + self.alpha * combined_random

        # random_values = np.random.rand(*self.genre_matrix.shape) ** 2  # Quadrado para enviesar os valores
        # regularized_genre_matrix = (1 - self.alpha) * self.genre_matrix + self.alpha * random_values


        # # Calculando a similaridade
        # similarity = pd.DataFrame(
        #     cosine_similarity(genre_matrix_filled),
        #     index=self.genre_matrix.index,
        #     columns=self.genre_matrix.index
        # )

        similarity = pd.DataFrame(
            cosine_similarity(regularized_genre_matrix) ** (1 / self.alpha),  # Potencia inversa
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
        )


        
        # Ajustando a regularização com base em alpha
        adjusted_regularization = self.regularization * (1 + self.alpha)
        similarity = similarity + adjusted_regularization
        
        return similarity





    def knn_filtering(self, similarity):
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k + 1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        similarity = similarity.div(similarity.sum(axis=1) + 1e-8, axis=0)
        return similarity

    def fit_model(self):
        if self.movie_file is None:
            raise ValueError("O arquivo de filmes não foi fornecido.")

        self.load_movie_genres()
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)
        knn_similarity = knn_similarity.reindex(columns=self.ratings.columns, index=self.ratings.columns)

        pred_raw = self.ratings.fillna(0).dot(knn_similarity)
        sum_similarities = knn_similarity.sum(axis=0)
        pred = pred_raw.copy()

        for i in range(len(pred.columns)):
            if sum_similarities[i] > 0:
                pred.iloc[:, i] = pred.iloc[:, i] / sum_similarities[i]

        # Clipping entre os valores 1 e 5
        pred = pred.clip(lower=1, upper=5)

        # Pós-processamento: normalização com base na média e desvio padrão por usuário
        avg_ratings_user = self.ratings.mean(axis=1, skipna=True)
        std_ratings_user = self.ratings.std(axis=1, skipna=True)
        for user in self.ratings.index:
            pred.loc[user] = (pred.loc[user] - avg_ratings_user[user]) / (std_ratings_user[user] + 1e-8)
            pred.loc[user] = (pred.loc[user] * 0.5) + avg_ratings_user[user]

        self.pred = pred.fillna(0)
        self.U = self.ratings
        self.V = knn_similarity

        return self.pred

    # def fit_model(self):
    #     # Calcula a matriz de similaridade
    #     similarity = self.get_similarity_matrix()  

    #     # Calcula as predições
    #     predictions = (similarity.dot(self.utility_matrix)) ** self.alpha

    #     # Opcional: Normalizar para evitar explosão de valores
    #     predictions = predictions / predictions.max()

    #     return predictions

