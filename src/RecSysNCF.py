import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from math import sqrt

class RecSysNCF:
    def __init__(self, n_users, n_items, n_factors, ratings=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        self.ratings = ratings

    def fit_model(self, epochs=5, batch_size=64):
        # Preparação dos dados: assume que self.ratings já está na forma de matriz usuários x itens
        ratings_matrix = self.ratings.to_numpy()
        known_ratings_mask = ~np.isnan(ratings_matrix)
        user_ids, item_ids = np.where(known_ratings_mask)
        ratings_values = ratings_matrix[known_ratings_mask]

        # Divisão dos dados em treinamento e teste
        user_ids_train, user_ids_test, item_ids_train, item_ids_test, ratings_train, ratings_test = train_test_split(
            user_ids, item_ids, ratings_values, test_size=0.2, random_state=42)

        # Definição do modelo NCF
        user_input = Input(shape=(1,), dtype='int32')
        item_input = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(input_dim=self.n_users, output_dim=self.n_factors)(user_input)
        item_embedding = Embedding(input_dim=self.n_items, output_dim=self.n_factors)(item_input)
        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)
        concat = Concatenate()([user_vec, item_vec])
        dense = Dense(128, activation='relu')(concat)
        predictions = Dense(1)(dense)
        model = Model(inputs=[user_input, item_input], outputs=predictions)
        model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Treinamento do modelo
        model.fit([user_ids_train, item_ids_train], ratings_train, epochs=epochs, batch_size=batch_size, validation_data=([user_ids_test, item_ids_test], ratings_test))

        # Geração de previsões para a matriz completa
        all_user_ids = np.array(range(self.n_users)).repeat(self.n_items)
        all_item_ids = np.tile(np.array(range(self.n_items)), self.n_users)
        all_predictions = model.predict([all_user_ids, all_item_ids]).flatten()

        # Construção do DataFrame de previsões com a mesma estrutura que self.ratings
        predictions_matrix = all_predictions.reshape(self.n_users, self.n_items)
        X_est = pd.DataFrame(predictions_matrix, index=self.ratings.index, columns=self.ratings.columns)

        # Cálculo do RMSE usando as avaliações conhecidas
        # Primeiro, extraímos as previsões correspondentes às avaliações conhecidas do DataFrame original 'ratings'
        predictions_known = predictions_matrix[known_ratings_mask]

        # Agora, calculamos o RMSE
        error = sqrt(mean_squared_error(ratings_values, predictions_known))

        return X_est, error
