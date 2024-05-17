import sys
sys.path.insert(0, './src/')

from RecSys import RecSys
from UserFairness import IndividualLossVariance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Leitura dos dados
Data_path = 'Data/MovieLens-1M'
n_users = 300
n_items = 1000
top_users = True
top_items = True

# Algoritmo de recomendação
algorithm = 'RecSysALS'

# Parâmetros para calcular as medidas de justiça
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, users_info, items_info = recsys.read_movielens_1M(n_users, n_items, top_users, top_items, data_dir=Data_path) 
omega = ~X.isnull()

X_est = recsys.compute_X_est(X, algorithm) 

ilv = IndividualLossVariance(X, omega, 1)
losses = ilv.get_losses(X_est)

# Criando o DataFrame com as informações dos usuários
df = pd.DataFrame(columns=['Gender', 'Age'])
df['Gender'] = users_info['Gender']
df['Age'] = users_info['Age']
df['NR'] = users_info['NR']
df['Loss'] = losses

df.dropna(subset=['Loss'], inplace=True)

# # Gráfico da Correlação
# plt.figure(figsize=(7, 7))
# corr = np.corrcoef(df.values, rowvar=False)
# sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', cbar=False, xticklabels=df.columns, yticklabels=df.columns)
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------

# Transformando os dados de gênero
df['Gender'] = df['Gender'].replace({1: 'M', 2: 'F'})

# Preparando os dados para o pairplot
df = df.drop(columns=['Loss'])
df.rename(columns={'Gender': 'Gênero', 'Age': 'Idade', 'NR': 'Avaliações'}, inplace=True)

# Criando o pairplot com colormap personalizado
palette = {"M": "blue", "F": "red"}
pairplot = sns.pairplot(df, hue='Gênero', palette=palette, diag_kind='hist', plot_kws={'s': 40})

# Mostrando a legenda
plt.legend()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------

# # Criando o gráfico 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Mapear os valores de Gênero para números
# df['Gênero'] = df['Gênero'].map({'M': 1, 'F': 2})

# # Plotando os pontos
# sc = ax.scatter(df['Gênero'], df['Idade'], df['Avaliações'], c=df['Gênero'], cmap='viridis', s=20)

# # Ajustando os rótulos dos eixos
# ax.set_xlabel('Gênero')
# ax.set_ylabel('Idade')
# ax.set_zlabel('Avaliações')

# # Definindo os ticks dos eixos para Gênero
# ax.set_xticks([1, 2])
# ax.set_xticklabels(['M', 'F'])

# # Adicionando a barra de cores
# cbar = plt.colorbar(sc, ax=ax, pad=0.1)
# cbar.set_ticks([1.5, 2.5])
# cbar.set_ticklabels(['M', 'F'])

# plt.title('Distribuição 3D de Gênero, Idade e Avaliações dos Usuários')
# plt.show()
