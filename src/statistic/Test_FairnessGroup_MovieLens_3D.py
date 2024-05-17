import sys
sys.path.insert(0, './src/')

from RecSys import RecSys
from UserFairness import IndividualLossVariance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import MinMaxScaler

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

df.rename(columns={'Gender': 'Gênero', 'Age': 'Idade', 'NR': 'Avaliações', 'Loss': 'Perdas'}, inplace=True)

print(df)

from mpl_toolkits.mplot3d import Axes3D

# Gráfico de Dispersão 3D: Idade, Avaliações e Perda
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Idade'], df['Avaliações'], df['Perdas'], c='blue', marker='o')

ax.set_xlabel('Idade')
ax.set_ylabel('Avaliações')
ax.set_zlabel('Perdas')

plt.title('Gráfico de Dispersão 3D: Idade, Avaliações e Perdas')
plt.show()
