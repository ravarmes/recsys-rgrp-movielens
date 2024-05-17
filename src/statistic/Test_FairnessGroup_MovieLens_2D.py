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

df.rename(columns={'Gender': 'Gênero', 'Age': 'Idade', 'NR': 'Avaliações', 'Loss': 'Perda'}, inplace=True)

print(df)



# Gráfico de Dispersão Idade vs. Perda
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Idade', y='Perda', data=df)
plt.title('Gráfico de Dispersão: Idade vs. Perda')
plt.xlabel('Idade')
plt.ylabel('Perda')
plt.show()

# Gráfico de Dispersão Avaliações vs. Perda
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Avaliações', y='Perda', data=df)
plt.title('Gráfico de Dispersão: Avaliações vs. Perda')
plt.xlabel('Avaliações')
plt.ylabel('Perda')
plt.show()
