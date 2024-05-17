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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Padronizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Idade', 'Avaliações', 'Perda']])

# PCA com 2 componentes
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

# Criar DataFrame com os componentes principais
df_pca = pd.DataFrame(data=principal_components, columns=['Componente Principal 1', 'Componente Principal 2'])

# Plotar gráfico dos componentes principais
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Componente Principal 1', y='Componente Principal 2', data=df_pca)
plt.title('Análise de Componentes Principais')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
