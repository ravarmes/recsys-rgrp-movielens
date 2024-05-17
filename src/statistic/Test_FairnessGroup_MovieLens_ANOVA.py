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


# Leitura dos dados
Data_path = 'Data/MovieLens-1M'
n_users = 300
n_items = 1000
top_users = True
top_items = True

# Algoritmos de recomendação
algorithms = ['RecSysALS', 'RecSysNMF', 'RecSysKNN']

for algorithm in algorithms:

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

  df['Gênero'] = pd.Categorical(df['Gênero'])


  print(f"\nAlgoritmo: {algorithm}")
  #----------------------------------------------------------------------------
  model = ols('Perda ~ C(Gênero) * Idade * Avaliações', data=df).fit()
  print(anova_lm(model, typ=2))
  #----------------------------------------------------------------------------


  # Opcional: Visualização
  # plt.figure(figsize=(10, 5))
  # sns.boxplot(x='Gênero', y='Perda', data=df)
  # plt.title('Distribuição da Perda por Gênero')
  # plt.show()

  # plt.figure(figsize=(10, 5))
  # sns.boxplot(x='Idade', y='Perda', data=df)
  # plt.title('Distribuição da Perda por Idade')
  # plt.show()

  # plt.figure(figsize=(10, 5))
  # sns.boxplot(x='Avaliações', y='Perda', data=df)
  # plt.title('Distribuição da Perda por Avaliações')
  # plt.show()

  """

  # RESULTADOS

Esses resultados se referem a análises de variância (ANOVA) para diferentes algoritmos de recomendação (RecSysALS, RecSysNMF e RecSysKNN) em relação às variáveis 'Gênero', 'Idade', 'Avaliações' e suas interações. Vamos interpretar os principais pontos:

**Para o Algoritmo RecSysALS:**
- A variável 'Gênero' tem um efeito significativo sobre a variável resposta em um nível de significância de 5% (p=0.001034), indicando que o gênero dos usuários influencia a 'Perda'.
- As variáveis 'Idade', 'Avaliações' e suas interações com 'Gênero' não são estatisticamente significativas, pois seus valores de p são maiores que 0.05.

**Para o Algoritmo RecSysNMF:**
- Novamente, a variável 'Gênero' mostra um efeito significativo sobre a 'Perda' com p=0.003199, indicando que o gênero influencia a 'Perda'.
- As outras variáveis ('Idade', 'Avaliações' e suas interações) não são estatisticamente significativas, pois seus valores de p são maiores que 0.05.

**Para o Algoritmo RecSysKNN:**
- Aqui, a variável 'Idade' é significativa com p=0.003133, mostrando que a idade tem um impacto na 'Perda'.
- As variáveis 'Gênero', 'Avaliações' e a interação 'Gênero:Avaliações' não são estatisticamente significativas, com valores de p acima de 0.05.

**Observações:**
- Os resultados indicam que o impacto das variáveis varia de acordo com o algoritmo de recomendação utilizado.
- Os valores de p (PR(>F)) são fundamentais para determinar a significância estatística. Um valor baixo (geralmente abaixo de 0.05) sugere significância.

Lembre-se de que a interpretação dos resultados deve ser feita com base no conhecimento do domínio e do contexto específico do estudo. Se precisar de mais detalhes ou tiver outras dúvidas, estou à disposição para ajudar.

  """