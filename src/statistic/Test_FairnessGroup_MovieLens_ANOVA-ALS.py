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

df['Gênero'] = pd.Categorical(df['Gênero'])


#----------------------------------------------------------------------------
model = ols('Perda ~ C(Gênero) * Idade * Avaliações', data=df).fit()
print(anova_lm(model, typ=2))
#----------------------------------------------------------------------------


# Opcional: Visualização
plt.figure(figsize=(10, 5))
sns.boxplot(x='Gênero', y='Perda', data=df)
plt.title('Distribuição da Perda por Gênero')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='Idade', y='Perda', data=df)
plt.title('Distribuição da Perda por Idade')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='Avaliações', y='Perda', data=df)
plt.title('Distribuição da Perda por Avaliações')
plt.show()

"""

# RESULTADOS

                               sum_sq     df          F    PR(>F)
C(Gênero)                    0.836162    1.0  10.985136  0.001034
Idade                        0.057676    1.0   0.757722  0.384757
C(Gênero):Idade              0.002209    1.0   0.029020  0.864850
Avaliações                   0.003289    1.0   0.043209  0.835476
C(Gênero):Avaliações         0.131325    1.0   1.725293  0.190044
Idade:Avaliações             0.052754    1.0   0.693058  0.405807
C(Gênero):Idade:Avaliações   0.050293    1.0   0.660730  0.416965
Residual                    22.226333  292.0        NaN       NaN

Os resultados obtidos são referentes a um teste de análise de variância (ANOVA) para avaliar a significância das variáveis e suas interações no modelo de regressão. Abaixo está a interpretação dos resultados:

- C(Gênero):
  - O valor de sum_sq (soma dos quadrados) para a variável categórica 'Gênero' é 0.836162.
  - O teste F calculado (F) para essa variável é 10.985136.
  - O valor PR(>F) é 0.001034, o que indica que a variável 'Gênero' tem um efeito significativo sobre a variável resposta 'Perda' (considerando um nível de significância de 0.05). Em outras palavras, há uma diferença estatisticamente significativa na "Perda" entre os diferentes gêneros.

- Idade e as outras variáveis:
  - As variáveis 'Idade', 'Avaliações', bem como as interações entre elas e com 'Gênero', não apresentam efeito significativo sobre a variável 'Perda'. Isso é indicado pelos valores de PR(>F) acima de 0.05.

- Residual:
  - A soma dos quadrados dos resíduos é 22.226333.

Em resumo, com base nos resultados da ANOVA, a variável categórica 'Gênero' possui uma influência estatisticamente significativa na variável 'Perda', enquanto as outras variáveis e suas interações não apresentam impacto significativo. Lembre-se de que a interpretação dos resultados deve sempre levar em consideração o contexto do estudo e a significância prática dos achados. Se precisar de mais alguma informação, estou à disposição para ajudar.


IMPORTANTE
ao analisar o valor de p (PR(>F)) de 0.190044 para a interação entre a variável categórica 'Gênero' e a variável contínua 'Avaliações', podemos dizer que este valor sugere que, a um nível de significância de 0.05, essa interação não é estatisticamente significativa para explicar a variação na variável resposta 'Perda'.

Embora o valor p de 0.190044 esteja acima do limite de significância comumente adotado (0.05), ele não é muito alto em comparação com outros valores em sua análise. Isso significa que, apesar de não atingir significância estatística a 0.05, pode haver um potencial efeito ou sinal de associação entre a variável 'Gênero' e 'Avaliações' sobre a variável 'Perda' que poderia ser considerado em uma investigação mais aprofundada.

Portanto, você pode mencionar que, embora a interação entre 'Gênero' e 'Avaliações' não tenha alcançado significância estatística, o valor do p está próximo do limite e merece uma análise mais detalhada ou, talvez, a inclusão de mais dados para investigar melhor essa relação. Sempre é importante considerar o contexto específico do seu estudo ao interpretar esses resultados.


"""