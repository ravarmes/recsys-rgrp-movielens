import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Dados
categorias = ['Gênero', 'Idade', 'Avaliações', 'Aglomerativo']
estrategias = ['ALS', 'NMF', 'KNN']
dados = [
    [0.00426530, 0.00170270, 0.00129660, 0.00615080],  # ALS
    [0.00301780, 0.00161200, 0.00400160, 0.00487470],  # NMF
    [0.00053500, 0.00767110, 0.00195270, 0.00304140]   # KNN
]

# Número de variáveis
num_vars = len(categorias)

# Ângulos
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

# Primeiro gráfico de radar
for i, estrategia in enumerate(dados):
    axs[0].plot(angles, estrategia + estrategia[:1], label=estrategias[i])
    axs[0].fill(angles, estrategia + estrategia[:1], alpha=0.1)
axs[0].set_xticks(angles[:-1])
axs[0].set_xticklabels(categorias, fontsize=12)  # Aumentando o tamanho da fonte das categorias
axs[0].set_title('Perspectiva por Estratégias de Filtragem', y=1.1, fontsize=14)
axs[0].legend(fontsize=10)  # Aumentando o tamanho da fonte da legenda

# Segundo gráfico de radar - Inversão de perspectiva
# Preparando dados para o segundo gráfico
dados_invertidos = np.array(dados).T  # Transpõe a matriz de dados
dados_invertidos = np.concatenate((dados_invertidos, dados_invertidos[:, 0:1]), axis=1)  # Fecha o círculo

angles_invertidos = np.linspace(0, 2 * np.pi, len(estrategias), endpoint=False).tolist()
angles_invertidos += angles_invertidos[:1]

for i, grupo in enumerate(categorias):
    axs[1].plot(angles_invertidos, dados_invertidos[i], label=grupo)
    axs[1].fill(angles_invertidos, dados_invertidos[i], alpha=0.1)

axs[1].set_xticks(angles_invertidos[:-1])
axs[1].set_xticklabels(estrategias, fontsize=12)  # Aumentando o tamanho da fonte das estratégias
axs[1].set_title('Perspectiva por Agrupamentos de Usuários', y=1.1, fontsize=14)
axs[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10)  # Aumentando o tamanho da fonte da legenda

plt.tight_layout()
plt.show()
