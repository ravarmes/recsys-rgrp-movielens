import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Dados
categorias = ['Activity', 'Age', 'Gender', 'Agglomerative']
estrategias = ['ALS', 'NCF', 'CBF']
dados = [
    [0.0006405, 0.0006335, 0.0008252, 0.0009184],  # ALS
    [0.0002518, 0.0009184, 0.0013087, 0.0014805],  # NCF
    [0.0011641, 0.0022225, 0.0002542, 0.0019898]   # CBF
]

# Cores personalizadas para o subplot 1
colors_sub1 = ['#8c564b', '#e377c2', '#bcbd22', '#7f7f7f']  # Cores diferentes das do subplot 2

# Cores personalizadas para o subplot 2
colors_sub2 = ['#17becf', '#9467bd', '#ff7f0e', '#2ca02c']

# Número de variáveis
num_vars = len(categorias)

# Ângulos
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

show_ticks = [0.0005, 0.0010]

# Primeiro gráfico de radar
for i, estrategia in enumerate(dados):
    axs[0].plot(angles, estrategia + estrategia[:1], label=estrategias[i], color=colors_sub1[i])
    axs[0].fill(angles, estrategia + estrategia[:1], color=colors_sub1[i], alpha=0.1)
axs[0].set_xticks(angles[:-1])
axs[0].set_xticklabels(categorias, fontsize=12)  # Aumentando o tamanho da fonte das categorias
axs[0].set_title('Perspective by Filtering Strategies', y=1.1, fontsize=14)
axs[0].legend(loc='lower right', fontsize=10)  # Aumentando o tamanho da fonte da legenda

axs[0].set_yticks(show_ticks)
axs[0].set_yticklabels([f'{tick:.4f}' for tick in show_ticks], fontsize=12)  # Configurando rótulos dos ticks

# Segundo gráfico de radar - Inversão de perspectiva
# Preparando dados para o segundo gráfico
dados_invertidos = np.array(dados).T  # Transpõe a matriz de dados
dados_invertidos = np.concatenate((dados_invertidos, dados_invertidos[:, 0:1]), axis=1)  # Fecha o círculo

angles_invertidos = np.linspace(0, 2 * np.pi, len(estrategias), endpoint=False).tolist()
angles_invertidos += angles_invertidos[:1]

for i, grupo in enumerate(categorias):
    axs[1].plot(angles_invertidos, dados_invertidos[i], label=grupo, color=colors_sub2[i])
    axs[1].fill(angles_invertidos, dados_invertidos[i], color=colors_sub2[i], alpha=0.1)

axs[1].set_xticks(angles_invertidos[:-1])
axs[1].set_xticklabels(estrategias, fontsize=12)  # Aumentando o tamanho da fonte das estratégias
axs[1].set_title('Perspective by User Groupings', y=1.1, fontsize=14)
axs[1].legend(loc='lower right', fontsize=10)  # Aumentando o tamanho da fonte da legenda

# Mostrar círculos em intervalos específicos
# Definindo os valores que queremos mostrar


# Configurando os valores no eixo radial
axs[1].set_yticks(show_ticks)
axs[1].set_yticklabels([f'{tick:.4f}' for tick in show_ticks], fontsize=12)  # Configurando rótulos dos ticks

plt.tight_layout()
plt.show()
