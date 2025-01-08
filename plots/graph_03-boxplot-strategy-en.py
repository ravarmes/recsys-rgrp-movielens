import matplotlib.pyplot as plt
import numpy as np

# Dados de injustiça para cada estratégia e configuração de agrupamento
dados = {
    'ALS': [0.0006405, 0.0006335, 0.0008252, 0.0009184],
    'NCF': [0.0002518, 0.0009184, 0.0013087, 0.0014805],
    'CBF': [0.0011641, 0.0022225, 0.0002542, 0.0019898]
}

# Convertendo dados para formato adequado para boxplot
dados_boxplot = [dados['ALS'], dados['NCF'], dados['CBF']]

# Nomes para os eixos X
estrategias = ['ALS', 'NCF', 'CBF']

# Criando o gráfico de caixa
plt.figure(figsize=(10, 6))

# Personalizando as cores
colors = ['skyblue', 'salmon', 'lightgreen']

# Criando o boxplot com cores personalizadas
plt.boxplot(dados_boxplot, labels=estrategias, patch_artist=True, showmeans=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none'),
            meanprops=dict(marker='s', markerfacecolor='green', markersize=10))

plt.xticks(fontsize=12)

# Adicionando título e rótulos aos eixos
plt.title('Distribution of Unfairness by Filtering Strategy', fontsize=14)
plt.ylabel('Group Unfairness', fontsize=14)
plt.xlabel('Filtering Strategy', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrando o gráfico
plt.tight_layout()
plt.show()
