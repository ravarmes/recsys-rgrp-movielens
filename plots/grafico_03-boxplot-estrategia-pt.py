import matplotlib.pyplot as plt
import numpy as np

# Dados de injustiça para cada estratégia e configuração de agrupamento
dados = {
    'ALS': [0.00426530, 0.00170270, 0.00129660, 0.00615080],
    'NMF': [0.00301780, 0.00161200, 0.00400160, 0.00487470],
    'KNN': [0.00053500, 0.00767110, 0.00195270, 0.00304140]
}

# Convertendo dados para formato adequado para boxplot
dados_boxplot = [dados['ALS'], dados['NMF'], dados['KNN']]

# Nomes para os eixos X
estrategias = ['ALS', 'NMF', 'KNN']

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
