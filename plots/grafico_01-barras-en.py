import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})  # Ajusta o tamanho da fonte globalmente

# Dados dos agrupamentos e suas respectivas injustiças para cada estratégia
agrupamentos = ['Gender', 'Age', 'Evaluations', 'Agglomerative']
R_grp_ALS = [0.00426530, 0.00170270, 0.00129660, 0.00615080]
R_grp_NMF = [0.00301780, 0.00161200, 0.00400160, 0.00487470]
R_grp_KNN = [0.00053500, 0.00767110, 0.00195270, 0.00304140]

# Cores para cada agrupamento
# cores = ['blue', 'green', 'red', 'purple']
cores = ['#17becf', '#9467bd', '#ff7f0e', '#2ca02c']

# Criando a figura e os eixos para os gráficos
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Função para adicionar gráficos
def adicionar_grafico(ax, dados, titulo, mostrar_ylabel):
    ax.bar(range(len(agrupamentos)), dados, color=cores)
    ax.set_title(titulo, fontsize=14)
    ax.set_xticks([])  # Remove os rótulos do eixo X
    if mostrar_ylabel:
        ax.set_ylabel('Group Unfairness', fontsize=14)
    ax.set_ylim(0.000, 0.008)  # Define os limites do eixo Y para todos os gráficos

# Adicionando gráficos
adicionar_grafico(axs[0], R_grp_ALS, 'ALS', True)
adicionar_grafico(axs[1], R_grp_NMF, 'NMF', False)
adicionar_grafico(axs[2], R_grp_KNN, 'KNN', False)

# Preparando a legenda
handles = [plt.Rectangle((0,0),1,1, color=cor) for cor in cores]
labels = agrupamentos

# Ajustando o layout dos subplots para criar espaço para a legenda
plt.tight_layout(rect=[0, 1, 1, 0.88])

# Adicionando a legenda centralizada abaixo dos gráficos
fig.legend(handles, labels, loc='upper center', ncol=len(agrupamentos), fontsize='large', bbox_to_anchor=(0.5, 0.10))

plt.show()
