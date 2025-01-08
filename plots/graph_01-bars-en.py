import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})  # Ajusta o tamanho da fonte globalmente

# Dados dos agrupamentos e suas respectivas injustiças para cada estratégia
agrupamentos = ['Activity', 'Age', 'Gender', 'Agglomerative']
R_grp_ALS = [0.0006405, 0.0006335, 0.0008252, 0.0009184]
R_grp_NCF = [0.0002518, 0.0009184, 0.0013087, 0.0014805]
R_grp_CB = [0.0011641, 0.0022225, 0.0002542, 0.0019898]

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
    ax.set_ylim(0.000, 0.0025)  # Define os limites do eixo Y para todos os gráficos

# Adicionando gráficos
adicionar_grafico(axs[0], R_grp_ALS, 'ALS', True)
adicionar_grafico(axs[1], R_grp_NCF, 'NCF', False)
adicionar_grafico(axs[2], R_grp_CB, 'CBF', False)

# Preparando a legenda
handles = [plt.Rectangle((0,0),1,1, color=cor) for cor in cores]
labels = agrupamentos

# Ajustando o layout dos subplots para criar espaço para a legenda
plt.tight_layout(rect=[0, 1, 1, 0.88])

# Adicionando a legenda centralizada abaixo dos gráficos
fig.legend(handles, labels, loc='upper center', ncol=len(agrupamentos), fontsize='large', bbox_to_anchor=(0.5, 0.10))

plt.show()
