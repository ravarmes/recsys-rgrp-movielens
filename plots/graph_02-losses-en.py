import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
data = {
    "ALS | Activity | 0.0012966": {
        "advantaged_group": 0.5072996,
        "disadvantaged_group": 0.5579151
    },
    "ALS | Age | 0.0017027": {
        "00": 0.5988816,
        "18": 0.5765943,
        "25": 0.5545493,
        "35": 0.5379932,
        "45": 0.5287662,
        "50": 0.5270961,
        "56": 0.5724602
    },
    "ALS | Gender | 0.0042653": {
        "Male": 0.5414160,
        "Female": 0.5988678
    },
    "ALS | Agglomerative | 0.0061508": {
        "G1": 0.5988678,
        "G2": 0.5165295,
        "G3": 0.5257680,
        "G4": 0.5672233,
        "G5": 0.5356081
    },
    "NCF | Activity | 0.0040016": {
        "advantaged_group": 0.6736042,
        "disadvantaged_group": 0.7053388
    },
    "NCF | Age | 0.0016120": {
        "00": 0.7503187,
        "18": 0.7484501,
        "25": 0.6980305,
        "35": 0.6789333,
        "45": 0.6721374,
        "50": 0.6790036,
        "56": 0.7068404
    },
    "NCF | Gender | 0.0030178": {
        "Male": 0.6834183,
        "Female": 0.7557694
    },
    "NCF | Agglomerative | 0.0048747": {
        "G1": 0.7597630,
        "G2": 0.6992425,
        "G3": 0.6409658,
        "G4": 0.7004714,
        "G5": 0.6793387
    },
    "CBF | Activity | 0.0019527": {
        "advantaged_group": 0.6958071,
        "disadvantaged_group": 0.6275687
    },
    "CBF | Age | 0.0076711": {
        "00": 0.6896335,
        "18": 0.6388872,
        "25": 0.6278901,
        "35": 0.5693930,
        "45": 0.5658008,
        "50": 0.5538033,
        "56": 0.5680818
    },
    "CBF | Gender | 0.0005350": {
        "Male": 0.6217455,
        "Female": 0.6536329
    },
    "CBF | Agglomerative | 0.0030414": {
        "G1": 0.6515506,
        "G2": 0.6689123,
        "G3": 0.5439165,
        "G4": 0.6351156,
        "G5": 0.6531393
    }
}

# Definindo os títulos dos subplots
titles = [
    "ALS - Activity",
    "ALS - Age",
    "ALS - Gender",
    "ALS - Agglomerative",
    "NCF - Activity",
    "NCF - Age",
    "NCF - Gender",
    "NCF - Agglomerative",
    "CBF - Activity",
    "CBF - Age",
    "CBF - Gender",
    "CBF - Agglomerative"
]

# Paletas de cores
cmap_1 = plt.cm.get_cmap('Blues')
cmap_2 = plt.cm.get_cmap('Purples')
cmap_3 = plt.cm.get_cmap('Oranges')
cmap_4 = plt.cm.get_cmap('Greens')

# Mapeando cores para os subplots
colors_1 = cmap_1(np.linspace(0.3, 0.7, 2))  # Para 2 grupos
colors_2 = cmap_2(np.linspace(0.3, 0.7, 7))  # Para 7 grupos
colors_3 = cmap_3(np.linspace(0.3, 0.7, 2))  # Para 2 grupos
colors_4 = cmap_4(np.linspace(0.3, 0.7, 5))  # Para 5 grupos

# Criando os subplots com altura ajustada
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
fig.subplots_adjust(left=0.314, bottom=0.23, right=0.993, top=0.945, wspace=0.463, hspace=0.451)

# Iterando sobre os dados e os subplots
for i, (alg, groups) in enumerate(data.items()):
    ax = axs[i // 4, i % 4]
    if i % 4 == 0:
        colors = colors_1
    elif i % 4 == 1:
        colors = colors_2
    elif i % 4 == 2:
        colors = colors_3
    else:
        colors = colors_4

    for j, (group, loss) in enumerate(groups.items()):
        ax.bar(group, loss, color=colors[j % len(colors)])
    if i in [0, 4, 8]:  # Apenas para os subplots 1, 5 e 9
        ax.set_ylabel('Group Loss')
    else:
        ax.set_yticklabels([])  # Remove os rótulos do eixo y para os outros subplots
    ax.set_ylim(0, 0.72)  # Definindo a escala do eixo y
    ax.set_title(titles[i])

# Ajustando layout
plt.tight_layout()
plt.show()
