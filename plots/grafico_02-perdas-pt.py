import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
data = {
    "ALS | Gênero | 0.0042653": {
        "Masculino": 0.7406300,
        "Feminino": 0.8712488
    },
    "ALS | Idade | 0.0017027": {
        "00": 0.8355848,
        "18": 0.8093482,
        "25": 0.7505571,
        "35": 0.7775164,
        "45": 0.7509080,
        "50": 0.6974159,
        "56": 0.7716997
    },
    "ALS | Avaliações | 0.0012966": {
        "Favorecidos": 0.6991967,
        "Desfavorecidos": 0.7712141
    },
    "ALS | Aglomerativo | 0.0061508": {
        "G1": 0.7352054,
        "G2": 0.7408430,
        "G3": 0.8431107,
        "G4": 0.7617414,
        "G5": 0.9408125
    },
    "NMF | Gênero | 0.0030178": {
        "Masculino": 0.6741322,
        "Feminino": 0.7840013
    },
    "NMF | Idade | 0.0016120": {
        "00": 0.7717287,
        "18": 0.7343003,
        "25": 0.6766549,
        "35": 0.6940361,
        "45": 0.6967239,
        "50": 0.6588567,
        "56": 0.7630716
    },
    "NMF | Avaliações | 0.0040016": {
        "Favorecidos": 0.5782639,
        "Desfavorecidos": 0.7047802
    },
    "NMF | Aglomerativo | 0.0048747": {
        "G1": 0.6753650,
        "G2": 0.6954240,
        "G3": 0.7655907,
        "G4": 0.6309879,
        "G5": 0.8285309
    },
    "KNN | Gênero | 0.0005350": {
        "Masculino": 1.0599420,
        "Feminino": 1.1062011
    },
    "KNN | Idade | 0.0076711": {
        "00": 1.1206043,
        "18": 1.2151433,
        "25": 1.0720398,
        "35": 0.9772514,
        "45": 0.9611199,
        "50": 1.1039704,
        "56": 0.9748200
    },
    "KNN | Avaliações | 0.0019527": {
        "Favorecidos": 0.9870601,
        "Desfavorecidos": 1.0754399
    },
    "KNN | Aglomerativo | 0.0030414": {
        "G1": 1.1109219,
        "G2": 0.9779038,
        "G3": 1.1163396,
        "G4": 1.0125284,
        "G5": 1.0811366
    }
}

# Definindo os títulos dos subplots
titles = [
    "ALS - Gênero",
    "ALS - Idade",
    "ALS - Avaliações",
    "ALS - Aglomerativo",
    "NMF - Gênero",
    "NMF - Idade",
    "NMF - Avaliações",
    "NMF - Aglomerativo",
    "KNN - Gênero",
    "KNN - Idade",
    "KNN - Avaliações",
    "KNN - Aglomerativo"
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
        ax.set_ylabel('Perda de Grupo')
    else:
        ax.set_yticklabels([])  # Remove os rótulos do eixo y para os outros subplots
    ax.set_ylim(0, 1.23)  # Definindo a escala do eixo y
    ax.set_title(titles[i])

# Ajustando layout
plt.tight_layout()
plt.show()
