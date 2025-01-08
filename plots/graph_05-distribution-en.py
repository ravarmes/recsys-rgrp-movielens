import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Função para ler o arquivo Excel e preparar os dados.
def ler_dados(arquivo):
    dados = pd.read_excel(arquivo, header=1, index_col=0)
    return dados

# Função para transformar valores inteiros em contínuos usando uma distribuição normal
def transformar_continuo_normal(valores, escala=0.25):
    valores_continuos = []
    for valor in valores:
        valor_continuo = np.random.normal(loc=valor, scale=escala)
        valor_continuo = np.clip(valor_continuo, 1, 5)  # Garante que os valores estejam entre 1 e 5
        valores_continuos.append(valor_continuo)
    return np.array(valores_continuos)

# Arquivos Excel
arquivo_movielens_X = 'X_RecSysALS.xlsx'
arquivo_movielens_Xest_ALS = 'X_est_RecSysALS.xlsx'
arquivo_movielens_Xest_NMF = 'X_est_RecSysNMF.xlsx'
arquivo_movielens_Xest_KNN = 'X_est_RecSysKNN.xlsx'

# Ler os dados
dados_movielens_X = ler_dados(arquivo_movielens_X)
dados_movielens_Xest_ALS = ler_dados(arquivo_movielens_Xest_ALS)
dados_movielens_Xest_NMF = ler_dados(arquivo_movielens_Xest_NMF)
dados_movielens_Xest_KNN = ler_dados(arquivo_movielens_Xest_KNN)

# Criar máscara para dados válidos do arquivo 1
mascara_movielens = np.isfinite(dados_movielens_X.values)

# Aplicar máscara aos arquivos
valores_movielens_X = dados_movielens_X.values[mascara_movielens].flatten()
valores_movielens_Xest_ALS = dados_movielens_Xest_ALS.values[mascara_movielens].flatten()
valores_movielens_Xest_NMF = dados_movielens_Xest_NMF.values[mascara_movielens].flatten()
valores_movielens_Xest_KNN = dados_movielens_Xest_KNN.values[mascara_movielens].flatten()

# Transformar valores inteiros em contínuos usando distribuição normal
valores_movielens_X_continuos = transformar_continuo_normal(valores_movielens_X, escala=0.5)
valores_movielens_Xest_ALS_continuos = transformar_continuo_normal(valores_movielens_Xest_ALS, escala=0.5)
valores_movielens_Xest_NMF_continuos = transformar_continuo_normal(valores_movielens_Xest_NMF, escala=0.5)
valores_movielens_Xest_KNN_continuos = transformar_continuo_normal(valores_movielens_Xest_KNN, escala=0.5)

# Criar a figura e os subplots
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Comparative Density Plot', fontsize=16)

# Plot de densidade para cada conjunto de dados
sns.kdeplot(valores_movielens_X_continuos, color='blue', linestyle='-', label='valores_movielens_X_continuos', ax=axs[0, 0], fill=True, alpha=0.05)
sns.kdeplot(valores_movielens_Xest_ALS_continuos, color='red', linestyle='-', label='valores_movielens_Xest_ALS_continuos', ax=axs[0, 0], fill=True, alpha=0.05)
sns.kdeplot(valores_movielens_Xest_NMF_continuos, color='green', linestyle='--', label='valores_movielens_Xest_NMF_continuos', ax=axs[0, 0], fill=True, alpha=0.05)
sns.kdeplot(valores_movielens_Xest_KNN_continuos, color='black', linestyle='--', label='valores_movielens_Xest_KNN_continuos', ax=axs[0, 0], fill=True, alpha=0.05)

# Adicionar títulos e rótulos aos eixos
axs[0, 0].set_title('Comparação de Densidade (Movielens X e Estimativas)', fontsize=14)
axs[0, 0].set_xlabel('Valor', fontsize=12)
axs[0, 0].set_ylabel('Densidade', fontsize=12)

# Adicionar a legenda
axs[0, 0].legend(title="Métodos", fontsize=10)

# Ajustar o layout para que as legendas e títulos não se sobreponham
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Mostrar o gráfico
plt.show()
