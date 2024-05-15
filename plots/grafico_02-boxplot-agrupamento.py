import matplotlib.pyplot as plt
import numpy as np

# Dados de injustiça para cada estratégia e configuração de agrupamento
dados = {
    'ALS': [0.00426530, 0.00170270, 0.00129660, 0.00569490],
    'NMF': [0.00301780, 0.00161200, 0.00400160, 0.00487470],
    'KNN': [0.00053500, 0.00767110, 0.00195270, 0.00712780]
}

# Preparando dados para comparação por agrupamentos
dados_agrupamentos = []
agrupamentos = ['Gênero', 'Idade', 'Avaliações', 'Aglomerativo']

# Reorganizando os dados
for i in range(len(agrupamentos)):
    injusticas_agrupamento = [dados['ALS'][i], dados['NMF'][i], dados['KNN'][i]]
    dados_agrupamentos.append(injusticas_agrupamento)

# Nomes para os eixos X
estrategias = ['ALS', 'NMF', 'KNN']

# Criando o gráfico de caixa
plt.figure(figsize=(10, 6))
plt.boxplot(dados_agrupamentos, labels=agrupamentos)

# Adicionando título e rótulos aos eixos
plt.title('Comparação de Injustiça por Agrupamento', fontsize=14)
plt.ylabel('Injustiça do Grupo', fontsize=12)
plt.xlabel('Agrupamentos de Usuários', fontsize=12)

# Mostrando o gráfico
plt.tight_layout()
plt.show()
