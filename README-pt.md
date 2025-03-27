<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-rgrp-movielens/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  Análise de Justiça de Grupo no Dataset MovieLens
</h3>

<p align="center">Exemplo de agrupamentos utilizando medidas de justiça social </p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-rgrp-movielens?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-rgrp-movielens/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-rgrp-movielens?style=social">
  </a>
</p>

<p align="center">
  <a href="#-sobre">Sobre o projeto</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-instalacao">Instalação</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-arquivos">Arquivos</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-datasets">Datasets</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">Licença</a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

Este estudo investiga a equidade em sistemas de recomendação utilizando o dataset MovieLens, aplicando estratégias de filtragem colaborativa: ALS, KNN e NMF. Avaliamos a injustiça em diferentes configurações de agrupamento: Gênero, Idade, Avaliações e Aglomerativo. Os resultados indicam variações significativas de injustiça entre as estratégias, com o método Aglomerativo destacando-se por apresentar os maiores níveis de injustiça do grupo na maioria das abordagens. Esta análise sugere a necessidade de uma seleção cuidadosa da estratégia de filtragem e do método de agrupamento para promover sistemas de recomendação mais justos e inclusivos, destacando a importância de considerar múltiplas dimensões de injustiça na concepção destes sistemas.

### Funções de Objetivo Social (Social Objective Functions)

* Individual fairness (Justiça Individual): a perda do usuário i é a estimativa do erro quadrático médio sobre as classificações conhecidas do usuário i;
* Group Fairness (Justiça de Grupo): a perda do grupo Li como a estimativa do erro quadrático médio sobre todas as avaliações conhecidas no grupo i.

## :computer: Instalação <a name="-instalacao"/></a>

1. Clone o repositório:
```bash
git clone https://github.com/ravarmes/recsys-rgrp-movielens.git
cd recsys-rgrp-movielens
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

4. Baixe o dataset MovieLens-1M:
   - Visite [MovieLens](https://grouplens.org/datasets/movielens/1m/)
   - Baixe o dataset
   - Extraia os arquivos para o diretório `Data/MovieLens-1M`

## :file_folder: Arquivos <a name="-arquivos"/></a>

| Arquivo                               | Descrição                                                                                                                                                                                                                                   |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmImpartiality                | Classe para promover justiça nas recomendações de algoritmos de sistemas de recomendação.                                                                                                                                                   |
| AlgorithmUserFairness                | Classes para medir a justiça (polarização, justiça individual e justiça do grupo) das recomendações de algoritmos de sistemas de recomendação.                                                                                               |
| RecSys                               | Classe no padrão fábrica para instanciar um sistema de recomendação com base em parâmetros string.                                                                                                                                           |
| RecSysALS                            | Alternating Least Squares (ALS) para Filtragem Colaborativa é um algoritmo que otimiza iterativamente duas matrizes para melhor prever avaliações de usuários em itens, baseando-se na ideia de fatoração de matrizes.                       |
| RecSysKNN                            | K-Nearest Neighbors para Sistemas de Recomendação é um método que recomenda itens ou usuários baseando-se na proximidade ou similaridade entre eles, utilizando a técnica dos K vizinhos mais próximos.                                      |
| RecSysNMF                            | Non-Negative Matrix Factorization para Sistemas de Recomendação utiliza a decomposição de uma matriz de avaliações em duas matrizes de fatores não-negativos, revelando padrões latentes que podem ser usados para prever avaliações faltantes. |
| Test_FairnessGroup_MovieLens_Age         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento dos usuários por idade                                                                                                |
| Test_FairnessGroup_MovieLens_Agglomerative         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento aglomerativo                                                                                                |
| Test_FairnessGroup_MovieLens_Gender         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento dos usuários por gênero                                                                                                |
| Test_FairnessGroup_MovieLens_NR         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento dos usuários por número de avaliações                                                                                                |
| Test_FairnessGroup_MovieLens_Activity_KNN_k         | Script para análise de sensibilidade do parâmetro k no algoritmo KNN                                                                                                |
| Test_FairnessGroup_MovieLens_Agglomerative_Clusters         | Script para análise de sensibilidade do número de clusters na injustiça de grupo do agrupamento aglomerativo                                                                                                |

## :database: Datasets <a name="-datasets"/></a>

### Dataset MovieLens-1M
O dataset MovieLens-1M contém 1 milhão de avaliações de 6.040 usuários sobre 3.883 filmes. O dataset inclui:
- Informações dos usuários (gênero, idade, ocupação, código postal)
- Informações dos filmes (título, gêneros)
- Informações das avaliações (valor da avaliação, timestamp)

### Usando Outros Datasets
Para usar um dataset diferente, você precisa:

1. Criar um novo método de leitura de dados na classe `RecSys`:
```python
def read_your_dataset(self, n_users, n_items, top_users, top_items, data_dir):
    # Implemente sua lógica de leitura de dados aqui
    # Retorne X (matriz de avaliações), users_info, items_info
    pass
```

2. Adicionar o novo algoritmo ao método `compute_X_est` na classe `RecSys`:
```python
def compute_X_est(self, X, algorithm):
    if algorithm == 'YourNewAlgorithm':
        # Implemente seu algoritmo aqui
        pass
```

3. Atualizar os scripts de teste para usar seu novo dataset e algoritmo.

## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink: