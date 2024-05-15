<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-cluster-loss/blob/master/assets/logo.jpg" />
</h1>

<h3 align="center">
  Análise de Justiça de Grupo no Dataset MovieLens
</h3>

<p align="center">Exemplo de agrupamentos utilizando medidas de justiça individual </p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-cluster-loss?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-cluster-loss/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-cluster-loss?style=social">
  </a>
</p>

<p align="center">
  <a href="#-sobre">Sobre o projeto</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-links">Links</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">Licença</a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

O objetivo deste repositório é implementar os cálculos para uma análise de agrupamento no contexto de sistemas de recomendação, considerando as perdas da medida de justiça individual.

Os cálculos da justiça individual são baseados nas implementações do respositório [antidote-data-framework](https://github.com/rastegarpanah/antidote-data-framework) 

### Funções de Objetivo Social (Social Objective Functions)

* Individual fairness (Justiça Individual): a perda do usuário i é a estimativa do erro quadrático médio sobre as classificações conhecidas do usuário i

### Arquivos

- RecSys: implementação da classe genérica para a utilização do sistema de recomendação
- RecSysALS: implementação do sistema de recomendação baseado em filtragem colaborativa utilizando ALS (mínimos quadrados alternados)
- RecSysExampleData20Items: implementação de uma matriz de recomendações estimadas (apenas exemplo com valores aleatórios)
- UserFairness: implementação das funções de objetivo social (polarização, justiça individual e justiça do grupo)
- TestUserFairness_Books: arquivo para testar a implementação UserFairness com base no conjunto de dados Books
- TestUserFairness_MovieLens_1M: arquivo para testar a implementação UserFairness com base no conjunto de dados MovieLens-1M
- TestUserFairness_MovieLens_Small: arquivo para testar a implementação UserFairness com base no conjunto de dados MovieLens-Small (40 usuários e 20 itens)
- TestCluster_Books_01: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados Books e nas variáveis justiça individual, idade e localização.
- TestCluster_Books_02: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados Books e na variável justiça individual.
- TestCluster_MovieLens_1M_01: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-1M e nas variáveis justiça individual, idade e ocupação.
- TestCluster_MovieLens_1M_02: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-1M e na variável justiça individual.
- TestCluster_MovieLens_Small_01: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-Small e nas variáveis justiça individual, idade, NA, SPI, MA e MR.
- TestCluster_MovieLens_Small_02: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-Small e na variável justiça individual.


## :link: Links <a name="-links"/></a>

- [Google Sheets](https://github.com/ravarmes/recsys-cluster-loss/blob/master/docs/recsys-cluster-loss-example.xlsx) - Planilha para demonstrar a utilização do algoritmo para uma base de dados pequena (40 usuários e 20 filmes);
- [Artigo](https://arxiv.org/pdf/1812.01504.pdf) - Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems;


## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink: