<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-rgrp-movielens/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  Group Fairness in Recommendation Systems: MovieLens Dataset
</h3>

<p align="center">Example of Clustering Using Social Fairness Measures</p>

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
  <a href="#-sobre">About the Project</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">License</a>
</p>

## :page_with_curl: About the Project <a name="-sobre"/></a>

This study investigates fairness in recommendation systems using the MovieLens dataset, applying collaborative filtering strategies: ALS, KNN, and NMF. We assess unfairness across different clustering configurations: Gender, Age, Activity, and Agglomerative. The results indicate significant variations in unfairness among the strategies, with the Agglomerative method standing out for exhibiting the highest levels of group unfairness in most approaches. This analysis suggests the need for careful selection of both filtering strategy and clustering method to promote fairer and more inclusive recommendation systems, highlighting the importance of considering multiple dimensions of unfairness in the design of these systems.

<h1 align="center">
    <img alt="abstract" src="https://github.com/ravarmes/recsys-rgrp-movielens/blob/main/assets/graphical_abstract.png" />
</h1>

### Social Objective Functions

* Individual fairness: the loss of user \(i\) is the estimated mean squared error over the known ratings of user \(i\);
* Group fairness: the loss of group \(L_i\) is the estimated mean squared error over all the known ratings in group \(i\).

### Files

| File                               | Description                                                                                                                                                                                                                                   |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmImpartiality                | Class to Promote Fairness in Recommendation Algorithms of Recommendation Systems.                                                                                                                                                   |
| AlgorithmUserFairness                | Classes to Measure Fairness (Polarization, Individual Fairness, and Group Fairness) of Recommendation Algorithms in Recommendation Systems.                                                                                               |
| RecSys                               | Factory Class to Instantiate a Recommendation System Based on String Parameters.                                                                                                                                           |
| RecSysALS                            | Alternating Least Squares (ALS) for Collaborative Filtering is an algorithm that iteratively optimizes two matrices to better predict user ratings on items, based on the idea of matrix factorization.                       |
| RecSysKNN                            | K-Nearest Neighbors for Recommendation Systems is a method that recommends items or users based on the proximity or similarity between them, utilizing the technique of K nearest neighbors.                                      |
| RecSysNMF                            | Non-Negative Matrix Factorization for Recommendation Systems decomposes a rating matrix into two non-negative factor matrices, revealing latent patterns that can be used to predict missing ratings. |
| Test_FairnessGroup_MovieLens_Age         | Test script for the fairness measurement algorithm (AlgorithmUserFairness) considering user grouping by age.                                                                                                |
| Test_FairnessGroup_MovieLens_Agglomerative         | Test script for the fairness measurement algorithm (AlgorithmUserFairness) considering agglomerative clustering.                                                                                                |
| Test_FairnessGroup_MovieLens_Gender         | Test script for the fairness measurement algorithm (AlgorithmUserFairness) considering user grouping by gender.                                                                                                |
| Test_FairnessGroup_MovieLens_NR         | Test script for the fairness measurement algorithm (AlgorithmUserFairness) considering user grouping by the number of ratings.                                                                                                |

## 📺 Video Abstract

Check out the explanatory video for this project on [YouTube](https://youtu.be/AUhsiSWMjzA).


## :memo: License <a name="-licenca"/></a>

This project is under the MIT License. See the [LICENSE](LICENSE.md) file for more details.

## :email: Contact

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink:
