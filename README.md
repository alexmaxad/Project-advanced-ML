# Project: A Comparative Study of Spectral Clustering and Guided PCA Clustering

## Introduction

Welcome to the project comparing two popular clustering techniques: Spectral Clustering and K-Means Clustering. This project aims to explore and compare these two approaches to data clustering, highlighting their advantages, disadvantages, and providing practical application examples.

## Project Objectives

1. *Theoretical Understanding:* Delve deep into the theoretical foundations of Spectral Clustering and K-Means Clustering, emphasizing their key concepts, algorithms, and functionalities.

2. *Practical Implementation:* Implement both clustering techniques on real or synthetic datasets. Utilize popular libraries such as scikit-learn, NumPy, or others as per the project requirements.

3. *Comparative Analysis:* Conduct a thorough comparative analysis of the performance of both methods across different data configurations. Consider factors such as cluster quality, stability, sensitivity to parameters, etc.

4. *Results Visualization:* Employ visualization techniques such as graphs, scatter plots, or heatmaps to illustrate clustering results. This will aid in understanding the differences and similarities between the two methods.

5. *Final Report:* Present the results, conclusions, and recommendations in a clear and concise final report. Include detailed explanations regarding parameter choices, implementation steps, and reasons behind observed performances.

## Project Structure

The project is organized into several sections for clear understanding and logical progression. Here is a suggested structure:

1. *Introduction*
   - Project overview
   - Objectives and motivation

2. *Theory*
   - Detailed explanation of Spectral Clustering
   - Detailed explanation of K-Means Clustering
   - Strengths and weaknesses of each method

3. *Methodology*
   - Description of datasets used
   - Experiment configuration
   - Choice of evaluation metrics

4. *Implementation*
   - Source code for Spectral Clustering
   - Source code for K-Means Clustering
   - Parameter configuration

5. *Comparative Analysis*
   - Experiment results
   - Performance comparison
   - Interpretation of results

6. *Visualization*
   - Graphs and visualizations of formed clusters
   - Illustration of differences between the two methods

7. *Conclusion*
   - Summary of results
   - Final conclusions
   - Suggestions for future work

8. *References*
   - List of used bibliographic references

## Code Structure

All codes have been written by us, except if it is mentionned.

1. **Spectral clustering**
    - ```spectral_functions.py``` : This file contains all the functions used to perform spectral clustering. The code has been largely inspired by the article "*A tutorial on spectral clustering*" by Luxburg. However, even if we tried to use the power method to compute the smallest eigenvectors of a laplacian matrix as the article advisesÒ, we prefered using the ARPACK package, available on scipy, which is a way quicker method for sparse matrices. The script gives functions to perform spectral clustering with different types of graphs and laplacians, but in practice we only used the KNN graph with random walk or symetric laplacians, as it was advised in the previously mentionned article. The Kmeans stage of spectral clustering is done using our functions in ```K_means.py```.
    - ```spectral_class.py``` :  This file contains similar functions as the previous one, but built in a spectral clustering class. For reasons that we haven't fully understood for now, the use of this class takes more time to run than directly using the functions from ```spectra_functions```, so we decided no to use it for our tests. Therefore, this file doesn't really matter. 

2. **Tests and comparisons**
    - ```Test missing data.ipynb``` : This notebook studies and compares the accuracy of guided PCA and spectral clustering on syntethic data. We set a number of clusters, a number of features and cluster centers at the begginning, then we remove different proportions of the data. The data is then completed with a 5 nearest neighbor search. After that, we look at the evolution of number of correctly classified points by our two different methods, compared to the proportion of missing data.
    - ```Test noisy data.ipynb``` : This notebook allows to observe the performances of our two methods on noisy synthetic data. We first generate data with a fixed number of features and clusters, then we add homoscedastic and heteroscedastic noise to this data. After that we compare the accuracy of standard Kmeans, guided PCA and spectral clustering on the different types of data. We also look at the influence on the clustering's accuracy of the number of clusters, the number of clusters, and the number of nearest neighbors considered in the spectral clustering. 

## Results reproduction

All the results can be directly obtained by executing the notebooks, as the data is loaded from scikit-learn. 
However, the results of spectrl clustering depend largely on the similarity function used, if we use a weighted graph or not, and the number of nearest neighbors we use to construct the adjacency matrix. To get satisfying results, it is thus advised to set the arguments : ```similarity_function = euclidian_similarity_function``` and ```weighted = True```.

## Prerequisites

Before running the code, ensure you have the necessary libraries installed. You can use the following command to install the required dependencies:

```bash
pip install -r requirements.txt