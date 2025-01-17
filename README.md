# Project: A Comparative Study of Spectral Clustering and Guided PCA Clustering

## Objectives

1. *Theoretical Understanding:* Delve deep into the theoretical foundations of Spectral Clustering and K-Means Clustering, emphasizing their key concepts, algorithms, and functionalities.

2. *Practical Implementation:* Implement both clustering techniques on real or synthetic datasets. Utilize popular libraries such as scikit-learn, NumPy, or others as per the project requirements.

3. *Comparative Analysis:* Conduct a thorough comparative analysis of the performance of both methods across different data configurations. Consider factors such as cluster quality, stability, sensitivity to parameters, etc.

4. *Results Visualization:* Employ visualization techniques such as graphs, scatter plots, or heatmaps to illustrate clustering results. This will aid in understanding the differences and similarities between the two methods.

## Code Structure

All codes have been written by us, except if it is mentionned.

1. **Kmeans algorithm**
   - ```Kmeans.py``` : This file contains all the functions used to perform the Kmeans clustering.  The file contains the generalized_Kmeans() class and a store of initialisation functions (We coded it using a dictionary where the attributes are the functions). Our Kmeans can perform guided_PCA, forgy, random-partition and Kmeans++ initialisations.

2. **Spectral clustering**

    - ```spectral_functions.py``` : This file contains all the functions used to perform spectral clustering. The code has been largely inspired by the article "*A tutorial on spectral clustering*" by Luxburg. However, even if we tried to use the power method to compute the smallest eigenvectors of a laplacian matrix as the article advises, we prefered using the ARPACK package, available on scipy, which is a quicker way for sparse matrices. The script gives functions to perform spectral clustering with different types of graphs and laplacians, but in practice we only used the KNN graph with random walk or symetric laplacians, as it was advised in the previously mentionned article. The Kmeans stage of spectral clustering is done using our functions in ```K_means.py```.

    - ```spectral_class.py``` :  This file contains similar functions as the previous one, but built in a spectral clustering class. For reasons that we haven't fully understood for now, the use of this class takes more time to run than directly using the functions from ```spectral_functions```, so we decided no to use it for our tests. Therefore, this file doesn't really matter.

3. **Tests and comparisons**

    - ```Test missing data.ipynb``` : This notebook studies and compares the accuracy of guided PCA and spectral clustering on syntethic data. We set a number of clusters, a number of features and cluster centers at the begginning, then we remove different proportions of the data. The data is then completed with a 5 nearest neighbor search. After that, we look at the evolution of number of correctly classified points by our two different methods, compared to the proportion of missing data.

    - ```Test noisy data.ipynb``` : This notebook allows to observe the performances of our two methods on noisy synthetic data. We first generate data with a fixed number of features and clusters, then we add homoscedastic and heteroscedastic noise to this data. After that we compare the accuracy of standard Kmeans, guided PCA and spectral clustering on the different types of data. We also look at the influence on the clustering's accuracy of the number of clusters, the number of clusters, and the number of nearest neighbors considered in the spectral clustering.

    - ```Test high dimension data.ipynb``` : This notebook allows to observe the performances of our two methods on high dimensional data. We put ourselves in the conditions of the experiment on the MNIST dataset of the article of Qin Xu untitled "PCA-guided search for K-means". We show that spectral and guided-PCA Kmeans clusterings are the most efficient in execution time. In addition, while the best accuracies are quite the same between the different algorithms, we show that spectral clustering is quite bad at inertia minimizations. Nevertheless, this is quite logical since, contrary to the Kmeans algorithms, the aim of the spectral clustering isn't inertia minimization,   

## Results reproduction

All the results can be directly obtained by executing the notebooks, as the data is loaded from scikit-learn. 
However, the results of spectral clustering depend largely on the similarity function used, if we use a weighted graph or not, and the number of nearest neighbors we use to construct the adjacency matrix. To get satisfying results, it is thus advised to set the arguments : ```similarity_function = euclidian_similarity_function``` and ```weighted = True```.

## References 

- von Luxburg, U. A tutorial on spectral clustering. Stat Comput 17, 395–416 (2007).
- Xu, Qin & Ding, Chris & Liu, Jinpei & Bin, Luo. (2014). PCA-guided search for K-means. Pattern Recognition Letters.
- Géron, Aurélien. Hands-on machine learning with Scikit-Learn and TensorFlow : concepts, tools, and techniques to build intelligent systems. Sebastopol, CA: O'Reilly Media, 2017.

