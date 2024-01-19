import numpy as np
from numpy import inf
from typing import Literal
from sklearn.metrics import DistanceMetric
from sklearn.decomposition import PCA
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, squareform

def random_partition_init(model, X: NDArray) -> NDArray:
    """ Random Partition randomly initializes clusters
    then performs a center step.
    """
    clusters = model._random_generator.choice(model.k, len(X)) 
    # replace precendent centers with an empty matrix (np.empty)
    # in the _center_step call
    return model._center_step(np.empty((model.k, X.shape[1])), clusters, X)

def forgy_init(model, X: NDArray) -> NDArray:
    """ Forgy initialization assigns centers directly from dataset points """
    return X[model._random_generator.choice(len(X), model.k, replace=False)] 

def kmeans_plus_plus_init(model, X: NDArray) -> NDArray:
    """ Initialization kmeans ++ """

    # random assignment of the first center
    centers = [model._random_generator.choice(len(X), 1)[0]]
    # iteration until k centers
    for _i in range(2, model.k + 1):
        # calculates the distance from points to centers

        distances = cdist(X, X[centers], metric=model._distance) ** 2 #type: ignore

        # we take the distance to the nearest centroid
        min_distances = (distances).min(axis=1).squeeze()
        # standardization in [0,1]
        proba = min_distances / np.sum(min_distances)

        # random selection of (center size)+1 points
        # to be sure that at least 1 of the randomly selected is not already a center
        picks = model._random_generator.choice(
            len(X), size=len(centers) + 1, p=proba, replace=False
        )
        # addition of the first center selected that isn't already one.
        centers.append(picks[np.isin(picks, centers, invert=True)][0])    
    return X[centers]
    
def guided_PCA_init(model, X: NDArray) -> NDArray:
    
    if model.df_PCA is None:

        n_components = model.k # Choose the number of principal components
        pca = PCA(n_components=n_components)
        model.df_PCA = pca.fit_transform(X)
        model.model_PCA = pca

    kmeans_initialisation = generalized_Kmeans(model.k, init='kmeans++')
    kmeans_initialisation.fit(model.df_PCA)
    cluster_centroids = kmeans_initialisation.centers_

    init_centroids = model.model_PCA.inverse_transform(cluster_centroids)
    return init_centroids


dico_init_fcts = {
    'kmeans++': kmeans_plus_plus_init,
    'random-partition': random_partition_init,
    'forgy': forgy_init,
    'guided_PCA': guided_PCA_init
}

class generalized_Kmeans():
    """
    The generalized_Kmeans class is a class for K-means clustering algorithms.

    Attributes:
        - k (int): Number of clusters.
        - init (Callable): Initialization function used to initialize centers.
        - loss (str): Loss function used to evaluate clustering quality.
        The string passed will retrieve the loss from the store

        - epochs (int): Number of trained estimators.
        - max_iter (int): Maximum number of iterations for each estimator.

        - sensitivity (float): Sensitivity to determine algorithm convergence (difference between centers between two steps).

        - random_seed (int = 42): the random seed.
        - distance (str): Distance metric used to calculate similarity between points.
        The string passed must correspond to a sklearn distance

        - norm : Standard used to calculate the difference in centers between each stage.

        These 2 attributes are purely technical and are used to register the PCA in the event of multiple guided_PCA executions on the same data set.
        - df_PCA : the reduced data set
        - model_PCA : The PCA model
    """


    def __init__(
            self, 
            k: int, 
            init: Literal['kmeans++','random-partition','guided_PCA', 'forgy'], 
            epochs: int = 10,
            max_iter: int = 100,
            sensitivity: float = 0.0001, 
            random_seed: int = 0, 
            distance: str = 'euclidean', 
            center_step_reducer: Literal['mean', 'median'] = 'mean', 
            loss: Literal['inertia'] = 'inertia',
            norm = np.linalg.norm,
            df_PCA = None, 
            model_PCA = None,
        ):

        # initializing a random generator
        if random_seed == 0:
            self._random_generator = np.random.default_rng()
        else:
            self._random_generator = np.random.default_rng(random_seed)

        # distance recovery
        self._distance: str = distance

        self.init = init
        self._init = dico_init_fcts[init]

        # loss recovery
        self._loss = loss

        self.k = k
        self.epochs = epochs
        self.max_iter = max_iter
        self.sensitivity = sensitivity
        self.norm = norm
        self.df_PCA = df_PCA
        self.model_PCA = model_PCA

        self._center_step_reducer =  np.mean if center_step_reducer == 'mean' else np.median

    def inertia(self, centers: NDArray, clusters: NDArray, X: NDArray) -> float:
        """ 
        The inertia represents the intra-cluster variance of the estimator. You can change the
        distance, and reduction (by default, the sum of intra-cluster variances is used)
        """
        inertia = 0
        # iteration on centers
        for cluster, center in enumerate(centers):
            # cluster data
            cluster_data = X[clusters == cluster]
            cluster_count = len(cluster_data)
            if cluster_count > 0:
                # intra-cluster variance added for this cluster
                inertia += np.linalg.norm(cluster_data - center)**2
            else:
                inertia = np.inf           
        return inertia  # type: ignore

    def _set_loss(self, centers, clusters, X):
        if self._loss == 'inertia':
            return self.inertia(centers, clusters, X)

    def _center_step(self, centers: NDArray, clusters: NDArray, X: NDArray) -> NDArray:
        # iteration on clusters:
        for cluster in np.unique(clusters):
            # points contained in a cluster
            cluster_data = X[clusters == cluster]
            if len(cluster_data) > 0:
                # replace the center with the new center (average or median of points)
                centers[cluster] = self._center_step_reducer(cluster_data, axis=0)
        return centers

    def _cluster_step(self, centers: NDArray, X: NDArray) -> NDArray:
        # for each point, choose the nearest center according to distance
        return np.argmin(cdist(X, centers, metric=self._distance), axis=1)

    def fit(self, X: NDArray):
        best_model: NDArray
        best_loss: float = inf
        # iteration according to desired number of models
        if self.init == 'guided_PCA':
            self.epochs = 1
        for self.epoch in range(self.epochs):
            # center initialization
            centers: NDArray = self._init(self, X)
            clusters: NDArray
            # step counter initialization
            step_counter: int = 0
            
            # initalizing the difference between centers at infinity
            center_diff = np.inf
            while (step_counter < self.max_iter) and (center_diff > self.sensitivity):
                # memory copy of current centers as they will be modified
                # in the next center stage
                last_centers = centers.copy()

                # cluster assignment
                clusters = self._cluster_step(centers, X)

                # center assignment
                centers = self._center_step(centers, clusters, X)

                step_counter+=1

                # calculating the difference between previous and new centers
                center_diff = self.norm(centers - last_centers)
            
            # loss calculation
            loss = self._set_loss(centers, clusters, X) #type: ignore
            if loss < best_loss:
                best_model = centers
                best_loss = loss
                best_count = step_counter

        # best estimator assignment
        self.centers_ = best_model #type: ignore
        self.loss_ = best_loss
        self.step_counter_ = best_count #type: ignore
        return self

    def predict(self, X: NDArray) -> NDArray:
        # prediction by cluster_step with trained centers
        return self._cluster_step(self.centers_, X)