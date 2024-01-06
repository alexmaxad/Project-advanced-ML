import numpy as np
from numpy import inf
from typing import Literal
from sklearn.metrics import DistanceMetric
from sklearn.decomposition import PCA
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, squareform

def random_partition_init(model, X: NDArray) -> NDArray:
    """ Random Partition initialise aleatoirement les clusters
    puis effectue une etape de centre.
    """
    clusters = model._random_generator.choice(model.k, len(X)) 
    # on remplace les centres precendent par une matrice vide (np.empty) 
    # dans l'appel de _center_step
    return model._center_step(np.empty((model.k, X.shape[1])), clusters, X)

def forgy_init(model, X: NDArray) -> NDArray:
    """ L'initialisation forgy assigne directement les centres a partir des points du jeu de données"""
    return X[model._random_generator.choice(len(X), model.k, replace=False)] 

def kmeans_plus_plus_init(model, X: NDArray) -> NDArray:
    """Initialisation kmeans ++ """

    # assignation du premier centre aleatoirement
    centers = [model._random_generator.choice(len(X), 1)[0]]
    # iteration jusqu'a avoir k centres
    for _i in range(2, model.k + 1):
        # calcule de la distance des points aux centres

        distances = cdist(X, X[centers], metric=model._distance) ** 2 #type: ignore

        # on prend la distance au centroid le plus proche
        min_distances = (distances).min(axis=1).squeeze()
        # normalisation dans [0,1]
        proba = min_distances / np.sum(min_distances)

        # selection aleatoire de (taille des centres)+1 points
        # pour etre sur qu'au moins 1 des tiré au hasard n'est pas deja un centre
        picks = model._random_generator.choice(
            len(X), size=len(centers) + 1, p=proba, replace=False
        )
        # ajout du premier centre selectionné qui n'en est pas deja un.
        centers.append(picks[np.isin(picks, centers, invert=True)][0])    
    return X[centers]
    
def guided_PCA_init(model, X: NDArray) -> NDArray:
    
    if model.df_PCA is None:

        n_components = model.k  # Choisissez le nombre de composantes principales
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
    La classe BaseKAlgorithme est une classe de base pour les algorithmes de clustering K-means.
    On héritera cette classe pour simplifier les classes Kmeans et FuzzyMeans.
    
    Attributes:
        - k (int): Nombre de clusters.
        - init (Callable): Fonction d'initialisation utilisée pour initialiser les centres.
        - loss (str): Fonction de perte utilisée pour évaluer la qualité du clustering.
        La string passée permettra de récupérer la loss dans le store

        - epochs (int): Nombre d'estimateurs entrainés.
        - max_iter (int): Nombre maximum d'itérations pour chaque estimateur.

        - sensitivity (float): Sensibilité pour déterminer la convergence de l'algorithme (difference entre les centres entre deux étapes).

        - random_seed (int = 42): la graine aléatoire.
        - distance (str): Métrique de distance utilisée pour calculer la similarité entre les points.
        La string passée doit correspondre a une distance de sklearn 

        - norm : Norme utilisée pour calculer la différence des centre entre chaque étape.

        Ces 2 attributs sont purement techniques et permettent d'enregistrer la PCA en cas de multiples executions de la guided_PCA sur le même data set 
        - df_PCA : la reduced data set
        - model_PCA : le modèle de l'ACP 
    
    Classe Pour l'algorithme de Kmeans.

    Matrix data fitted by this class needs to be of the form (n,d) with :
    - n : number of data points
    - d : number of features
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

        # initialisation d'un generateur aleatoire
        if random_seed == 0:
            self._random_generator = np.random.default_rng()
        else:
            self._random_generator = np.random.default_rng(random_seed)

        # recuperation de la distance
        self._distance: str = distance

        self.init = init
        self._init = dico_init_fcts[init]

        # recuperation de la loss
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
        L'inertie représente la variance intra-cluster de l'estimateur. On peut en changer la 
        distance, ainsi que la réduction (par défaut, on prend la somme des variances intra-clusters)
        """
        inertia = 0
        # iteration sur les centres
        for cluster, center in enumerate(centers):
            # donnees liees a un cluster 
            cluster_data = X[clusters == cluster]
            cluster_count = len(cluster_data)
            if cluster_count > 0:
                # ajout de la variance intra-cluster pour ce cluster
                inertia += np.linalg.norm(cluster_data - center)**2
            else:
                inertia = np.inf           
        return inertia  # type: ignore

    def _set_loss(self, centers, clusters, X):
        if self._loss == 'inertia':
            return self.inertia(centers, clusters, X)

    def _center_step(self, centers: NDArray, clusters: NDArray, X: NDArray) -> NDArray:
        # iteration sur les clusters:
        for cluster in np.unique(clusters):
            # les points contenus dans un cluster
            cluster_data = X[clusters == cluster]
            if len(cluster_data) > 0:
                # remplace le centre par le nouveau (moyenne ou mediane des points)
                centers[cluster] = self._center_step_reducer(cluster_data, axis=0)
        return centers

    def _cluster_step(self, centers: NDArray, X: NDArray) -> NDArray:
        # choisi pour chaque point le centre le plus proche selon la distance
        return np.argmin(cdist(X, centers, metric=self._distance), axis=1)

    def fit(self, X: NDArray):
        best_model: NDArray
        best_loss: float = inf
        # iteration selon le nombre de model souhaité
        if self.init == 'guided_PCA':
            self.epochs = 1
        for self.epoch in range(self.epochs):
            # initialisation des centres
            centers: NDArray = self._init(self, X)
            clusters: NDArray
            # initialisation du compteur d'etape 
            step_counter: int = 0
            
            # initalisation de la difference entre les centre a l'infini
            center_diff = np.inf
            # boucle while 
            while (step_counter < self.max_iter) and (center_diff > self.sensitivity):
                # copie memoire des centres actuels car ceux-ci vont etre modifies 
                # dans la prochaine etape de centre
                last_centers = centers.copy()

                # assignation des clusters
                clusters = self._cluster_step(centers, X)

                # assignation des centres
                centers = self._center_step(centers, clusters, X)

                step_counter+=1

                # calcul de la difference entre les centres precedents et les nouveaux
                center_diff = self.norm(centers - last_centers)
            
            # calcul de la loss
            loss = self._set_loss(centers, clusters, X) #type: ignore
            if loss < best_loss:
                best_model = centers
                best_loss = loss
                best_count = step_counter

        # assignation du meilleur estimateur
        self.centers_ = best_model #type: ignore
        self.loss_ = best_loss
        self.step_counter_ = best_count #type: ignore
        return self

    def predict(self, X: NDArray) -> NDArray:
        # prediction en faisant une cluster_step avec les centres entraîné
        return self._cluster_step(self.centers_, X)