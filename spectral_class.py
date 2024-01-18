import pandas as pd
from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Literal
from K_means import *
from sklearn.datasets import make_blobs
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler, normalize
from scipy.sparse.linalg import eigs, eigsh
from scipy.spatial.distance import cdist

## Similarity measures

def gaussian_similarity_function_(data_point_1, data_point_2, sigma) : 
    """ Computes the similarity between two datapoints with gaussian similarity function"""
    dist = np.linalg.norm(data_point_1 - data_point_2)
    return np.exp( - ((dist**2) / (2*sigma)))

def euclidian_similarity_function(data_point_1, data_point_2, sigma) :
    """Euclidian distance between two points"""
    return np.linalg.norm(data_point_1 - data_point_2)

## Laplacians

def degree_matrix(adjacency_matrix) :
    """Creates the degree matrix from an adjacency matrix"""
    dimension = len(adjacency_matrix)
    d = []
    for i in range(dimension) :
        d.append(sum(adjacency_matrix[i]))
    D = np.diag(d)
    return D

def regular_laplacian(adjacency_matrix) :
    """Regular laplacian from adjacency matrix"""
    W = adjacency_matrix
    D = degree_matrix(W)

    return D - W

def laplacian_sym(adjacency_matrix) :
    """Symetric laplacian from adjacency matrix"""
    W = adjacency_matrix
    I = np.identity(len(W))

    dimension = len(W)
    d = []
    for i in range(dimension) :
        d.append(1/(math.sqrt(sum(adjacency_matrix[i]))))

    D_minus_one_half = np.diag(d)

    step_1 = np.matmul(W, D_minus_one_half)
    step_2 = np.matmul(D_minus_one_half, step_1)

    return I - step_2

def laplacian_rw(adjacency_matrix) : 
    """Random walk laplacian from adjacency matrix"""
    W = adjacency_matrix
    I = np.identity(len(W))

    dimension = len(W)
    d = []
    for i in range(dimension) :
        d.append(1/(sum(adjacency_matrix[i])))

    D_minus_1 = np.diag(d)

    return I - np.matmul(D_minus_1, W)

## Eigenvectors computation

def simultaneous_power_iteration(A):
    """A first method to get the smallest eigenvectors of a matrix, based on the QR decomposition. But not used in practice because too slow."""

    # QR method

    n, m = A.shape
    Q = np.random.rand(n, n)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
 
    for i in range(1000):
        Z = A.dot(Q)
        Q, R = np.linalg.qr(Z)

        err = ((Q - Q_prev) ** 2).sum()

        Q_prev = Q
        if err < 1e-6:
            break

    return np.diag(R), Q

def compute_matrix_U_simult_power(A, k) :

    Q = simultaneous_power_iteration(A)[1]
    U = Q[:, -k:] 

    return U

def compute_matrix_U_ARPACK(A, k) :
    """Computing the smallest eigen vectors using the package ARPACK."""
    vecp = eigsh(A, k, which='SM')[1]
    
    return vecp


class spectral_clustering():

    def __init__(
            self, 
            similarity_function = None,
            k_nearest_neighbors : int = 4, 
            number_of_clusters : int = 3,
            type_of_graph : Literal['k_nearest_neighbors','fully_connected'] = 'k_nearest_neighbors', 
            weighted : bool = True,
            type_of_laplacian : Literal['regular', 'sym', 'rw'] = 'sym',
            eigensolver = None,
            loss: Literal['inertia'] = 'inertia',
            sigma = 1,
        ):

    
        self.similarity_function = similarity_function
        self.type_of_laplacian = type_of_laplacian
        self.k_nearest_neighboors = k_nearest_neighbors
        self.number_of_clusters = number_of_clusters
        self.type_of_graph = type_of_graph
        self.weighted = weighted
        self.eigensolver = eigensolver
        self.sigma = sigma
        self.loss = loss

    ## Adjacency matrices

    def fully_connected_adjacency_matrix(self, data) : 
        (nbr_of_points, dimension) = np.shape(data)
        W = np.zeros((nbr_of_points,nbr_of_points))
        for i in range(nbr_of_points) :
            for j in range(i+1, nbr_of_points) :
                W[i, j] = self.similarity_function(data[i], data[j])
        W = W + W.T
        for i in range(nbr_of_points) :
            W[i,i] = self.similarity_function(data[i], data[i])
        # Not sure if we need to do it. 
        return W

    def KNN_adjacency_matrix(self, data) :
        (nbr_of_points, dimension) = np.shape(data)
        W = np.zeros((nbr_of_points,nbr_of_points))
        for i in range(nbr_of_points) :
            distances = {}
            for j in range(nbr_of_points) :
                if j != i :
                    distances[j] = euclidian_similarity_function(data[i], data[j], self.sigma)
                if j == i :
                    distances[j] = np.inf
            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=False))
            keys_to_zero = list(sorted_distances.keys())[self.k_nearest_neighboors:]
            keys_to_one = list(sorted_distances.keys())[:self.k_nearest_neighboors]
            for key in keys_to_zero :
                distances[key] = 0
            if self.weighted :
                for key in keys_to_one :
                    distances[key] = self.similarity_function(data[i], data[key], self.sigma) #for weighted graph
            else :
                for key in keys_to_one :
                    distances[key] = 1
            W[i] = list(distances.values())
        return 0.5 * (W + W.T)

    def fit_predict(self, data):
        
        if self.type_of_graph == 'k_nearest_neighbors' :
            W = self.KNN_adjacency_matrix(data)
        if self.type_of_graph == 'fully_connected' :
            W = self.fully_connected_adjacency_matrix(data)

        if self.type_of_laplacian == 'regular' :
            L = regular_laplacian(W)
        if self.type_of_laplacian == 'sym' :
            L = laplacian_sym(W)
        else :
            L = laplacian_rw(W)

        n_clusters = self.number_of_clusters
        U = self.eigensolver(L, n_clusters)

        if self.type_of_laplacian == 'sym' :
            U = normalize(U, norm='l2')

        kmeans = generalized_Kmeans(k=self.number_of_clusters, init='forgy', epochs=10, random_seed=42)
        kmeans.fit(U)

        self.labels = kmeans.predict(U)
        self.loss = kmeans.loss_

        return self
        