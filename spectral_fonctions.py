import pandas as pd
from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt
from K_means import *
from sklearn.datasets import make_blobs
from numpy import dot
from numpy.linalg import norm
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from scipy.sparse.linalg import eigs, eigsh
from spectral_class import *
from scipy.spatial.distance import cdist

## Similarity measures

def gaussian_similarity_function(data_point_1, data_point_2) : 
    sigma = 1
    dist = np.linalg.norm(data_point_1 - data_point_2)
    return np.exp( - ((dist**2) / (2*sigma)))

def euclidian_similarity_function(data_point_1, data_point_2) :
    return np.linalg.norm(data_point_1 - data_point_2)




## Adjacency matrices

def fully_connected_adjacency_matrix(data, similarity_funtion) : 
    (nbr_of_points, dimension) = np.shape(data)
    W = np.zeros((nbr_of_points,nbr_of_points))
    for i in range(nbr_of_points) :
        for j in range(i+1, nbr_of_points) :
            W[i, j] = similarity_funtion(data[i], data[j])
    W = W + W.T
    for i in range(nbr_of_points) :
        W[i,i] = similarity_funtion(data[i], data[i])
    # Not sure if we need to do it. 
    return W

def KNN_adjacency_matrix(data, similarity_function, K, weighted : bool) :
    (nbr_of_points, dimension) = np.shape(data)
    W = np.zeros((nbr_of_points,nbr_of_points))
    for i in range(nbr_of_points) :
        distances = {}
        for j in range(nbr_of_points) :
            if j != i :
                distances[j] = euclidian_similarity_function(data[i], data[j])
            if j == i :
                distances[j] = np.inf
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1], reverse=False))
        keys_to_zero = list(sorted_distances.keys())[K:]
        keys_to_one = list(sorted_distances.keys())[:K]
        for key in keys_to_zero :
            distances[key] = 0
        if weighted :
            for key in keys_to_one :
                distances[key] = similarity_function(data[i], data[key]) #for weighted graph
        else :
            for key in keys_to_one :
                distances[key] = 1
        W[i] = list(distances.values())
    return 0.5 * (W + W.T)




## Laplacians

def degree_matrix(adjacency_matrix) :
    dimension = len(adjacency_matrix)
    d = []
    for i in range(dimension) :
        d.append(sum(adjacency_matrix[i]))
    D = np.diag(d)
    return D

def regular_laplacian(adjacency_matrix) :

    W = adjacency_matrix
    D = degree_matrix(W)

    return D - W

def laplacian_sym(adjacency_matrix) :

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

    W = adjacency_matrix
    I = np.identity(len(W))

    dimension = len(W)
    d = []
    for i in range(dimension) :
        d.append(1/(sum(adjacency_matrix[i])))

    D_minus_1 = np.diag(d)

    return I - np.matmul(D_minus_1, W)




## Eigenvectors computation

def power_iteration(A, n_simulations):

    vector = np.random.rand(A.shape[1])

    for i in range(n_simulations):
        y = np.dot(A, vector)
        new_vector = y / np.linalg.norm(y)
        if np.linalg.norm(new_vector - vector) < 1e-6:
            break
        vector = new_vector

    lambda_k = np.dot(new_vector, np.dot(A, new_vector)) / np.dot(new_vector, new_vector)

    return new_vector, lambda_k

def simultaneous_power_iteration(A):

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

    vecp = eigsh(A, k, which='SM')[1]
    
    return vecp




## Final spectral clustering functions

def final_spectral_clustering_regular(data, similarity_function, k_nearest_neighboors, number_of_clusters, weighted : bool, eigensolver) :

    W = KNN_adjacency_matrix(data, similarity_function, k_nearest_neighboors, weighted)
    L = regular_laplacian(W)
    U = eigensolver(L, number_of_clusters)

    kmeans = generalized_Kmeans(k=number_of_clusters, init='forgy', epochs=10, random_seed=42)
    kmeans.fit(U)

    labels = kmeans.predict(U)

    return labels

def final_spectral_clustering_sym(data, similarity_function, k_nearest_neighboors, number_of_clusters, weighted : bool, eigensolver) :

    W = KNN_adjacency_matrix(data, similarity_function, k_nearest_neighboors, weighted)
    L = laplacian_sym(W)
    U = eigensolver(L, number_of_clusters)
    T = normalize(U, norm="l2")

    kmeans = generalized_Kmeans(k=number_of_clusters, init='forgy', epochs=10, random_seed=42)
    kmeans.fit(T)

    labels = kmeans.predict(T)

    return labels

def final_spectral_clustering_rw(data, similarity_function, k_nearest_neighboors, number_of_clusters, weighted : bool, eigensolver) :

    W = KNN_adjacency_matrix(data, similarity_function, k_nearest_neighboors, weighted)
    L = laplacian_rw(W)
    U = eigensolver(L, number_of_clusters)

    kmeans = generalized_Kmeans(k=number_of_clusters, init='forgy', epochs=10, random_seed=42)
    kmeans.fit(U)

    labels = kmeans.predict(U)

    return labels


## Loss and inertia

def centers(labels, data) :

    centers = []

    for label in set(labels) : 
        
        points = data[labels == label]
        sum_coordinates = np.sum(points, axis=0)
        centers.append(sum_coordinates / len(points))
        
    return centers

def loss_inertia(labels, data) :

    inertia = 0

    for label in set(labels) : 
        
        points = data[labels == label]
        sum_coordinates = np.sum(points, axis=0)
        center = sum_coordinates / len(points)

        inertia += np.linalg.norm(points - center)**2
    
    return inertia