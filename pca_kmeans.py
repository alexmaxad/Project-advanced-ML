import numpy as np
import random as rd


def sqrt_matrix(a):
    evalues, evectors = np.linalg.eig(a)
    # Ensuring square root matrix exists
    assert (evalues >= 0).all()
    return evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)


class PCA:
    def __init__(self, X, w=None):
        self.X = X
        self.d, self.N = self.X.shape  # N is the number of datapoints and d is their dimension
        if w is not None:
            self.w = w  # w has dimension K
        else:
            self.w = np.ones(self.N) / self.N
        self.D = np.diag(self.w)
        # self.g = self.X @ self.D@np.ones(self.N)  # center of gravity of the dataset
        self.g = np.average(self.X, axis=1, weights=self.w)
        average = np.mean(X, axis=1)
        self.centered_X = self.X - self.g[:, np.newaxis]

        self.cov_X = np.cov(self.X)

    #        self.centered_reduced_X = sqrt_matrix(self.cov_X) @ self.centered_X

    def compute_PCA(self):
        eigen_values, eigen_vectors = np.linalg.eig(self.centered_X @ self.centered_X.transpose() / self.N)
        eigen_values = np.real(eigen_values)
        indices_sorted = np.argsort(eigen_values)
        return eigen_vectors[indices_sorted]


class KmeansByPCA:
    def __init__(self, X, K, beta):
        self.X = X
        self.d, self.N = self.X.shape  # K is the number of datapoints and N is their dimension
        self.K = K
        self.beta = beta
        self.centered_X = self.X - np.mean(self.X, axis=1)[:, np.newaxis]
        self.cov_X = np.cov(self.X)
        # self.centered_reduced_X = sqrt_matrix(self.cov_X) @ self.centered_X

        self.pca = PCA(self.centered_X)
        self.Q = self.compute_Q()
        eigen_values, eigen_vectors = np.linalg.eig(self.centered_X.transpose() @ self.centered_X)
        self.connectivity_matrix = self.compute_connectivity(self.beta)
        self.clusters = self.get_clusters()
        number_clusters = np.max(self.clusters) + 1
        assert number_clusters == K, "the number of clusters is not "+str(K)
        self.mu = np.zeros((number_clusters, self.d))
        size_cluster = np.zeros(number_clusters)
        for i in range(self.N):
            k = self.clusters[i]
            self.mu[k, :] += self.X[:, i]
            size_cluster[k] += 1
        self.mu /= size_cluster

    def get_clusters(self):
        clusters = np.full(self.N, -1)
        cluster_index = -1
        for i in range(self.N):
            if clusters[i] == -1:
                cluster_index += 1
                clusters[i] = cluster_index
                for j in range(i + 1, self.N):
                    if self.connectivity_matrix[i, j]:
                        clusters[j] = cluster_index
        return clusters

    def J_K(self):
        loss = 0
        for k in range(self.K):
            m_k = np.mean(self.X[:, self.clusters[k]])
            loss += np.sum((self.X[:, self.clusters[k]] - m_k) ** 2)
        return loss

    def compute_Q(self):
        eigen_values, eigen_vectors = np.linalg.eig(self.centered_X.transpose() @ self.centered_X / self.N)
        eigen_values = np.real(eigen_values)
        indices_sorted = np.argsort(eigen_values)
        return eigen_vectors[:, indices_sorted][:, -self.K + 1:][:, ::-1]

    def compute_size_clusters(self):
        assert self.Q is not None
        return [np.linalg.norm(self.Q[k]) for k in range(self.K)]

    def compute_connectivity(self, beta):
        assert self.Q is not None and beta > 0 and beta < 1
        P = self.Q @ self.Q.transpose()
        diag_P = np.diagonal(P)
        diag_P_sqrt_inv = np.linalg.inv(np.sqrt(np.diag(diag_P)))
        R = diag_P_sqrt_inv @ P @ diag_P_sqrt_inv
        R = R >= beta
        return R


def create_test_dataset(d, n, mu_list):
    # return K-mixture model with n points in each cluster
    K = len(mu_list)
    X = [np.random.randn(d, n) + mu_list[k][:, np.newaxis] for k in range(K)]
    return np.hstack(X)

mu_list = [np.zeros(2), 5 * np.ones(2)]
d = mu_list[0].shape[0]
n = 100
X = create_test_dataset(2, n, mu_list)
K = len(mu_list)
beta = 0.5
kmeans = KmeansByPCA(X, K, beta)
