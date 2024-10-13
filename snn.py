import numpy as np
from scipy.sparse import csr_matrix
import numba as nb


# Implementation of the Shared Nearest Neighbors (SNN) algorithm based on Seurat ComputeSNN function in C++.
# https://github.com/satijalab/seurat/blob/1549dcb3075eaeac01c925c4b4bb73c73450fc50/src/snn.cpp


@nb.njit(parallel=True)
def get_indices(indptr, indices, data, n_samples, max_k):
    result = np.zeros((n_samples, max_k), dtype=np.int32)
    for i in nb.prange(n_samples):
        start, end = indptr[i], indptr[i + 1]
        row_indices = indices[start:end]
        row_data = data[start:end]
        sorted_order = np.argsort(row_data)
        sorted_indices = row_indices[sorted_order]
        result[i, : len(sorted_indices)] = sorted_indices
    return result


def get_indices_from_sparse_distances(distances):
    n_samples = distances.shape[0]
    max_k = distances.getnnz(axis=1).max()
    return get_indices(
        distances.indptr, distances.indices, distances.data, n_samples, max_k
    )


@nb.njit(parallel=True)
def prune_snn(indptr, data, k, prune):
    """
    Normalize the SNN matrix and prune the values below the threshold using Jaccard similarity.
    """
    for i in nb.prange(len(indptr) - 1):
        start, end = indptr[i], indptr[i + 1]
        for j in range(start, end):
            data[j] = data[j] / (k + (k - data[j]))
            if data[j] < prune:
                data[j] = 0
    return data


def snn(distances, prune: float = 1 / 15):
    """
    Compute the Shared Nearest Neighbors (SNN) graph from a sparse KNN distance matrix

    Parameters
    ----------
    distances : scipy.sparse.csr_matrix
        The sparse distance matrix.
    prune : float, default=1/15
        The threshold to prune the SNN graph.

    Returns
    -------
    snn_graph : scipy.sparse.csr_matrix
        The SNN graph (pruned KNN graph).
    """
    # Extract KNN indices
    knn_ranked = get_indices_from_sparse_distances(distances)
    n_samples, k = knn_ranked.shape

    # Create the initial KNN sparse matrix
    rows = np.repeat(np.arange(n_samples), k)
    cols = knn_ranked.ravel()  # No need to subtract 1 if indices are already 0-based
    data = np.ones(n_samples * k, dtype=np.float64)

    knn_graph = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    # Compute shared neighbors
    snn = knn_graph.dot(knn_graph.T)

    # Normalize and prune to obtain final SNN graph
    snn.data = prune_snn(snn.indptr, snn.data, k, prune)

    # Remove zero entries to finalize SNN graph
    snn_graph = snn.eliminate_zeros()

    return snn_graph
