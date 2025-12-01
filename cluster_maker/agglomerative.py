###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional, List, Dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


def hierarchical_clustering(
    X: np.ndarray,
    n_clusters: int,
    linkage_method: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical agglomerative clustering using scikit-learn.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_clusters : int
        Number of clusters to form.
    linkage_method : {"ward", "complete", "average", "single"}, default "ward"
        Linkage criterion:
        - "ward": minimizes variance (only with Euclidean distance)
        - "complete": maximum distance between clusters
        - "average": average distance between clusters
        - "single": minimum distance between clusters

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    model : AgglomerativeClustering
        Fitted agglomerative clustering model.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > X.shape[0]:
        raise ValueError("n_clusters cannot be larger than the number of samples.")

    valid_linkages = {"ward", "complete", "average", "single"}
    if linkage_method not in valid_linkages:
        raise ValueError(
            f"linkage_method must be one of {valid_linkages}, got '{linkage_method}'."
        )

    # Create and fit model
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
    )
    labels = model.fit_predict(X)

    return labels, model


def compare_linkage_methods(
    X: np.ndarray,
    n_clusters: int,
    linkage_methods: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compare different linkage methods for hierarchical clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_clusters : int
    linkage_methods : list of str or None, default None
        Linkage methods to test. If None, tests all four standard methods:
        ["ward", "complete", "average", "single"]

    Returns
    -------
    results : dict
        Dictionary mapping linkage method names to their cluster labels.
    """
    if linkage_methods is None:
        linkage_methods = ["ward", "complete", "average", "single"]

    results: Dict[str, np.ndarray] = {}
    for method in linkage_methods:
        labels, _ = hierarchical_clustering(X, n_clusters, linkage_method=method)
        results[method] = labels

    return results


def get_linkage_matrix(
    X: np.ndarray,
    linkage_method: str = "ward",
) -> np.ndarray:
    """
    Compute the linkage matrix for hierarchical clustering.

    This is useful for creating dendrograms and analyzing the clustering hierarchy.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    linkage_method : {"ward", "complete", "average", "single"}, default "ward"

    Returns
    -------
    Z : ndarray of shape (n_samples-1, 4)
        The hierarchical clustering encoded as a linkage matrix.
        Rows correspond to merges, columns are:
        [cluster_id_1, cluster_id_2, distance, sample_count]
    """
    valid_linkages = {"ward", "complete", "average", "single"}
    if linkage_method not in valid_linkages:
        raise ValueError(
            f"linkage_method must be one of {valid_linkages}, got '{linkage_method}'."
        )

    if linkage_method == "ward":
        method = "ward"
    else:
        method = linkage_method

    Z = linkage(X, method=method)
    return Z


def test_cluster_count_range(
    X: np.ndarray,
    k_values: List[int],
    linkage_method: str = "ward",
) -> Dict[int, np.ndarray]:
    """
    Test hierarchical clustering for multiple cluster counts.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k_values : list of int
        Different numbers of clusters to test.
    linkage_method : str, default "ward"

    Returns
    -------
    results : dict
        Dictionary mapping k (number of clusters) to cluster labels.
    """
    results: Dict[int, np.ndarray] = {}
    for k in k_values:
        labels, _ = hierarchical_clustering(X, k, linkage_method=linkage_method)
        results[k] = labels

    return results
