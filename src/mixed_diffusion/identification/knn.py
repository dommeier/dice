import numpy as np

from mixed_diffusion.helpers import ensure_numpy


def knn(x, atlas, k=5):
    """Apply k-NN to the input data."""
    from sklearn.neighbors import NearestNeighbors

    # Ensure x and atlas are numpy arrays
    x = ensure_numpy(x)

    reference_x, labels = atlas

    reference_x = ensure_numpy(reference_x)
    labels = ensure_numpy(labels)

    # Fit the k-NN model
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(reference_x, labels)

    # Find the k nearest neighbors
    distances, indices = knn_model.kneighbors(x)

    return indices, distances
