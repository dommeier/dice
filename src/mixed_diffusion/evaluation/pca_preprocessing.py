"""
PCA preprocessing module for dimensionality reduction before evaluation.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mixed_diffusion.helpers import ensure_numpy, ensure_tensor
import torch


class PCAPreprocessor:
    """
    A modular PCA preprocessor that can be fitted on training data
    and applied to test data for dimensionality reduction.
    """

    def __init__(self, n_components=None, standardize=True, random_state=42):
        """
        Initialize PCA preprocessor.

        Args:
            n_components: Number of components to keep. If None, keeps all components.
            standardize: Whether to standardize data before PCA
            random_state: Random state for reproducibility
        """
        self.n_components = n_components
        self.standardize = standardize
        self.random_state = random_state

        self.scaler = StandardScaler() if standardize else None
        self.pca = None
        self.is_fitted = False
        self.original_shape = None
        self.explained_variance_ratio_ = None
        self.total_variance_explained = None

    def fit(self, X_train):
        """
        Fit the PCA preprocessor on training data.

        Args:
            X_train: Training data (tensor or numpy array)

        Returns:
            self: Fitted preprocessor
        """
        # Convert to numpy
        X_train_np = ensure_numpy(X_train)

        # Store original shape info
        self.original_shape = X_train_np.shape

        # Reshape if needed (flatten all but last dimension)
        if X_train_np.ndim > 2:
            X_train_np = X_train_np.reshape(-1, X_train_np.shape[-1])

        # Standardize if requested
        if self.standardize:
            X_train_scaled = self.scaler.fit_transform(X_train_np)
        else:
            X_train_scaled = X_train_np

        # Determine number of components
        max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
        if self.n_components is None:
            n_comp = max_components
        else:
            n_comp = min(self.n_components, max_components)

        # Fit PCA
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca.fit(X_train_scaled)

        # Store variance information
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.total_variance_explained = self.explained_variance_ratio_.sum()

        self.is_fitted = True
        return self

    def transform(self, X, return_format="numpy"):
        """
        Transform data using fitted PCA.

        Args:
            X: Data to transform (tensor or numpy array)
            return_format: Format of returned data ("numpy" or "tensor")

        Returns:
            Transformed data in specified format
        """
        if not self.is_fitted:
            raise ValueError("PCA preprocessor must be fitted before transforming data")

        # Convert to numpy
        X_np = ensure_numpy(X)
        original_device = X.device if torch.is_tensor(X) else None
        original_dtype = X.dtype if torch.is_tensor(X) else None

        # Reshape if needed
        if X_np.ndim > 2:
            X_np = X_np.reshape(-1, X_np.shape[-1])

        # Standardize if needed
        if self.standardize:
            X_scaled = self.scaler.transform(X_np)
        else:
            X_scaled = X_np

        # Apply PCA transformation
        X_transformed = self.pca.transform(X_scaled)

        # Return in requested format
        if return_format == "tensor":
            return ensure_tensor(
                X_transformed, device=original_device, dtype=original_dtype
            )
        else:
            return X_transformed

    def fit_transform(self, X_train, return_format="numpy"):
        """
        Fit PCA on training data and transform it.

        Args:
            X_train: Training data to fit and transform
            return_format: Format of returned data ("numpy" or "tensor")

        Returns:
            Transformed training data
        """
        self.fit(X_train)
        return self.transform(X_train, return_format=return_format)

    def get_variance_info(self):
        """
        Get information about explained variance.

        Returns:
            dict: Dictionary containing variance information
        """
        if not self.is_fitted:
            raise ValueError("PCA preprocessor must be fitted first")

        return {
            "explained_variance_ratio": self.explained_variance_ratio_,
            "total_variance_explained": self.total_variance_explained,
            "n_components": self.pca.n_components_,
            "original_dimensions": (
                self.original_shape[-1] if self.original_shape else None
            ),
            "reduced_dimensions": self.pca.n_components_,
        }

    def print_variance_summary(self):
        """Print a summary of the PCA variance information."""
        if not self.is_fitted:
            print("PCA preprocessor not fitted yet")
            return

        info = self.get_variance_info()
        print(f"\n=== PCA Variance Summary ===")
        print(f"Original dimensions: {info['original_dimensions']}")
        print(f"Reduced dimensions: {info['reduced_dimensions']}")
        print(
            f"Total variance explained: {info['total_variance_explained']:.3f} ({info['total_variance_explained']*100:.1f}%)"
        )
        print(f"Top 5 components:")
        for i, var_ratio in enumerate(info["explained_variance_ratio"][:5]):
            print(f"  PC{i+1}: {var_ratio:.3f} ({var_ratio*100:.1f}%)")
        if len(info["explained_variance_ratio"]) > 5:
            print(
                f"  ... and {len(info['explained_variance_ratio']) - 5} more components"
            )
        print("=" * 29)


def apply_pca_preprocessing(
    train_data,
    test_data,
    n_components=None,
    standardize=True,
    return_format="numpy",
    verbose=True,
):
    """
    Convenience function to apply PCA preprocessing to train and test data.

    Args:
        train_data: Training data (tensor or numpy array)
        test_data: Test data (tensor or numpy array)
        n_components: Number of PCA components to keep
        standardize: Whether to standardize before PCA
        return_format: Format of returned data ("numpy" or "tensor")
        verbose: Whether to print variance summary

    Returns:
        tuple: (train_transformed, test_transformed, pca_preprocessor)
    """
    # Initialize and fit PCA
    pca_preprocessor = PCAPreprocessor(
        n_components=n_components, standardize=standardize
    )

    # Fit on training data and transform both datasets
    train_transformed = pca_preprocessor.fit_transform(
        train_data, return_format=return_format
    )
    test_transformed = pca_preprocessor.transform(
        test_data, return_format=return_format
    )

    if verbose:
        pca_preprocessor.print_variance_summary()

    return train_transformed, test_transformed, pca_preprocessor
