"""
Noise variance correction module for single-cell data preprocessing.

This module provides functionality to correct for noise variance in high-dimensional
single-cell data using SVD-based noise estimation and Marchenko-Pastur law validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import warnings


class NoiseVarianceCorrector:
    """
    A class for correcting noise variance in single-cell data using SVD-based methods.

    This implements noise variance correction based on the residual after removing
    the top principal components, following the Marchenko-Pastur law for validation.
    """

    def __init__(self, n_components: int = 20, plot_validation: bool = True):
        """
        Initialize the noise variance corrector.

        Args:
            n_components: Number of top principal components to use for denoising
            plot_validation: Whether to plot validation against Marchenko-Pastur law
        """
        self.n_components = n_components
        self.plot_validation = plot_validation
        self.fitted = False
        self.tau_sq_correct = None
        self.U_components = None
        self.S_components = None
        self.V_components = None

    def _marchenko_pastur_sqrt_law(
        self, x: np.ndarray, n_samples: int, n_features: int
    ) -> np.ndarray:
        """
        Square root of Marchenko-Pastur law density.

        Args:
            x: Input values (squared singular values)
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            Marchenko-Pastur density values
        """
        gamma = n_samples / n_features
        lambda_plus = (1 + np.sqrt(gamma)) ** 2
        lambda_minus = (1 - np.sqrt(gamma)) ** 2

        density = np.zeros_like(x)
        mask = (x >= lambda_minus) & (x <= lambda_plus)
        density[mask] = np.sqrt(gamma / (2 * np.pi * x[mask])) * np.sqrt(
            (lambda_plus - x[mask]) * (x[mask] - lambda_minus)
        )

        return density

    def fit(self, X: np.ndarray) -> "NoiseVarianceCorrector":
        """
        Fit the noise variance corrector on the input data.

        Args:
            X: Input data matrix (samples x features)

        Returns:
            self for method chaining
        """
        # Perform SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Store components for later use
        self.U_components = U
        self.S_components = S
        self.V_components = Vt

        # Calculate residual after removing top components
        k = self.n_components
        n_features = X.shape[1]
        n_samples = X.shape[0]

        # Reconstruct using top k components
        X_reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

        # Calculate residual
        X_residual = X - X_reconstructed

        # Calculate noise variance correction factor (tau_sq_correct)
        self.tau_sq_correct = np.sum(X_residual**2) / (n_features * n_samples)

        # Normalize residual by noise variance
        X_residual_normalized = X_residual / np.sqrt(self.tau_sq_correct)

        # Perform SVD on normalized residual for validation
        U_res, S_res, Vt_res = np.linalg.svd(X_residual_normalized, full_matrices=False)

        if self.plot_validation:
            self._plot_validation(S_res, n_samples, n_features)

        self.fitted = True
        return self

    def _plot_validation(
        self, S_residual: np.ndarray, n_samples: int, n_features: int
    ) -> None:
        """
        Plot validation of noise correction against Marchenko-Pastur law.

        Args:
            S_residual: Singular values of the residual matrix
            n_samples: Number of samples
            n_features: Number of features
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate squared singular values
        sq_singular_val = S_residual**2
        shorter_side = min(n_samples, n_features)

        # Plot histogram of sample singular values
        ax.hist(
            sq_singular_val[:shorter_side],
            density=True,
            bins=50,
            alpha=0.7,
            label="Sample singular values",
        )

        # Plot Marchenko-Pastur law prediction
        x = np.linspace(sq_singular_val.min(), sq_singular_val.max(), num=500)
        aspect_ratio = n_features / n_samples
        scaler = aspect_ratio if aspect_ratio > 1 else 1

        mp_density = scaler * self._marchenko_pastur_sqrt_law(x, n_samples, n_features)
        ax.plot(x, mp_density, "r-", linewidth=2, label="MP law prediction")

        ax.legend()
        ax.set_title(f"Noise Variance Validation (n_components={self.n_components})")
        ax.set_xlabel("Squared Singular Values")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"Noise variance correction factor (tau_sq): {self.tau_sq_correct:.6f}")
        print(f"Standard deviation of noise: {np.sqrt(self.tau_sq_correct):.6f}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply noise variance correction to input data.

        Args:
            X: Input data matrix

        Returns:
            Noise-corrected data matrix
        """
        if not self.fitted:
            raise ValueError(
                "NoiseVarianceCorrector must be fitted before transformation."
            )

        # Use the stored SVD components to reconstruct and correct
        k = self.n_components

        # Reconstruct using top k components
        X_reconstructed = (
            self.U_components[: X.shape[0], :k]
            @ np.diag(self.S_components[:k])
            @ self.V_components[:k, :]
        )

        # Calculate residual
        X_residual = X - X_reconstructed

        # Apply noise correction
        X_corrected = X_reconstructed + X_residual / np.sqrt(self.tau_sq_correct)

        return X_corrected

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the corrector and transform the data in one step.

        Args:
            X: Input data matrix

        Returns:
            Noise-corrected data matrix
        """
        self.fit(X)
        return self.transform(X)

    def get_noise_statistics(self) -> Dict[str, float]:
        """
        Get noise correction statistics.

        Returns:
            Dictionary containing noise statistics
        """
        if not self.fitted:
            raise ValueError("NoiseVarianceCorrector must be fitted first.")

        return {
            "tau_squared": self.tau_sq_correct,
            "noise_std": np.sqrt(self.tau_sq_correct),
            "n_components_used": self.n_components,
        }

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the explained variance ratio of the top components.

        Returns:
            Explained variance ratios
        """
        if not self.fitted:
            raise ValueError("NoiseVarianceCorrector must be fitted first.")

        total_variance = np.sum(self.S_components**2)
        explained_variance = self.S_components**2 / total_variance

        return explained_variance

    def plot_singular_values(self, n_values: int = 100) -> None:
        """
        Plot the first n singular values to help choose n_components.

        Args:
            n_values: Number of singular values to plot
        """
        if not self.fitted:
            raise ValueError("NoiseVarianceCorrector must be fitted first.")

        plt.figure(figsize=(10, 6))

        # Plot singular values
        plot_values = self.S_components[: min(n_values, len(self.S_components))]
        plt.subplot(1, 2, 1)
        plt.plot(plot_values, "b.-", markersize=4)
        plt.axvline(
            x=self.n_components - 1,
            color="r",
            linestyle="--",
            label=f"n_components={self.n_components}",
        )
        plt.xlabel("Component Index")
        plt.ylabel("Singular Value")
        plt.title("Singular Values")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot explained variance ratio
        plt.subplot(1, 2, 2)
        explained_var = self.get_explained_variance_ratio()
        cumsum_var = np.cumsum(explained_var[: min(n_values, len(explained_var))])
        plt.plot(cumsum_var, "g.-", markersize=4)
        plt.axvline(
            x=self.n_components - 1,
            color="r",
            linestyle="--",
            label=f"Cumulative variance at n_components: {cumsum_var[self.n_components-1]:.3f}",
        )
        plt.xlabel("Component Index")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Cumulative Explained Variance")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def correct_noise_variance(
    X: np.ndarray, n_components: int = 20, plot_validation: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to apply noise variance correction to data.

    Args:
        X: Input data matrix (samples x features)
        n_components: Number of top principal components for denoising
        plot_validation: Whether to plot validation

    Returns:
        Tuple of (corrected_data, correction_info)
    """
    corrector = NoiseVarianceCorrector(
        n_components=n_components, plot_validation=plot_validation
    )
    X_corrected = corrector.fit_transform(X)

    correction_info = {
        "noise_statistics": corrector.get_noise_statistics(),
        "explained_variance_ratio": corrector.get_explained_variance_ratio(),
        "corrector": corrector,
    }

    return X_corrected, correction_info


def analyze_noise_components(
    X: np.ndarray, max_components: int = 50, plot_results: bool = True
) -> Dict[str, np.ndarray]:
    """
    Analyze the effect of different numbers of components on noise correction.

    Args:
        X: Input data matrix
        max_components: Maximum number of components to test
        plot_results: Whether to plot the analysis results

    Returns:
        Dictionary with analysis results
    """
    component_range = range(5, min(max_components + 1, min(X.shape) // 2), 5)
    noise_variances = []
    explained_variances = []

    # Perform SVD once
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    total_variance = np.sum(S**2)

    for n_comp in component_range:
        # Calculate residual variance for this number of components
        X_reconstructed = U[:, :n_comp] @ np.diag(S[:n_comp]) @ Vt[:n_comp, :]
        X_residual = X - X_reconstructed
        tau_sq = np.sum(X_residual**2) / (X.shape[0] * X.shape[1])

        noise_variances.append(tau_sq)

        # Calculate explained variance
        explained_var = np.sum(S[:n_comp] ** 2) / total_variance
        explained_variances.append(explained_var)

    if plot_results:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(component_range, noise_variances, "b.-", markersize=6)
        plt.xlabel("Number of Components")
        plt.ylabel("Noise Variance (τ²)")
        plt.title("Noise Variance vs Number of Components")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(component_range, explained_variances, "r.-", markersize=6)
        plt.xlabel("Number of Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Explained Variance vs Number of Components")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "component_range": np.array(component_range),
        "noise_variances": np.array(noise_variances),
        "explained_variances": np.array(explained_variances),
    }
