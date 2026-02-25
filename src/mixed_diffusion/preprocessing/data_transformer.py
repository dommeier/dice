"""
Data transformation module for mixed diffusion preprocessing.

This module provides functionality to transform datasets based on JSON configuration files
that specify preprocessing pipelines including StandardScaler, PCA, and other transformations.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle
import warnings


class DataTransformer:
    """
    A class to transform datasets based on JSON configuration files.

    The JSON configuration should have the following structure:
    {
        "labels": {
            "train": "path/to/train_labels.csv",
            "test": "path/to/test_labels.csv"
        },
        "data": {
            "train": "path/to/train_data.npy",
            "test": "path/to/test_data.npy"
        },
        "observation_transform": [  # Optional - observation-level transformation
            "PCA", {"n_components": 25, "random_state": 42}
        ],
        "fit_observation_on": "train",  # Optional - "train" or "test", defaults to "train"
        "pipeline": [  # Optional - can be empty list or omitted for no transformation
            ["StandardScaler", {}],
            ["PCA", {"n_components": 20, "random_state": 42}]
        ]
    }

    Transformation order:
    1. Pipeline transforms are applied first (e.g., StandardScaler, PCA)
    2. Observation transforms are applied to the pipeline-transformed data

    The observation_transform creates an A matrix (e.g., PCA components) that can be
    extracted and used for data transformation in the observation space.
    """

    def __init__(self, config_path: str):
        """
        Initialize the DataTransformer with a JSON configuration file.

        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.base_dir = os.path.dirname(config_path)
        self.pipeline = None
        self.observation_transformer = None
        self.fitted = False

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the JSON configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

        # Validate required keys
        required_keys = ["labels", "data"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

        # Pipeline is optional - if not provided or empty, no transformations will be applied
        if "pipeline" not in config:
            config["pipeline"] = []

        # fit_observation_on is optional - defaults to "train"
        if "fit_observation_on" not in config:
            config["fit_observation_on"] = "train"
        elif config["fit_observation_on"] not in ["train", "test"]:
            raise ValueError(
                f"fit_observation_on must be 'train' or 'test', got: {config['fit_observation_on']}"
            )

        return config

    def _get_full_path(self, relative_path: str) -> str:
        """Convert relative path to full path based on config file location."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.base_dir, relative_path)

    def _create_pipeline(self) -> Optional[Pipeline]:
        """Create sklearn pipeline from configuration."""
        steps = []

        for step_name, step_params in self.config["pipeline"]:
            if step_name == "StandardScaler":
                transformer = StandardScaler(**step_params)
            elif step_name == "PCA":
                transformer = PCA(**step_params)
            else:
                raise ValueError(f"Unsupported transformer: {step_name}")

            steps.append((step_name.lower(), transformer))

        # Return None if no steps, indicating identity transformation
        if not steps:
            return None

        return Pipeline(steps)

    def _create_observation_transformer(self) -> Optional[object]:
        """Create observation transformer from configuration."""
        if "observation_transform" not in self.config:
            return None

        transform_config = self.config["observation_transform"]
        if not transform_config:
            return None

        transform_name, transform_params = transform_config

        if transform_name == "PCA":
            return PCA(**transform_params)
        elif transform_name == "StandardScaler":
            return StandardScaler(**transform_params)
        else:
            raise ValueError(f"Unsupported observation transformer: {transform_name}")

    def _validate_pca_observation_transform(self) -> None:
        """
        Validate that the observation transform is configured as PCA.

        Raises:
            ValueError: If observation transform is not PCA or not configured
        """
        if "observation_transform" not in self.config:
            raise ValueError("No observation_transform configured in config.")

        transform_config = self.config["observation_transform"]
        if not transform_config:
            raise ValueError("observation_transform is empty.")

        transform_name, _ = transform_config
        if transform_name != "PCA":
            raise ValueError(
                f"observation_transform is configured as '{transform_name}', not 'PCA'. "
                f"PCA-specific methods require observation_transform to be PCA."
            )

    def load_data(self, split: str = "both") -> Union[
        Tuple[np.ndarray, pd.DataFrame],
        Tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    ]:
        """
        Load data and labels for specified split(s).

        Args:
            split: 'train', 'test', or 'both'

        Returns:
            If split is 'train' or 'test': (data, labels)
            If split is 'both': (train_data, train_labels, test_data, test_labels)
        """

        def _load_split_data(split_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
            # Load data
            data_path = self._get_full_path(self.config["data"][split_name])
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            data = np.load(data_path)

            # Load labels
            labels_path = self._get_full_path(self.config["labels"][split_name])
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            labels = pd.read_csv(labels_path)

            return data, labels

        if split == "train":
            return _load_split_data("train")
        elif split == "test":
            return _load_split_data("test")
        elif split == "both":
            train_data, train_labels = _load_split_data("train")
            test_data, test_labels = _load_split_data("test")
            return train_data, train_labels, test_data, test_labels
        else:
            raise ValueError("split must be 'train', 'test', or 'both'")

    def fit(
        self,
        X_train: Optional[np.ndarray] = None,
        fit_observation_on: Optional[str] = None,
    ) -> "DataTransformer":
        """
        Fit the preprocessing pipeline on training data, and optionally fit the observation_transform on train or test data.

        Args:
            X_train: Training data. If None, will load from config.
            fit_observation_on: Which split to fit the observation_transformer on ("train" or "test").
                               If None, uses value from config (defaults to "train").

        Returns:
            self for method chaining
        """
        if X_train is None:
            X_train, _ = self.load_data("train")

        # Use config value if fit_observation_on not explicitly provided
        if fit_observation_on is None:
            fit_observation_on = self.config.get("fit_observation_on", "train")

        # Create and fit the main pipeline first
        self.pipeline = self._create_pipeline()

        # Transform data with pipeline if available
        if self.pipeline is not None:
            self.pipeline.fit(X_train)
            X_transformed = self.pipeline.transform(X_train)
        else:
            X_transformed = X_train

        # Optionally fit observation transformer on train or test split
        if fit_observation_on == "train":
            obs_data = X_transformed
        elif fit_observation_on == "test":
            X_test, _ = self.load_data("test")
            print("Fitting observation transform (\hat\{V\}) on test data...")
            obs_data = (
                self.pipeline.transform(X_test) if self.pipeline is not None else X_test
            )
        else:
            raise ValueError("fit_observation_on must be 'train' or 'test'")

        self.observation_transformer = self._create_observation_transformer()
        if self.observation_transformer is not None:
            self.observation_transformer.fit(obs_data)

        self.fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted pipeline.

        Args:
            X: Data to transform

        Returns:
            Transformed data (or original data if pipeline is empty)
        """
        if not self.fitted:
            raise ValueError(
                "Pipeline must be fitted before transformation. Call fit() first."
            )

        # If pipeline is None (empty), return data unchanged
        if self.pipeline is None:
            return X

        return self.pipeline.transform(X)

    def fit_transform(
        self,
        X_train: Optional[np.ndarray] = None,
        fit_observation_on: Optional[str] = None,
    ) -> np.ndarray:
        """
        Fit pipeline on training data and transform it. Optionally fit observation_transform on train or test data.

        Args:
            X_train: Training data. If None, will load from config.
            fit_observation_on: Which split to fit the observation_transformer on ("train" or "test").
                               If None, uses value from config (defaults to "train").

        Returns:
            Transformed training data
        """
        if X_train is None:
            X_train, _ = self.load_data("train")

        self.fit(X_train, fit_observation_on=fit_observation_on)
        return self.transform(X_train)

    def transform_with_observation(self, X: np.ndarray) -> np.ndarray:
        """
        Apply both observation transform and pipeline transform to data.

        Args:
            X: Data to transform

        Returns:
            Fully transformed data
        """
        if not self.fitted:
            raise ValueError(
                "Pipeline must be fitted before transformation. Call fit() first."
            )

        # First apply pipeline transform if available
        if self.pipeline is not None:
            X_transformed = self.pipeline.transform(X)
        else:
            X_transformed = X

        # Then apply observation transform if available
        if self.observation_transformer is not None:
            X_transformed = self.observation_transformer.transform(X_transformed)

        return X_transformed

    def transform_all_splits(
        self,
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Load all data, fit on training data, and transform both splits.

        Returns:
            (X_train_transformed, y_train, X_test_transformed, y_test)
        """
        # Load all data
        X_train, y_train, X_test, y_test = self.load_data("both")

        # Fit and transform training data
        X_train_transformed = self.fit_transform(X_train)

        # Transform test data
        X_test_transformed = self.transform(X_test)

        return X_train_transformed, y_train, X_test_transformed, y_test

    def transform_all_splits_with_observation(
        self,
    ) -> Tuple[
        np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, Optional[np.ndarray]
    ]:
        """
        Load all data, fit on training data, and transform both splits with observation transform.

        Returns:
            (X_train_transformed, y_train, X_test_transformed, y_test, observation_matrix)
        """
        # Load all data
        X_train, y_train, X_test, y_test = self.load_data("both")

        # Fit transformers
        self.fit(X_train)

        # Transform training data with observation transform
        X_train_transformed = self.transform_with_observation(X_train)

        # Transform test data with observation transform
        X_test_transformed = self.transform_with_observation(X_test)

        # Get observation transform matrix (A matrix)
        observation_matrix = self.get_observation_transform_matrix()

        return (
            X_train_transformed,
            y_train,
            X_test_transformed,
            y_test,
            observation_matrix,
        )

    def get_observation_transform_matrix(self) -> Optional[np.ndarray]:
        """
        Get the observation transform matrix (A matrix) if available.

        Returns:
            The observation transform matrix (e.g., PCA components) or None if not available
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        if self.observation_transformer is None:
            return None

        if isinstance(self.observation_transformer, PCA):
            return self.observation_transformer.components_
        else:
            # For other transformers that might not have a meaningful matrix representation
            return None

    def get_pca_observation_matrix(self) -> np.ndarray:
        """
        Get the PCA observation transform matrix that transforms high-dimensional data to latent PCA space.

        This method specifically requires that the observation transform is PCA and throws an error otherwise.
        The returned matrix can be used to transform data from the high-dimensional space to the PCA latent space.

        Returns:
            PCA components matrix (shape: n_components x n_features) that transforms
            high-dimensional data to PCA latent representation

        Raises:
            ValueError: If pipeline is not fitted, no observation transformer exists,
                       or observation transformer is not PCA
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        # Validate that observation transform is PCA
        self._validate_pca_observation_transform()

        if self.observation_transformer is None:
            raise ValueError(
                "Observation transformer is None despite PCA configuration."
            )

        if not isinstance(self.observation_transformer, PCA):
            # This should not happen if validation passed, but extra safety check
            transformer_type = type(self.observation_transformer).__name__
            raise ValueError(
                f"Observation transformer is {transformer_type}, not PCA. "
                f"This should not happen after validation."
            )

        return self.observation_transformer.components_

    def get_pca_observation_info(self) -> Dict[str, Any]:
        """
        Get comprehensive PCA observation transform information.

        Returns:
            Dictionary containing:
            - 'components': PCA components matrix (n_components x n_features)
            - 'explained_variance_ratio': Explained variance ratio for each component
            - 'explained_variance': Explained variance for each component
            - 'n_components': Number of PCA components
            - 'n_features': Number of original features

        Raises:
            ValueError: If pipeline is not fitted, no observation transformer exists,
                       or observation transformer is not PCA
        """
        # Reuse the validation from get_pca_observation_matrix
        components = self.get_pca_observation_matrix()

        pca = self.observation_transformer

        return {
            "components": components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance": pca.explained_variance_,
            "n_components": pca.n_components_,
            "n_features": components.shape[1],
            "mean": pca.mean_,  # PCA mean for centering
        }

    def apply_observation_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the observation transform to data.

        Args:
            X: Data to transform using observation transformer

        Returns:
            Transformed data or original data if no observation transformer
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        if self.observation_transformer is None:
            return X

        return self.observation_transformer.transform(X)

    def apply_pipeline_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the pipeline transform to data.

        Args:
            X: Data to transform using pipeline

        Returns:
            Transformed data or original data if no pipeline
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        if self.pipeline is None:
            return X

        return self.pipeline.transform(X)

    def inverse_observation_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply inverse observation transform to data.

        Args:
            X: Transformed data to inverse transform

        Returns:
            Data in original space or original data if no observation transformer
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        if self.observation_transformer is None:
            return X

        try:
            return self.observation_transformer.inverse_transform(X)
        except AttributeError:
            raise ValueError(
                "Observation transformer does not support inverse transformation."
            )

    def save_pipeline(self, path: str) -> None:
        """
        Save the fitted pipeline to disk.

        Args:
            path: Path to save the pipeline
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before saving.")

        # Save both the pipeline, observation transformer, and flags
        pipeline_data = {
            "pipeline": self.pipeline,
            "is_empty": self.pipeline is None,
            "observation_transformer": self.observation_transformer,
            "has_observation_transform": self.observation_transformer is not None,
        }

        with open(path, "wb") as f:
            pickle.dump(pipeline_data, f)

    def load_pipeline(self, path: str) -> "DataTransformer":
        """
        Load a fitted pipeline from disk.

        Args:
            path: Path to the saved pipeline

        Returns:
            self for method chaining
        """
        with open(path, "rb") as f:
            pipeline_data = pickle.load(f)

        # Handle both old format (just pipeline) and new format (dict with metadata)
        if isinstance(pipeline_data, dict):
            self.pipeline = pipeline_data["pipeline"]
            # Load observation transformer if available
            self.observation_transformer = pipeline_data.get(
                "observation_transformer", None
            )
        else:
            # Backward compatibility with old format
            self.pipeline = pipeline_data
            self.observation_transformer = None

        self.fitted = True
        return self

    def get_feature_names_out(self) -> Optional[List[str]]:
        """
        Get feature names after transformation if available.

        Returns:
            List of feature names or None if not available
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        # If no pipeline (empty), return None
        if self.pipeline is None:
            return None

        try:
            return self.pipeline.get_feature_names_out()
        except AttributeError:
            # Some transformers might not have this method
            return None

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio if PCA is in the pipeline.

        Returns:
            Explained variance ratios or None if PCA not found or no pipeline
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        # If no pipeline (empty), return None
        if self.pipeline is None:
            return None

        for name, step in self.pipeline.named_steps.items():
            if isinstance(step, PCA):
                return step.explained_variance_ratio_

        return None

    def get_components(self) -> Optional[np.ndarray]:
        """
        Get PCA components if PCA is in the pipeline.

        Returns:
            PCA components or None if PCA not found or no pipeline
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        # If no pipeline (empty), return None
        if self.pipeline is None:
            return None

        for name, step in self.pipeline.named_steps.items():
            if isinstance(step, PCA):
                return step.components_

        return None

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data back to original space if possible.

        Args:
            X: Transformed data

        Returns:
            Data in original space
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first.")

        # If no pipeline (empty), return data unchanged
        if self.pipeline is None:
            return X

        try:
            return self.pipeline.inverse_transform(X)
        except AttributeError:
            raise ValueError("Pipeline does not support inverse transformation.")

    def __str__(self) -> str:
        """String representation of the transformer."""
        return f"DataTransformer(config='{self.config_path}', fitted={self.fitted})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


def transform_dataset_from_config(
    config_path: str, save_transformed: bool = False, output_dir: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Convenience function to transform a dataset using a JSON configuration.

    Args:
        config_path: Path to the JSON configuration file
        save_transformed: Whether to save transformed data to disk
        output_dir: Directory to save transformed data (defaults to config directory)

    Returns:
        (X_train_transformed, y_train, X_test_transformed, y_test)
    """
    transformer = DataTransformer(config_path)
    X_train_transformed, y_train, X_test_transformed, y_test = (
        transformer.transform_all_splits()
    )

    if save_transformed:
        if output_dir is None:
            output_dir = os.path.dirname(config_path)

        # Save transformed data
        np.save(
            os.path.join(output_dir, "X_train_transformed.npy"), X_train_transformed
        )
        np.save(os.path.join(output_dir, "X_test_transformed.npy"), X_test_transformed)

        # Save pipeline
        transformer.save_pipeline(
            os.path.join(output_dir, "preprocessing_pipeline.pkl")
        )

        print(f"Transformed data saved to {output_dir}")

    return X_train_transformed, y_train, X_test_transformed, y_test
