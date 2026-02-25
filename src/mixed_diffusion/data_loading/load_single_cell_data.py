import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset

from mixed_diffusion.preprocessing.data_transformer import DataTransformer


def load_single_cell_data(args):
    """
    Load single cell data based on a JSON configuration file and return as PyTorch tensors.

    Args:
        args: Arguments object containing config_file path.

    Returns:
        tuple: (train_data, test_data, data_config) where:
            - train_data: tuple of (train_data_tensor, train_labels_tensor)
            - test_data: tuple of (test_data_tensor, test_labels_tensor)
            - data_config: dict with config and label_encoder
    """
    # Load configuration
    with open(args["data_path"], "r") as f:
        config = json.load(f)
    print(f"Loading single cell data from config: {args['data_path']}")

    data_transformer = DataTransformer(args["data_path"])
    X_train, y_train, X_test, y_test = data_transformer.transform_all_splits()

    # Print summary information
    print(f"\n=== Data Summary ===")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Print label distribution
    if "x" in y_train.columns:
        print(f"\nTraining label distribution:")
        print(y_train["x"].value_counts().sort_index())
        print(f"\nTest label distribution:")
        print(y_test["x"].value_counts().sort_index())

    # Validate data consistency
    assert X_train.shape[0] == len(
        y_train
    ), f"Training data and labels size mismatch: {X_train.shape[0]} vs {len(y_train)}"
    assert X_test.shape[0] == len(
        y_test
    ), f"Test data and labels size mismatch: {X_test.shape[0]} vs {len(y_test)}"
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), f"Training and test data feature dimension mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"

    print(f"✓ Data validation passed")

    # Convert data to tensors
    train_data_tensor = torch.from_numpy(X_train).float()
    test_data_tensor = torch.from_numpy(X_test).float()


    # Encode labels as integers if they exist
    label_encoder = {}
    train_labels_tensor = None
    test_labels_tensor = None

    if "x" in y_train.columns:
        # Get unique labels and create encoding
        all_labels = pd.concat([y_train["x"], y_test["x"]]).astype(str).unique()

        label_encoder = {label: idx for idx, label in enumerate(sorted(all_labels))}

        # Fix: Convert the original labels to strings to match the label_encoder keys
        train_labels_encoded = y_train["x"].astype(str).map(label_encoder).values

        test_labels_encoded = y_test["x"].astype(str).map(label_encoder).values

        # Check for NaN values and throw error if found
        if pd.isna(train_labels_encoded).any():
            raise ValueError(
                "Found NaN values in train_labels_encoded after label encoding. Check if all training labels exist in the label encoder."
            )

        if pd.isna(test_labels_encoded).any():
            raise ValueError(
                "Found NaN values in test_labels_encoded after label encoding. Check if all test labels exist in the label encoder."
            )

        train_labels_tensor = torch.from_numpy(train_labels_encoded).long()
        test_labels_tensor = torch.from_numpy(test_labels_encoded).long()

        print(f"\nLabel encoding:")
        for label, idx in label_encoder.items():
            print(f"  {label} -> {idx}")

    print(f"\n=== Tensor Summary ===")
    print(f"Training data tensor shape: {train_data_tensor.shape}")
    print(f"Test data tensor shape: {test_data_tensor.shape}")
    if train_labels_tensor is not None:
        print(f"Training labels tensor shape: {train_labels_tensor.shape}")
        print(f"Test labels tensor shape: {test_labels_tensor.shape}")

    # Create return tuples
    train_data = TensorDataset(train_data_tensor, train_labels_tensor)
    test_data = TensorDataset(test_data_tensor, test_labels_tensor)
    data_config = {
        "config": config,
        "label_encoder": label_encoder,
        "observation_transform_matrix": data_transformer.get_pca_observation_matrix(),
    }

    return train_data, test_data, data_config
