#!/usr/bin/env python3
"""
scVI Benchmark Script

This script runs scVI on preprocessed single-cell data and saves results
in an organized folder structure under /results/dataset/scVI.

Usage:
    python scripts/run_scvi_benchmark.py --config data/neo_cortex/raw_data.json --dataset neo_cortex
    python scripts/run_scvi_benchmark.py --config data/synthetic/config.json --dataset synthetic
"""

import argparse
import os
import sys
from pathlib import Path
import scanpy as sc
import scvi
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import matplotlib.pyplot as plt
import torch
import anndata as ad
import json

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mixed_diffusion.preprocessing.data_transformer import DataTransformer


def setup_anndata_for_scvi(X_train, y_train, X_test, y_test):
    """
    Create AnnData objects from transformed data and prepare for scVI.
    """
    print(f"Setting up AnnData objects...")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Create training AnnData object
    adata_train = ad.AnnData(X=X_train)

    # Add cell type labels to observations
    if isinstance(y_train, pd.DataFrame):
        # If y_train is a DataFrame, add all columns as observations
        for col in y_train.columns:
            adata_train.obs[col] = y_train[col].values
        # Use the first column as the main cell type label
        main_label_col = y_train.columns[0]
        adata_train.obs["celltype"] = y_train[main_label_col].values
    else:
        # If y_train is a Series or array, add as celltype
        adata_train.obs["celltype"] = y_train

    # Create test AnnData object
    adata_test = ad.AnnData(X=X_test)

    # Add cell type labels to test observations
    if isinstance(y_test, pd.DataFrame):
        for col in y_test.columns:
            adata_test.obs[col] = y_test[col].values
        adata_test.obs["celltype"] = y_test[main_label_col].values
    else:
        adata_test.obs["celltype"] = y_test

    print(f"adata_train: {adata_train.shape}")
    print(f"adata_test: {adata_test.shape}")
    print(f"Training cell types: {adata_train.obs['celltype'].nunique()} unique")
    print(f"Test cell types: {adata_test.obs['celltype'].nunique()} unique")

    return adata_train, adata_test


def prepare_count_data(adata_train, adata_test, X_train, X_test):
    """
    Prepare count data for scVI by handling log-transformed data.
    """
    print("Preparing count data for scVI...")
    print(f"X_train min: {X_train.min():.3f}, max: {X_train.max():.3f}")
    print(f"X_train mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")

    # If data appears to be log-transformed (small values, often negative),
    # we may need to reverse the transformation for scVI
    if X_train.min() < 0 or X_train.max() < 20:
        print("Data appears to be log-transformed. Attempting to reverse transform...")

        # Store the transformed data in layers
        adata_train.layers["transformed"] = X_train.copy()
        adata_test.layers["transformed"] = X_test.copy()

        try:
            # Reverse log1p: expm1(x) = exp(x) - 1
            X_train_counts = np.expm1(X_train)
            X_test_counts = np.expm1(X_test)

            # Ensure non-negative (in case of numerical issues)
            X_train_counts = np.maximum(X_train_counts, 0)
            X_test_counts = np.maximum(X_test_counts, 0)

            # Convert to integers for count data (round and convert to int)
            X_train_counts = np.round(X_train_counts).astype(np.int32)
            X_test_counts = np.round(X_test_counts).astype(np.int32)

            # Store as counts layer
            adata_train.layers["counts"] = X_train_counts
            adata_test.layers["counts"] = X_test_counts

            print(
                f"Raw counts (integers) - min: {X_train_counts.min()}, max: {X_train_counts.max()}"
            )
            print(f"Data type: {X_train_counts.dtype}")
            print("✓ Added 'counts' layer with raw integer count data")

        except Exception as e:
            print(f"Could not reverse transform: {e}")
            print("Using transformed data as counts (converted to integers)")
            # Convert transformed data to integers as best effort
            X_train_int = np.round(np.maximum(X_train, 0)).astype(np.int32)
            X_test_int = np.round(np.maximum(X_test, 0)).astype(np.int32)
            adata_train.layers["counts"] = X_train_int
            adata_test.layers["counts"] = X_test_int
    else:
        print("Data appears to be raw counts already")
        # Convert to integers and ensure non-negative
        X_train_int = np.round(np.maximum(X_train, 0)).astype(np.int32)
        X_test_int = np.round(np.maximum(X_test, 0)).astype(np.int32)
        adata_train.layers["counts"] = X_train_int
        adata_test.layers["counts"] = X_test_int
        print(
            f"Converted to integers - min: {X_train_int.min()}, max: {X_train_int.max()}"
        )

    print(f"Count data type: {adata_train.layers['counts'].dtype}")
    return adata_train, adata_test


def train_scvi_model(adata_train, n_latent=25, max_epochs=400):
    """
    Setup and train scVI model.
    """
    print("Setting up scVI model...")

    # Setup scVI
    scvi.model.SCVI.setup_anndata(
        adata_train,
        layer="counts",  # Use the counts layer we created
        batch_key=None,  # Set to None if no batch information available
        labels_key="celltype",  # Use our celltype column
    )

    # Create and train scVI model
    model = scvi.model.SCVI(adata_train, n_latent=n_latent, gene_likelihood="zinb")

    print(f"Training scVI model (n_latent={n_latent}, max_epochs={max_epochs})...")
    model.train(
        max_epochs=max_epochs,
        early_stopping=True,
        early_stopping_patience=25,
        plan_kwargs={"lr": 1e-3},
        accelerator="mps" if torch.backends.mps.is_available() else "auto",
    )

    print("✓ scVI model training completed")
    return model


def evaluate_scvi_model(model, adata_train, adata_test):
    """
    Evaluate scVI model on test data and compute metrics.
    """
    print("Evaluating scVI model...")

    # Extract scVI outputs for training data
    Z_scvi_train = model.get_latent_representation()
    adata_train.obsm["X_scvi"] = Z_scvi_train

    # Clustering on training data
    sc.pp.neighbors(adata_train, use_rep="X_scvi", n_neighbors=30, metric="euclidean")
    sc.tl.leiden(adata_train, key_added="leiden_scvi", resolution=1.0)

    # Training metrics
    y_true_train = adata_train.obs["celltype"].astype(str).values
    y_pred_train = adata_train.obs["leiden_scvi"].astype(str).values
    ari_train = adjusted_rand_score(y_true_train, y_pred_train)
    nmi_train = normalized_mutual_info_score(y_true_train, y_pred_train)

    # Apply scVI model to test data
    Z_scvi_test = model.get_latent_representation(adata_test)
    Xden_scvi_test = model.get_normalized_expression(adata_test, library_size=1e4)

    # Add test latent representation to test data
    adata_test.obsm["X_scvi"] = Z_scvi_test

    # Perform clustering on test data using the same parameters
    sc.pp.neighbors(adata_test, use_rep="X_scvi", n_neighbors=30, metric="euclidean")
    sc.tl.leiden(adata_test, key_added="leiden_scvi", resolution=1.0)

    # Generate UMAP for test data
    sc.tl.umap(adata_test, random_state=42)

    # Calculate metrics for test data
    y_true_test = adata_test.obs["celltype"].astype(str).values
    y_pred_test = adata_test.obs["leiden_scvi"].astype(str).values
    ari_test = adjusted_rand_score(y_true_test, y_pred_test)
    nmi_test = normalized_mutual_info_score(y_true_test, y_pred_test)

    # Display results
    print("=== scVI Results ===")
    print(f"Train ARI: {ari_train:.4f} | Test ARI: {ari_test:.4f}")
    print(f"Train NMI: {nmi_train:.4f} | Test NMI: {nmi_test:.4f}")
    print(f"Latent dimensions: {Z_scvi_test.shape[1]}")
    print(f"Test data shape: {adata_test.shape}")

    return {
        "Z_scvi_test": Z_scvi_test,
        "Xden_scvi_test": Xden_scvi_test,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
        "ari_train": ari_train,
        "nmi_train": nmi_train,
        "ari_test": ari_test,
        "nmi_test": nmi_test,
        "adata_test": adata_test,
    }


def save_results(results, output_dir, dataset_name):
    """
    Save scVI results in organized folder structure.
    """
    print(f"Saving results to {output_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract results
    Z_scvi_test = results["Z_scvi_test"]
    y_true_test = results["y_true_test"]
    y_pred_test = results["y_pred_test"]
    ari_test = results["ari_test"]
    nmi_test = results["nmi_test"]
    adata_test = results["adata_test"]

    # Create label encoder similar to main.py
    print("Creating label encoder...")
    all_labels = np.unique(y_true_test)
    label_encoder = {label: idx for idx, label in enumerate(sorted(all_labels))}

    # Encode labels as integers
    y_true_test_encoded = np.array([label_encoder[label] for label in y_true_test])

    # Save test data latent embeddings (15D to match R script expectations)
    # If we have more than 15 dimensions, we'll take the first 15
    latent_dim = min(Z_scvi_test.shape[1], 15)
    embeddings_15d = Z_scvi_test[:, :latent_dim]

    # Create embeddings DataFrame (R script expects this format)
    embeddings_df = pd.DataFrame(embeddings_15d)
    embeddings_csv_path = os.path.join(output_dir, "denoised_embeddings.csv")
    embeddings_df.to_csv(embeddings_csv_path, index=True)

    # Create labels DataFrame (R script expects column named 'x')
    labels_df = pd.DataFrame({"x": y_true_test})
    labels_csv_path = os.path.join(output_dir, "cleaned_cell_labels_meta_tea_seq.csv")
    labels_df.to_csv(labels_csv_path, index=False)

    # Save as PyTorch tensor format (similar to mixed diffusion pipeline)
    # Convert denoised expression to numpy array if it's a DataFrame
    Xden_scvi_test = results["Xden_scvi_test"]
    if hasattr(Xden_scvi_test, "values"):  # Check if it's a DataFrame
        Xden_scvi_test = Xden_scvi_test.values

    results_dict = {
        "x_denoised": torch.tensor(embeddings_15d, dtype=torch.float32),
        "x_denoised_labels": torch.tensor(y_true_test_encoded, dtype=torch.long),
        "x_true": torch.tensor(embeddings_15d, dtype=torch.float32),
        "test_data_latent": torch.tensor(Z_scvi_test, dtype=torch.float32),
        "test_data_denoised": torch.tensor(Xden_scvi_test, dtype=torch.float32),
        "true_labels_str": y_true_test,
        "pred_labels_str": y_pred_test,
        "data_config": {
            "label_encoder": label_encoder,
        },
        "metrics": {
            "ari_train": results["ari_train"],
            "nmi_train": results["nmi_train"],
            "ari_test": ari_test,
            "nmi_test": nmi_test,
        },
    }

    pytorch_path = os.path.join(output_dir, "denoising_results.pt")
    torch.save(results_dict, pytorch_path)

    # Save metrics summary as JSON
    metrics_summary = {
        "dataset": dataset_name,
        "method": "scVI",
        "metrics": {
            "ari_train": float(results["ari_train"]),
            "nmi_train": float(results["nmi_train"]),
            "ari_test": float(ari_test),
            "nmi_test": float(nmi_test),
        },
        "data_info": {
            "n_cells_test": len(y_true_test),
            "n_features": Z_scvi_test.shape[1],
            "n_cell_types": len(np.unique(y_true_test)),
            "n_clusters_detected": len(np.unique(y_pred_test)),
        },
        "label_encoder": label_encoder,
    }

    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Create and save UMAP visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: UMAP colored by true cell types
    sc.pl.umap(
        adata_test,
        color="celltype",
        title="Test Data UMAP: True Cell Types",
        ax=axes[0],
        show=False,
        frameon=False,
    )

    # Plot 2: UMAP colored by scVI clustering results
    sc.pl.umap(
        adata_test,
        color="leiden_scvi",
        title=f"Test Data UMAP: scVI Clustering (ARI={ari_test:.3f})",
        ax=axes[1],
        show=False,
        frameon=False,
    )

    plt.tight_layout()
    umap_path = os.path.join(output_dir, "scvi_umap_comparison.png")
    plt.savefig(umap_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"\n=== Results saved to {output_dir} ===")
    print(
        f"  📄 denoised_embeddings.csv: {embeddings_15d.shape[0]} cells × {embeddings_15d.shape[1]} latent dimensions"
    )
    print(f"  📄 cleaned_cell_labels_meta_tea_seq.csv: {len(y_true_test)} cell labels")
    print(
        f"  📄 denoising_results.pt: PyTorch format with full results + label encoder"
    )
    print(f"  📄 metrics_summary.json: Performance metrics and metadata")
    print(f"  📄 scvi_umap_comparison.png: UMAP visualization")

    print(f"\nLabel encoder mapping:")
    for original_label, encoded_idx in label_encoder.items():
        count = np.sum(y_true_test == original_label)
        print(f"  {original_label} -> {encoded_idx} ({count} cells)")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run scVI benchmark on single-cell data"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to data configuration JSON file"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name for organizing results"
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=25,
        help="Number of latent dimensions for scVI (default: 25)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=400,
        help="Maximum training epochs (default: 400)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results",
        help="Root directory for saving results (default: results)",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = os.path.join(args.output_root, args.dataset, "scVI")

    print(f"=== scVI Benchmark ===")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"Latent dimensions: {args.n_latent}")
    print(f"Max epochs: {args.max_epochs}")

    try:
        # Load and transform data
        print("\n1. Loading and transforming data...")
        data_transformer = DataTransformer(args.config)
        X_train, y_train, X_test, y_test = data_transformer.transform_all_splits()

        # Setup AnnData objects
        print("\n2. Setting up AnnData objects...")
        adata_train, adata_test = setup_anndata_for_scvi(
            X_train, y_train, X_test, y_test
        )

        # Prepare count data for scVI
        print("\n3. Preparing count data...")
        adata_train, adata_test = prepare_count_data(
            adata_train, adata_test, X_train, X_test
        )

        # Train scVI model
        print("\n4. Training scVI model...")
        model = train_scvi_model(
            adata_train, n_latent=args.n_latent, max_epochs=args.max_epochs
        )

        # Evaluate model
        print("\n5. Evaluating model...")
        results = evaluate_scvi_model(model, adata_train, adata_test)

        # Save results
        print("\n6. Saving results...")
        output_path = save_results(results, output_dir, args.dataset)

        print(f"\n✅ scVI benchmark completed successfully!")
        print(f"Results saved to: {output_path}")
        print(f"\n▶️  Run R script: Rscript scripts/clustering_metrics_with_.R")

    except Exception as e:
        print(f"\n❌ Error running scVI benchmark: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
