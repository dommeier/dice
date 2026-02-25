from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import umap.umap_ as umap

import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

import torch


def calculate_clustering_ari(
    x_true, x_denoised, labels, emb_orig, emb_den, resolution=0.19
):
    """
    Calculate Adjusted Rand Index (ARI) for clustering performance comparison.

    Args:
        x_true: Original data array
        x_denoised: Denoised data array
        labels: True labels for the data
        emb_orig: UMAP embedding of original data
        emb_den: UMAP embedding of denoised data
        resolution: Clustering resolution parameter (default: 0.19)

    Returns:
        dict: Dictionary containing ARI scores and improvement
    """
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score
    import pandas as pd
    from anndata import AnnData

    # Convert data to AnnData objects for scanpy clustering
    # Original data
    adata_orig = AnnData(x_true)
    adata_orig.obsm["X_umap"] = emb_orig
    adata_orig.obs["true_labels"] = labels

    # Denoised data
    adata_denoised = AnnData(x_denoised)
    adata_denoised.obsm["X_umap"] = emb_den
    adata_denoised.obs["true_labels"] = labels

    # Step 1 & 2: Find Neighbors and Clusters for original data
    sc.pp.neighbors(adata_orig, n_neighbors=15, use_rep="X_umap")
    sc.tl.louvain(adata_orig, resolution=resolution)
    seuratcluster_orig = adata_orig.obs["louvain"]

    # Step 1 & 2: Find Neighbors and Clusters for denoised data
    sc.pp.neighbors(adata_denoised, n_neighbors=15, use_rep="X_umap")
    sc.tl.louvain(adata_denoised, resolution=resolution)
    seuratcluster_denoised = adata_denoised.obs["louvain"]

    # Step 4: Map true labels to integers
    cell_names = pd.Series(labels)
    cell_names_integer = cell_names.astype("category").cat.codes

    # Step 5: Calculate Adjusted Rand Index for both
    adj_r_index_orig = adjusted_rand_score(cell_names_integer, seuratcluster_orig)
    adj_r_index_denoised = adjusted_rand_score(
        cell_names_integer, seuratcluster_denoised
    )

    return {
        "ari_original": adj_r_index_orig,
        "ari_denoised": adj_r_index_denoised,
    }


def plot_scatter_plotly(
    saved_data,
    args,
    output_path,
):
    """Main plotting function using modular components."""

    # Get label encoder from data_config if available
    label_encoder = None
    if "data_config" in saved_data and saved_data["data_config"] is not None:
        data_config = saved_data["data_config"]
        if "label_encoder" in data_config:
            label_encoder = data_config["label_encoder"]
            print(
                f"Found label encoder with {len(label_encoder)} cell types: {list(label_encoder.keys())}"
            )

    def get_cell_type_name(label_value):
        """Convert numeric label to cell type name using label encoder."""
        if label_encoder is None:
            return str(label_value)

        # Check if label_encoder is properly formatted (cell_type -> numeric)
        # If it's malformed (numeric -> numeric), just return the label
        first_key = next(iter(label_encoder.keys()))
        if isinstance(first_key, (int, float)):
            return str(label_value)

        # label_encoder maps cell_type -> numeric_value, so we need to reverse it
        for cell_type, numeric_value in label_encoder.items():
            if numeric_value == label_value:
                return cell_type
        return str(label_value)

    # Step 3: Prepare datasets for dimensionality reduction
    # Try different possible field names based on data structure
    datasets = [
        {
            "name": "Original observations",
            "data": saved_data.get("x_true"),
            "labels": saved_data.get("observations_labels"),
        },
        {
            "name": "Noisy observations",
            "data": saved_data.get("x_noised"),
            "labels": saved_data.get("y_noised_repeated_labels"),
        },
        {
            "name": "Denoised samples",
            "data": saved_data.get("x_denoised", None),
            "labels": saved_data.get("x_denoised_labels"),
        },
    ]

    if args.test_noise_level <= 0:
        datasets = [d for d in datasets if d["name"] != "Noisy observations"]

    if args.joint_visualization:
        # Joint visualization: combine all datasets
        combined_data = np.vstack(
            [d["data"] for d in datasets if d["data"] is not None]
        )
        combined_labels = (
            np.hstack([d["labels"] for d in datasets if d["labels"] is not None])
            if any(d["labels"] is not None for d in datasets)
            else None
        )

        datasets = [
            {
                "name": "Combined Data",
                "data": combined_data,
                "labels": combined_labels,
                "indices": (
                    np.hstack(
                        [
                            np.full(d["data"].shape[0], d["name"])
                            for d in datasets
                            if d["data"] is not None
                        ]
                    )
                    if any(d["data"] is not None for d in datasets)
                    else None
                ),
            }
        ]

    # Apply pipeline
    for dataset in datasets:

        # Standardize and check for NaN values
        if dataset["data"] is not None:
            data = dataset["data"]

            # Check for NaN or infinite values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(
                    f"Warning: Found NaN or infinite values in {dataset['name']}, cleaning..."
                )
                data = np.nan_to_num(
                    data,
                    nan=0.0,
                    posinf=np.finfo(np.float32).max,
                    neginf=np.finfo(np.float32).min,
                )

            dataset["data"] = StandardScaler().fit_transform(data)

        method = args.visualization_method

        if method == "umap":
            pca = PCA(n_components=min(args.pca_components, dataset["data"].shape[1]))
            dataset["data"] = pca.fit_transform(dataset["data"])

            # UMAP reduction
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
            )
            dataset["data"] = reducer.fit_transform(dataset["data"])
        elif method == "pca":
            pca = PCA(n_components=2)
            dataset["data"] = pca.fit_transform(dataset["data"])
        else:
            raise ValueError(f"Unknown visualization method: {method}")

    if args.joint_visualization:
        # Split dataset again for plotting
        split_datasets = []
        unique_names = np.unique(datasets[0]["indices"])
        for d in unique_names:
            mask = datasets[0]["indices"] == d
            split_datasets.append(
                {
                    "name": d,
                    "data": datasets[0]["data"][mask],
                    "labels": (
                        datasets[0]["labels"][mask]
                        if datasets[0]["labels"] is not None
                        else None
                    ),
                }
            )
    else:
        split_datasets = datasets

    # Visualize using plotly - create subplots for all datasets in one HTML
    valid_datasets = [d for d in split_datasets if d["data"] is not None]
    n_datasets = len(valid_datasets)

    if n_datasets == 0:
        print("No valid datasets to plot")
        return

    # Create subplots
    cols = min(3, n_datasets)
    rows = (n_datasets + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[dataset["name"] for dataset in valid_datasets],
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    colors = qualitative.Plotly

    for i, dataset in enumerate(valid_datasets):
        data = dataset["data"]
        labels = dataset["labels"]
        name = dataset["name"]

        row = (i // cols) + 1
        col = (i % cols) + 1

        if labels is not None:
            unique_labels = np.unique(labels)
            for j, label in enumerate(unique_labels):
                mask = labels == label
                cell_type_name = get_cell_type_name(label)
                fig.add_trace(
                    go.Scatter(
                        x=data[mask, 0],
                        y=data[mask, 1],
                        mode="markers",
                        name=cell_type_name,
                        marker=dict(
                            color=colors[j % len(colors)],
                            size=4,
                            opacity=0.7,
                        ),
                        showlegend=(
                            i == 0
                        ),  # Only show legend for first dataset to avoid clutter
                        legendgroup=cell_type_name,  # Group same cell types across datasets
                    ),
                    row=row,
                    col=col,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data[:, 0],
                    y=data[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=colors[i % len(colors)], size=4, opacity=0.7),
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f"Denoising Visualization ({args.visualization_method.upper()})",
        height=max(600, 400 * rows),
        width=None,  # Let it fill available width
        hovermode="closest",
        autosize=True,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            title="Cell Types" if label_encoder else "Labels",
        ),
    )

    # Update axis labels for all subplots
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(
                title_text=f"{args.visualization_method.upper()} Component 1",
                row=i,
                col=j,
            )
            fig.update_yaxes(
                title_text=f"{args.visualization_method.upper()} Component 2",
                row=i,
                col=j,
            )

    # Save outputs
    html_path = output_path.replace(".png", ".html")

    # Custom HTML with full viewport styling
    html_string = fig.to_html(
        include_plotlyjs="cdn", div_id="plotly-div", config={"responsive": True}
    )

    # Add custom CSS to make the plot fill the viewport
    full_page_html = html_string.replace(
        "<head>",
        """<head>
    <style>
        body { margin: 0; padding: 0; }
        #plotly-div { 
            width: 100vw !important; 
            height: 100vh !important; 
        }
    </style>""",
    )

    with open(html_path, "w") as f:
        f.write(full_page_html)

    if output_path.endswith(".png"):
        try:
            fig.write_image(output_path)
        except Exception as e:
            print(f"Could not save static image: {e}")

    if args.visualize:
        fig.show()


def visualize_umap_python(
    saved_data, args, title="UMAP Visualization", underrepresented_threshold=10.0
):
    """
    Create UMAP visualization comparing original and denoised data.

    Args:
        saved_data: Dictionary containing data arrays and labels
        args: Arguments object with visualization settings
        title: Plot title
        underrepresented_threshold: Threshold percentage for marking test points as underrepresented.
                                  Test points from cell types with less than this percentage
                                  in the training set will be shown in grey.
    """

    # --- 6) Fair UMAP: fit on ORIGINAL TEST, transform DENOISED into same UMAP ---
    um = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0, metric="cosine").fit(
        saved_data["x_true"]
    )
    um2 = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0, metric="cosine").fit(
        saved_data["x_denoised"]
    )
    emb_orig = um.embedding_
    emb_den = um2.embedding_

    # --- 7) Plot side-by-side (same axes space) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].set_title("Original observations")
    axes[1].set_title("Denoised samples")
    import matplotlib.colors as mcolors

    # Handle integer labels properly
    labels = saved_data["x_denoised_labels"]
    if hasattr(labels, "cpu"):  # Handle torch tensors
        labels = labels.cpu().numpy()

    # Get label encoder from data_config if available
    label_encoder = None
    if "data_config" in saved_data and saved_data["data_config"] is not None:
        data_config = saved_data["data_config"]
        if "label_encoder" in data_config:
            label_encoder = data_config["label_encoder"]

    def get_cell_type_name(label_value):
        """Convert numeric label to cell type name using label encoder."""
        if label_encoder is None:
            return str(label_value)

        # Check if label_encoder is properly formatted (cell_type -> numeric)
        first_key = next(iter(label_encoder.keys()))
        if isinstance(first_key, (int, float)):
            return str(label_value)

        # label_encoder maps cell_type -> numeric_value, so we need to reverse it
        for cell_type, numeric_value in label_encoder.items():
            if numeric_value == label_value:
                return cell_type
        return str(label_value)

    # Check if training data is available to calculate representation percentages
    underrepresented_labels = set()

    # Look for training labels in saved_data
    train_labels_key = None
    for key in ["train_labels", "y_train_labels", "true_labels_train"]:
        if key in saved_data:
            train_labels_key = key
            break

    if train_labels_key is not None:
        train_labels = saved_data[train_labels_key]
        if hasattr(train_labels, "cpu"):  # Handle torch tensors
            train_labels = train_labels.cpu().numpy()

        # Calculate label percentages in training set
        unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
        total_train_samples = len(train_labels)

        print(f"Training set label distribution:")
        for label, count in zip(unique_train_labels, train_counts):
            percentage = (count / total_train_samples) * 100
            cell_type_name = get_cell_type_name(label)
            print(
                f"  {cell_type_name} (label {label}): {count}/{total_train_samples} ({percentage:.1f}%)"
            )

            # Mark labels with < threshold% representation as underrepresented
            if percentage < underrepresented_threshold:
                underrepresented_labels.add(label)
                print(
                    f"    -> Marking as underrepresented (< {underrepresented_threshold}%)"
                )

    else:
        print(
            "Warning: Training labels not found in saved_data. All test points will use normal colors."
        )
        print("Available keys:", list(saved_data.keys()))

    # Get unique labels and create color mapping
    unique_labels = np.unique(labels)
    palette = list(mcolors.TABLEAU_COLORS.values())
    lut = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    # Plot each label separately to create legend entries
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            cell_type_name = get_cell_type_name(label)

            # Use grey for underrepresented labels, normal color otherwise
            if label in underrepresented_labels:
                color = "lightgrey"
                cell_type_name += f" (< {underrepresented_threshold}% in training)"
                alpha = 0.6  # Make underrepresented points slightly more transparent
            else:
                color = lut[label]
                alpha = 0.8

            # Plot on both axes with same color and label
            axes[0].scatter(
                emb_orig[mask, 0],
                emb_orig[mask, 1],
                s=6,
                alpha=alpha,
                c=color,
                label=cell_type_name,
            )
            axes[1].scatter(
                emb_den[mask, 0], emb_den[mask, 1], s=6, alpha=alpha, c=color
            )

    # Add legend only to the first subplot to avoid duplication
    axes[0].legend(bbox_to_anchor=(0, 1), loc="best", fontsize="small")

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.2)

    plt.suptitle(title)
    plt.tight_layout()

    # Calculate ARI for both original and denoised data
    # ari_results = calculate_clustering_ari(
    #     saved_data["x_true"], saved_data["x_denoised"], labels, emb_orig, emb_den
    # )

    # Print the results
    # print(f"\n=== Clustering Performance (ARI) ===")
    # print(f"Original data ARI: {ari_results['ari_original']:.4f}")
    # print(f"Denoised data ARI: {ari_results['ari_denoised']:.4f}")

    # plt.savefig(f"{args.result_dir}/denoising_comparison_umap.png", dpi=300)

    plt.show()

    return


def visualize_denoising(
    saved_results,
    output_path,
    args=None,
):

    # Handle both file path and dict inputs
    if isinstance(saved_results, str):
        # Load saved data from file
        saved_data = torch.load(saved_results, map_location="cpu", weights_only=False)
    else:
        # Use provided dict directly
        saved_data = saved_results

    numpy_data = {}

    # Ensure numpy arrays:
    for key, value in saved_data.items():
        if isinstance(value, torch.Tensor):
            numpy_data[key] = value.cpu().numpy()
        else:
            numpy_data[key] = value

    if args.visualization_method == "python_umap":
        # Use matplotlib-based UMAP visualization
        threshold = getattr(args, "underrepresented_threshold", 0.5)
        return visualize_umap_python(
            numpy_data, args, underrepresented_threshold=threshold
        )

    # Call the plotting logic
    plot_scatter_plotly(
        numpy_data,
        args=args,
        output_path=output_path,
    )
    print(f"Visualization saved to: {output_path}")
