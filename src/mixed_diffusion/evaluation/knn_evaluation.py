"""
KNN-based evaluation module for comparing predicted and true labels.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from mixed_diffusion.helpers import ensure_numpy
from .pca_preprocessing import PCAPreprocessor


def evaluate_with_knn(
    x_denoised,
    train_data,
    true_labels,
    knn_func,
    k=1,
    pca_components=None,
    standardize_pca=True,
    verbose=True,
):
    """
    Evaluate denoised samples using KNN classification with optional PCA preprocessing.

    Args:
        x_denoised: Denoised samples to evaluate
        train_data: Training dataset (torch.utils.data.TensorDataset)
        true_labels: True labels for the test samples
        knn_func: KNN function to use for classification
        k: Number of nearest neighbors to consider
        pca_components: Number of PCA components to use for dimensionality reduction.
                       If None, no PCA is applied.
        standardize_pca: Whether to standardize data before PCA
        verbose: Whether to print detailed results

    Returns:
        dict: Dictionary containing evaluation results including:
            - predicted_labels: Predicted labels from KNN
            - accuracy: Classification accuracy
            - correct_predictions: Number of correct predictions
            - total_predictions: Total number of predictions
            - confusion_matrix: Confusion matrix (if applicable)
            - classification_report: Detailed classification metrics
            - average_distance: Average distance to nearest neighbors
            - pca_info: PCA preprocessing information (if PCA was used)
    """
    results = {}

    # Get training data and labels
    train_features = train_data.tensors[0]
    train_labels = train_data.tensors[1]
    labels = true_labels.cpu().numpy()

    # Apply PCA preprocessing if requested
    if pca_components is not None:
        if verbose:
            print(f"Applying PCA preprocessing (n_components={pca_components})")

        pca_preprocessor = PCAPreprocessor(
            n_components=pca_components, standardize=standardize_pca
        )

        # Fit PCA on training data
        train_features_pca = pca_preprocessor.fit_transform(
            train_features, return_format="tensor"
        )

        # Transform test data
        x_denoised_pca = pca_preprocessor.transform(x_denoised, return_format="tensor")

        if verbose:
            pca_preprocessor.print_variance_summary()

        # Store PCA info
        results["pca_info"] = pca_preprocessor.get_variance_info()

        # Use PCA-transformed data for KNN
        knn_train_data = (train_features_pca, train_labels)
        knn_test_data = x_denoised_pca

    else:
        # Use original data
        knn_train_data = (train_features, train_labels)
        knn_test_data = x_denoised
        results["pca_info"] = None

    # Perform KNN
    indices, distances = knn_func(
        knn_test_data,
        knn_train_data,
        k=k,
    )

    # Get labels for k nearest neighbors for each query point
    train_labels_np = train_labels.cpu().numpy()
    knn_labels = train_labels_np[indices]  # Shape: [num_queries, k]

    # Perform majority vote for each query point
    if k == 1:
        predicted_labels = knn_labels.flatten()
    else:
        predicted_labels = stats.mode(knn_labels, axis=1, keepdims=False)[0]

    # Convert true labels to numpy for comparison
    true_labels_np = ensure_numpy(true_labels)

    # Calculate accuracy
    correct_predictions = (predicted_labels == true_labels_np).sum()
    total_predictions = len(predicted_labels)
    accuracy = correct_predictions / total_predictions

    # Store basic results
    results.update(
        {
            "predicted_labels": predicted_labels,
            "accuracy": accuracy,
            "correct_predictions": int(correct_predictions),
            "total_predictions": int(total_predictions),
            "average_distance": distances.mean(),
        }
    )

    if verbose:
        pca_suffix = ""
        if results["pca_info"] is not None:
            pca_info = results["pca_info"]
            pca_suffix = f" with PCA ({pca_info['original_dimensions']}D → {pca_info['reduced_dimensions']}D)"

        print(f"\n=== KNN Label Comparison (k={k}){pca_suffix} ===")
        print(f"Predicted labels shape: {predicted_labels.shape}")
        print(f"True labels shape: {true_labels_np.shape}")
        print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")

    # Generate confusion matrix and classification report for manageable number of classes
    unique_true = sorted(set(true_labels_np))
    unique_pred = sorted(set(predicted_labels))
    all_labels = sorted(set(unique_true + unique_pred))

    if (
        len(all_labels) <= 20
    ):  # Only show detailed analysis for manageable number of classes
        cm = confusion_matrix(true_labels_np, predicted_labels, labels=all_labels)
        results["confusion_matrix"] = cm

        if verbose:
            print(f"\nConfusion Matrix:")
            print(f"True\\Pred", end="")
            for label in all_labels:
                print(f"{label:>6}", end="")
            print()

            for i, true_label in enumerate(all_labels):
                print(f"{true_label:>7}", end="")
                for j in range(len(all_labels)):
                    print(f"{cm[i,j]:>6}", end="")
                print()

        # Classification report
        class_report = classification_report(
            true_labels_np, predicted_labels, labels=all_labels, zero_division=0
        )
        results["classification_report"] = class_report

        if verbose:
            print(f"\nClassification Report:")
            print(class_report)
    else:
        if verbose:
            print(
                f"\nSkipping detailed analysis - too many classes ({len(all_labels)})"
            )

    if verbose:
        print(f"Average distance to nearest neighbors: {distances.mean():.4f}")

    return results


def print_knn_summary(results):
    """
    Print a summary of KNN evaluation results.

    Args:
        results: Dictionary returned from evaluate_with_knn
    """
    print(f"\n{'='*50}")
    print(f"KNN EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(
        f"Correct Predictions: {results['correct_predictions']}/{results['total_predictions']}"
    )
    print(f"Average NN Distance: {results['average_distance']:.4f}")

    if results.get("pca_info") is not None:
        pca_info = results["pca_info"]
        print(
            f"PCA Applied: {pca_info['original_dimensions']}D → {pca_info['reduced_dimensions']}D"
        )
        print(
            f"Variance Explained: {pca_info['total_variance_explained']:.3f} ({pca_info['total_variance_explained']*100:.1f}%)"
        )
    else:
        print(f"PCA Applied: No")

    if "confusion_matrix" in results:
        print(f"Confusion Matrix Available: Yes")
    else:
        print(f"Confusion Matrix Available: No (too many classes)")
    print(f"{'='*50}")


def compare_knn_results(results_list, labels=None):
    """
    Compare multiple KNN evaluation results.

    Args:
        results_list: List of result dictionaries from evaluate_with_knn
        labels: Optional list of labels for each result set
    """
    if labels is None:
        labels = [f"Result {i+1}" for i in range(len(results_list))]

    print(f"\n{'='*60}")
    print(f"KNN COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Accuracy':<10} {'Correct/Total':<15} {'Avg Distance':<12}")
    print(f"{'-'*60}")

    for label, results in zip(labels, results_list):
        accuracy = results["accuracy"]
        correct = results["correct_predictions"]
        total = results["total_predictions"]
        avg_dist = results["average_distance"]

        print(f"{label:<15} {accuracy:<10.4f} {correct}/{total:<10} {avg_dist:<12.4f}")

    print(f"{'='*60}")
