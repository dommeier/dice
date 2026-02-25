import torch
from mixed_diffusion.helpers import load_archetypes
from mixed_diffusion.sampling import log_likelihood_y_given_x, map_y_back_to_x
from mixed_diffusion.identification.knn import knn
from mixed_diffusion.evaluation import evaluate_with_knn
from mixed_diffusion.wasserstein_distance import wasserstein_distance_from_samples
from mixed_diffusion.main_utils import mse


def calculate_likelihood(args, y, x0_repeated, observation_transform):
    """Calculate likelihood of observations given archetypes"""
    if not args.likelihood:
        return

    Sigma = torch.eye(y.shape[1], device=y.device) * args.test_noise_level**2
    archetypes = load_archetypes(x0_repeated)

    # Stack log-likelihoods for each archetype: [B, K]
    log_likelihoods = []

    for archetype in archetypes:
        # Expand single archetype to match batch size
        expanded_archetype = archetype.unsqueeze(0).expand(y.shape[0], -1)  # [B, dx]

        logp = log_likelihood_y_given_x(
            expanded_archetype,
            y,  # [B, dy]
            observation_transform,
            Sigma,
        )  # shape: [B]

        log_likelihoods.append(logp)

    # Stack into shape [K, B] → transpose to [B, K]
    log_likelihoods = torch.stack(log_likelihoods).T  # [B, K]

    # Normalize in log-space
    log_posteriors = torch.log_softmax(log_likelihoods, dim=1)  # [B, K]
    posteriors = torch.exp(log_posteriors)  # [B, K]

    # Print average posterior per archetype across the batch
    print("Average posterior probability per archetype (calculated):")
    for idx in range(archetypes.shape[0]):
        print(f"  Archetype {idx}: {posteriors[:, idx].mean().item():.4f}")


def calculate_basic_metrics(r):
    """Calculate basic MSE metrics"""
    results = {}
    results["noisy_error"] = mse(
        r["y_noised_repeated"],
        r["y_repeated"],
    )
    results["error"] = mse(
        r["x_denoised"],
        r["x_true"],
    )
    print(f"MSE between original and noisy data: {results['noisy_error']:.6f}")
    print(f"MSE between repeated original and denoised data: {results['error']:.6f}")

    return results


def calculate_wasserstein_distance(args, x0, x0_repeated, x_denoised, y, results):
    """Calculate Wasserstein distances if requested"""
    if not args.wasserstein_distance:
        return results

    results["wasserstein_distance_true_noised"] = wasserstein_distance_from_samples(
        x0_repeated, y
    )
    print(
        f"Wasserstein distance between original and noisy data: {results['wasserstein_distance_true_noised']:.6f}"
    )

    results["wasserstein_distance_true_denoised"] = wasserstein_distance_from_samples(
        x0, x_denoised
    )
    print(
        f"Wasserstein distance between original and denoised data: {results['wasserstein_distance_true_denoised']:.6f}"
    )

    improvement = (
        results["wasserstein_distance_true_noised"]
        - results["wasserstein_distance_true_denoised"]
    )
    improvement_pct = 100 * improvement / results["wasserstein_distance_true_noised"]
    print(
        f"Denoising improved Wasserstein distance by: {improvement:.6f} ({improvement_pct:.2f}%)"
    )

    return results


def run_knn_evaluation(
    args, x_denoised, train_data, true_labels, y, observation_transform, results
):
    """Run KNN evaluation and comparison"""
    if not args.knn:
        return results, None

    # Determine if PCA should be applied
    pca_components = None
    if args.pca_components is not None and args.pca_components > 0:
        pca_components = args.pca_components
        print(f"Applying PCA preprocessing with {pca_components} components")

    # First, evaluate KNN on noisy measurements (before denoising)
    print("\n" + "=" * 60)
    print("EVALUATING KNN ON NOISY MEASUREMENTS (BEFORE DENOISING)")
    print("=" * 60)

    # Map noisy observations back to original space for fair comparison
    y_mapped_back = map_y_back_to_x(y, observation_transform)

    knn_noisy_results = evaluate_with_knn(
        x_denoised=y_mapped_back,
        train_data=train_data,
        true_labels=true_labels,
        knn_func=knn,
        k=args.knn_k if hasattr(args, "knn_k") else 5,
        pca_components=pca_components,
        standardize_pca=True,
        verbose=True,
    )

    # Now evaluate KNN on denoised samples
    print("\n" + "=" * 60)
    print("EVALUATING KNN ON DENOISED SAMPLES")
    print("=" * 60)

    knn_results = evaluate_with_knn(
        x_denoised=x_denoised,
        train_data=train_data,
        true_labels=true_labels,
        knn_func=knn,
        k=args.knn_k if hasattr(args, "knn_k") else 5,
        pca_components=pca_components,
        standardize_pca=True,
        verbose=True,
    )

    # Extract predicted labels for visualization (from denoised)
    predicted_labels = knn_results["predicted_labels"]

    # Store results in main results dictionary
    results.update(
        {
            # Noisy measurements results
            "knn_noisy_accuracy": knn_noisy_results["accuracy"],
            "knn_noisy_correct_predictions": knn_noisy_results["correct_predictions"],
            "knn_noisy_total_predictions": knn_noisy_results["total_predictions"],
            "knn_noisy_average_distance": knn_noisy_results["average_distance"],
            # Denoised results
            "knn_denoised_accuracy": knn_results["accuracy"],
            "knn_denoised_correct_predictions": knn_results["correct_predictions"],
            "knn_denoised_total_predictions": knn_results["total_predictions"],
            "knn_denoised_average_distance": knn_results["average_distance"],
            # Legacy names for backward compatibility
            "knn_accuracy": knn_results["accuracy"],
            "knn_correct_predictions": knn_results["correct_predictions"],
            "knn_total_predictions": knn_results["total_predictions"],
            "knn_average_distance": knn_results["average_distance"],
            # Comparison metrics
            "knn_accuracy_improvement": knn_results["accuracy"]
            - knn_noisy_results["accuracy"],
            "knn_pca_applied": pca_components is not None,
        }
    )

    # Store PCA info if available (same for both since using same preprocessing)
    if knn_results.get("pca_info") is not None:
        results["pca_info"] = knn_results["pca_info"]

    # Print comparison summary
    print("\n" + "=" * 60)
    print("KNN PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    accuracy_improvement = results["knn_accuracy_improvement"]
    print(f"Noisy Measurements Accuracy:  {knn_noisy_results['accuracy']:.4f}")
    print(f"Denoised Samples Accuracy:    {knn_results['accuracy']:.4f}")
    print(
        f"Accuracy Improvement:         {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.1f}%)"
    )

    distance_improvement = (
        knn_noisy_results["average_distance"] - knn_results["average_distance"]
    )
    print(f"Noisy Measurements Avg Dist:  {knn_noisy_results['average_distance']:.4f}")
    print(f"Denoised Samples Avg Dist:    {knn_results['average_distance']:.4f}")
    print(f"Distance Improvement:         {distance_improvement:+.4f}")

    if accuracy_improvement > 0:
        print(f"✅ Denoising IMPROVED classification accuracy!")
    elif accuracy_improvement < 0:
        print(f"⚠️  Denoising DECREASED classification accuracy.")
    else:
        print(f"➡️  Denoising had NO EFFECT on classification accuracy.")

    print("=" * 60)

    return results, predicted_labels


def run_cluster_mapping(args, x_denoised):
    """Run cluster mapping if requested"""
    if args.map_to_clusters:
        map_to_clusters(x_denoised, load_archetypes(x_denoised))
