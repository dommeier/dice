import argparse


def create_argument_parser():
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@", description="Train a diffusion model."
    )

    # Data and model selection arguments
    parser.add_argument(
        "--test_on",
        type=str,
        choices=["train", "test", "both", "one_type_each"],
        default="test",
        help="Which dataset to run denoising on: 'train', 'test', or 'both' (default: test)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to use. If not set, use all test samples.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to use for training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UNet",
        choices=["UNet", "TabularDiffusionMLP", "TabularDiffusionTransformer"],
    )

    # Directory arguments
    parser.add_argument(
        "--data_file",
        type=str,
        default="/home/ubuntu/data",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/home/ubuntu/results/mixed_diffusion",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/ubuntu/models/mixed_diffusion",
        help="Directory to load the diffusion model from",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file for the data generating process.",
    )

    # Training arguments
    parser.add_argument(
        "--from_scratch", action="store_true", help="Train the model from scratch."
    )
    parser.add_argument(
        "--checkpoints_to_keep",
        type=int,
        default=0,
        help="If set, save checkpoints every N epochs. 0 means save all checkpoints.",
    )

    # Model configuration arguments
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Configuration option for TabularDiffusionMLP",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=2,
        help="Configuration option for TabularDiffusionMLP",
    )

    # Diffusion and sampling arguments
    parser.add_argument(
        "--noise_step", type=int, default=256, help="Total diffusion steps."
    )
    parser.add_argument("--gibbs_iterations", type=int, default=10)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to generate for Langevin Monte Carlo.",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.01,
        help="Step size for Langevin Monte Carlo.",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        default=1000,
        help="Number of burn-in iterations for Langevin Monte Carlo.",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=-1,
        help="Number of initial steps for Gibbs sampling.",
    )
    parser.add_argument(
        "--initial_x",
        type=str,
        default="zero",
        choices=["zero", "measurement", "random", "true_x"],
        help="How to initialize x for Gibbs sampling. Options: 'zero', 'measurement', 'random', 'true_x'.",
    )

    # Noise and observation arguments
    parser.add_argument(
        "--rho",
        type=float,
        default=0.7,
        help="Noise level for generating noisy observation.",
    )
    parser.add_argument(
        "--rho_start",
        type=float,
        default=0.7,
        help="Initial noise level for Gibbs sampling.",
    )
    parser.add_argument(
        "--rho_end",
        type=float,
        default=0.2,
        help="Final noise level for Gibbs sampling.",
    )
    parser.add_argument("--rho_scheduling_type", type=str, default="exponential")
    parser.add_argument(
        "--test_noise_level",
        type=float,
        default=0.2,
        help="Noise level for generating noisy observation.",
    )
    parser.add_argument(
        "--same_noise",
        action="store_true",
        help="Use the same noise for all rows in the Gibbs sampling. (Useful for debugging)",
    )

    # Repeated sampling arguments
    parser.add_argument(
        "--repeated_sampling_factor",
        type=int,
        default=1,
        help="Factor by which to upsamples the data before processing.",
    )
    parser.add_argument(
        "--repeated_sampling_method",
        type=str,
        default="mean",
        choices=["mean", "none"],
    )

    # Evaluation arguments
    parser.add_argument(
        "--likelihood",
        action="store_true",
        help="Calculate the likelihood of the observations given the denoised samples.",
    )
    parser.add_argument(
        "--wasserstein_distance",
        action="store_true",
        help="Calculate Wasserstein distance.",
    )
    parser.add_argument(
        "--map_to_clusters",
        action="store_true",
        help="Map the denoised samples to clusters using k-means.",
    )
    parser.add_argument(
        "--knn",
        action="store_true",
        help="Run KNN on the denoised samples and print accuracy.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors to use for KNN classification.",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=None,
        help="Number of PCA components to use before KNN. If not set, no PCA is applied.",
    )

    # Output and visualization arguments
    parser.add_argument(
        "--save_data", action="store_true", help="Save the generated data."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the results."
    )
    parser.add_argument(
        "--visualize_with_train_context",
        action="store_true",
        help="When visualizing test data denoising, include training data points for context.",
    )
    parser.add_argument(
        "--open_browser",
        action="store_true",
        help="Automatically open the HTML visualization in the browser (requires --visualize).",
    )
    parser.add_argument(
        "--save_to_one_image",
        action="store_true",
        help="Save the samples to one image.",
    )
    parser.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Store the gibbs sampling iterations",
    )
    parser.add_argument(
        "--conditioning_vector",
        type=str,
        default=None,
        help="Path to the conditioning vector file (if using conditional training).",
    )
    parser.add_argument(
        "--noise_level_for_training",
        type=float,
        default=0.0,
        help="Noise level for training.",
    )
    parser.add_argument(
        "--visualization_method",
        type=str,
        choices=["pca", "umap", "python_umap"],
        default="pca",
        help="Dimensionality reduction method for visualization (default: pca)",
    )
    parser.add_argument(
        "--joint_visualization",
        action="store_true",
        help="Enable joint visualization of true, noisy, and denoised data.",
    )
    parser.add_argument(
        "--underrepresented_threshold",
        type=float,
        default=10.0,
        help="Threshold percentage for marking test set points as underrepresented (default: 10.0). Test points from cell types with less than this percentage in training set will be shown in grey.",
    )
    parser.add_argument(
        "--overwrite_data_load_path",
        type=str,
        default=None,
    )

    # Device arguments
    parser.add_argument("--mps", action="store_true", help="Use MPS for training.")

    return parser
