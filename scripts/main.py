import torch
import json
from mixed_diffusion.data_loading.data_loading import get_data
from mixed_diffusion.sampling import gibbs_sampling, map_y_back_to_x
from mixed_diffusion.train import load_noise_transform
from mixed_diffusion.utils import save_data
from mixed_diffusion.visualize import visualize_denoising

# Import new modules
from mixed_diffusion.main_utils import get_device, ensure_directories_exist
from mixed_diffusion.config_manager import load_model_config
from mixed_diffusion.data_preparation import (
    create_data_transform,
    select_test_data,
    prepare_observations,
    process_repeated_sampling,
)
from mixed_diffusion.model_handler import setup_and_train_model, load_existing_model
from mixed_diffusion.evaluation_handler import (
    calculate_basic_metrics,
)
from mixed_diffusion.argument_parser import create_argument_parser


def main(args):
    print("Running the process with the following arguments:", args)

    # Setup directories and configuration
    ensure_directories_exist(args)

    # Save arguments to file in results directory

    args_dict = vars(args)
    with open(f"{args.result_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4, default=str)
    print(f"Arguments saved to {args.result_dir}/args.json")

    config = load_model_config(args.model_dir)
    device = get_device(args)

    if args.overwrite_data_load_path is not None:
        config["data_path"] = args.overwrite_data_load_path
        print(f"Overwrote data_path in config to be: {args.overwrite_data_load_path}")

    # Prepare data
    transform = create_data_transform()
    train_data, test_data, data_config = get_data(config, transform)

    # Prepare observations
    y_test, y_test_labels, y_train, y_train_labels = select_test_data(
        args, train_data, test_data
    )
    y_noised_repeated, y_noised_repeated_labels, y_repeated = prepare_observations(
        y_test, y_test_labels, args, config, device
    )

    observation_transform = data_config.get("observation_transform_matrix", None)
    print(f"Observation transform shape: {observation_transform.shape}")

    observation_transform = torch.tensor(observation_transform, dtype=torch.float32).to(
        device
    )

    x_observed_train = y_train.to(device) @ observation_transform.T

    # Setup or load model
    if args.from_scratch:
        model, config, training_results = setup_and_train_model(
            config, x_observed_train, y_train_labels, device, args, train_data
        )
    else:
        model = load_existing_model(config, x_observed_train.shape, device, args)

    # Perform denoising
    if args.gibbs_iterations > 0:
        x_denoised = gibbs_sampling(
            args, y_noised_repeated, model, config, observation_transform.T, data_config
        )
    else:
        x_denoised = y_noised_repeated @ observation_transform.T

    # Process repeated sampling results
    x_denoised = process_repeated_sampling(
        x_denoised, y_test.to(device) @ observation_transform.T, args
    )

    # Handle labels based on repeated sampling method
    if args.repeated_sampling_method == "mean":
        # For mean method, denoised data corresponds to original x0 labels
        observation_labels = y_test_labels  # Original labels for x0
        x_denoised_labels = y_test_labels  # Same labels for averaged denoised data
    else:
        # For no repeated sampling, x0 uses original labels, x_denoised uses repeated labels
        observation_labels = y_test_labels  # Original labels for x0 (size matches x0)
        x_denoised_labels = y_noised_repeated_labels  # Repeated labels for x_denoised (size matches x_denoised)

    # Prepare results data
    saved_results = {
        "y_train": y_train,
        "train_labels": y_train_labels,
        "x_train": x_observed_train,
        "y_repeated": y_repeated,
        "y_noised_repeated": y_noised_repeated,
        "y_noised_repeated_labels": y_noised_repeated_labels,
        "x_denoised": x_denoised,
        "x_denoised_labels": x_denoised_labels,
        "x_noised": y_noised_repeated @ observation_transform.T,
        "x_true": y_test.to(device) @ observation_transform.T.to(device),
        "y_test_labels": y_test_labels,
        "observations": y_noised_repeated @ observation_transform.T,
        "observations_labels": observation_labels,
        "observation_transform": observation_transform,
        "data_config": data_config,
    }

    if args.save_data:
        torch.save(saved_results, f"{args.result_dir}/denoising_results.pt")
        print(f"Data saved to {args.result_dir}/denoising_results.pt")

        # Save CSV files for R analysis
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA

        # Perform PCA on denoised data (15 dimensions as expected by R script)
        x_denoised_np = x_denoised.cpu().numpy()

        # Save PCA embeddings for R
        denoised_df = pd.DataFrame(x_denoised_np)
        denoised_df.to_csv(f"{args.result_dir}/denoised_embeddings.csv", index=True)
        print(
            f"PCA embeddings (15D) saved to {args.result_dir}/denoised_embeddings.csv"
        )

        # Save cell labels metadata for R (using denoised labels)
        x0_denoised_labels_np = x_denoised_labels.cpu().numpy()
        meta_df = pd.DataFrame({"x": x0_denoised_labels_np})
        meta_df.to_csv(
            f"{args.result_dir}/cleaned_cell_labels_meta_tea_seq.csv", index=False
        )
        print(
            f"Cell labels metadata saved to {args.result_dir}/cleaned_cell_labels_meta_tea_seq.csv"
        )

    # Visualization and data saving
    if args.visualize:
        results = visualize_denoising(
            saved_results=saved_results,
            output_path=f"{args.result_dir}/denoising_comparison.png",
            args=args,
        )

        if results:
            with open(f"{args.result_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4)

        # Open browser if requested
        if args.open_browser:
            import os
            import webbrowser

            html_path = f"{args.result_dir}/denoising_comparison.html"
            if os.path.exists(html_path):
                print(f"🌐 Opening {html_path} in browser...")
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
            else:
                print(f"⚠️  HTML file not found: {html_path}")
                print("Note: HTML visualization may not have been generated")


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
