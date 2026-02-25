import itertools
import subprocess
import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[], help="List of models to use")
    parser.add_argument("--data_path", type=str, help="Path to the dataset JSON file")
    args = parser.parse_args()

    models = args.models
    data_path = args.data_path
    filename = data_path.split("/")[-1].split(".")[0] if data_path else None

    repetitions = [10]
    gibbs_iterations = [200]
    rho_pairs = [
        (20.0, 20.0),
        (10.0, 10.0),
        (10.0, 5.0),
        (5.0, 5.0),
    ]

    # Generate all combinations and convert to list for progress tracking
    combinations = list(
        itertools.product(models, repetitions, gibbs_iterations, rho_pairs)
    )

    print(f"Running {len(combinations)} total combinations...")

    # Single loop with progress bar
    for model, rep, gibbs_iter, (rho_start, rho_end) in tqdm.tqdm(
        combinations, desc="Grid Search Progress"
    ):
        print(
            f"Running: model={model}, repetitions={rep}, "
            f"gibbs_iterations={gibbs_iter}, rho_start={rho_start}, rho_end={rho_end}"
        )

        subprocess.run(
            [
                "python",
                "scripts/main.py",
                "--model_dir",
                f"models/{model}",
                "--visualize",
                "--test_noise_level=0",
                "--gibbs_iterations",
                str(gibbs_iter),
                "--rho_scheduling_type",
                "linear",
                "--rho_start",
                str(rho_start),
                "--rho_end",
                str(rho_end),
                "--initial_x",
                "measurement",
                "--result_dir",
                f"results/{model}/{filename}/gibbs{gibbs_iter}_rho{rho_start}-{rho_end}_rep{rep}",
                "--test_on",
                "test",
                "--repeated_sampling_factor",
                str(rep),
                "--repeated_sampling_method",
                "mean",
                "--save_data",
                "--visualization_method",
                "python_umap",
                "--overwrite_data_load_path",
                data_path,
            ]
        )


if __name__ == "__main__":
    main()
