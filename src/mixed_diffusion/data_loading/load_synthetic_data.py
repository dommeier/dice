import numpy as np
from sklearn.model_selection import train_test_split
import torch
from mixed_diffusion.data_loading.factor_model_data_generation import generate_factor_model_data
from mixed_diffusion.data_loading.multidimensional_data_generation import (
    generate_multidimensional_data,
)
from mixed_diffusion.data_loading.outlines_data_generation import generate_outlines_data

from torch.utils.data import TensorDataset
from mixed_diffusion.data_loading.mr_mash_data_generation import simulate_mr_mash_data


import os


def load_config_from_file(config_file):
    import json

    if not config_file:
        raise ValueError("Config file path must be provided.")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' does not exist.")

    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def load_synthetic_data(args):
    # Data will be noised later in the process, so get clean data here

    info = {}
    if args.dataset in ["circle", "letters", "ngons"]:
        data, labels = generate_outlines_data(
            num_samples=args.num_samples,
            data_type=args.dataset,
            noise_level=0,
            sample_size=args.sample_size,
        )
    elif args.dataset == "multidimensional":
        config = load_config_from_file(args.config_file)
        data, labels, info = generate_multidimensional_data(**config)
    elif args.dataset == "factor_model":
        config = load_config_from_file(args.config_file)
        data, labels, info = generate_factor_model_data(**config)
    elif args.dataset == "mr_mash":
        config = load_config_from_file(args.config_file)
        output = simulate_mr_mash_data(**config)
        data = torch.from_numpy(np.stack(output["X"], output["Y"])).to(torch.float32)
        labels = torch.zeros(data.shape)

    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    # Unpack the tuple using the star (*) operator
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.1
    )

    train = TensorDataset(train_data, train_labels)
    test = TensorDataset(test_data, test_labels)

    return train, test, info
