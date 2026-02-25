import json
import os


def load_model_config(model_dir):
    """Load model configuration from JSON file"""
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)
    config["model_dir"] = model_dir
    return config


def save_model_config(config):
    """Save model configuration to JSON file"""
    with open(f"{config['model_dir']}/config.json", "w") as f:
        json.dump(config, f)


def save_training_results(config, results):
    """Save training results to JSON file"""
    with open(f"{config['model_dir']}/results.json", "w") as f:
        json.dump(results, f)


def save_training_data_info(config, train_data, args):
    """Save training data information to JSON file"""
    with open(f"{config['model_dir']}/training_data.json", "w") as f:
        json.dump(
            {
                "num_samples": len(train_data),
                "sample_size": train_data.tensors[0].shape[1],
                "args.dataset": args.dataset,
                "args.config_file": args.config_file,
            },
            f,
        )