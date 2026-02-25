import torch
import os
import glob


def mse(x1, x2):
    """Calculate Mean Squared Error between two tensors"""
    return torch.mean((x1 - x2) ** 2).item()


def get_device(args):
    if args.mps:
        return "mps"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


def get_final_model_path(result_dir):
    result_dir = result_dir.rstrip("/")
    if os.path.exists(f"{result_dir}/checkpoints/model_final.pt"):
        return f"{result_dir}/checkpoints/model_final.pt"
    return None


def get_latest_checkpoint(result_dir):
    checkpoint_files = glob.glob(f"{result_dir}/checkpoints/model_epoch_*.pt")
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the directory.")
    latest_checkpoint = max(
        checkpoint_files, key=lambda x: int(x.split("_epoch_")[1].split(".pt")[0])
    )
    return latest_checkpoint


def get_linear_transform(dim_features, dim_observation):
    # TODO: implement:
    # - loading it from a file
    # - generating it randomly
    # - using identiy
    # - add and args option
    # Generates a rademacher distributed linear transform
    return torch.randint(0, 2, (dim_observation, dim_features)).float() * 2.0 - 1.0


def ensure_directories_exist(args):
    """Ensure result and model directories exist"""
    args.result_dir = args.result_dir.rstrip("/")
    args.model_dir = args.model_dir.rstrip("/")
    
    if os.path.exists(args.result_dir):
        print(f"Result directory {args.result_dir} already exists.")
    else:
        os.makedirs(args.result_dir)