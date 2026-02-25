import torch
import json
import numpy as np


def debug_shape(tensor, name=None):
    """
    Print shape and device information for a tensor.

    Args:
        tensor: PyTorch tensor or a collection of tensors
        name: Optional name to identify the tensor in the output
    """
    if name is None:
        # Try to get the variable name from the caller's frame
        import inspect

        frame = inspect.currentframe().f_back
        try:
            for var_name, var_val in frame.f_locals.items():
                if var_val is tensor:
                    name = var_name
                    break
        except:
            name = "tensor"

    if isinstance(tensor, torch.Tensor):
        print(
            f"{name}: shape={tensor.shape}, device={tensor.device}, dtype={tensor.dtype}"
        )
    elif isinstance(tensor, (list, tuple)) and all(
        isinstance(t, torch.Tensor) for t in tensor
    ):
        print(f"{name}: [")
        for i, t in enumerate(tensor):
            print(f"  [{i}]: shape={t.shape}, device={t.device}, dtype={t.dtype}")
        print("]")
    else:
        print(f"{name}: type={type(tensor)}")


def get_beta_schedule(config):
    if config["beta_type"] == "linear":
        betas = torch.linspace(
            config["beta_start"], config["beta_end"], config["noise_step"]
        )

    elif config["beta_type"] == "cosine":
        steps = config["noise_step"] + 1
        x = torch.linspace(0, config["noise_step"], steps)
        alphas_cumprod = (
            torch.cos(((x / config["noise_step"]) + 0.008) / 1.008 * torch.pi / 2) ** 2
        )
        alphas_cumprod /= alphas_cumprod[0].clone()
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=1e-8, max=0.999)

    elif config["beta_type"] == "quadratic":
        betas = (
            torch.linspace(
                config["beta_start"] ** 0.5,
                config["beta_end"] ** 0.5,
                config["noise_step"],
            )
            ** 2
        )

    elif config["beta_type"] == "sigmoid":
        steps = torch.linspace(-6, 6, config["noise_step"])
        betas = (
            torch.sigmoid(steps) * (config["beta_end"] - config["beta_start"])
            + config["beta_start"]
        )

    elif config["beta_type"] == "exponential":
        betas = config["beta_start"] * (
            config["beta_end"] / config["beta_start"]
        ) ** torch.linspace(0, 1, config["noise_step"])

    else:
        raise ValueError(f"Unknown beta schedule: {config['beta_type']}")

    return betas


def ensure_numpy(data):
    """
    Convert input data to numpy array.

    Args:
        data: Input data (tensor, numpy array, list, tuple, or scalar)

    Returns:
        numpy.ndarray: Data converted to numpy array
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    elif isinstance(data, (int, float)):
        return np.array([data])
    else:
        # Try to convert using numpy
        try:
            return np.array(data)
        except Exception as e:
            raise TypeError(f"Cannot convert {type(data)} to numpy array: {e}")


def ensure_tensor(data, device=None, dtype=None):
    """
    Convert input data to PyTorch tensor.

    Args:
        data: Input data (tensor, numpy array, list, tuple, or scalar)
        device: Target device for the tensor (optional)
        dtype: Target dtype for the tensor (optional)

    Returns:
        torch.Tensor: Data converted to PyTorch tensor
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        tensor = torch.tensor(data)
    elif isinstance(data, (int, float)):
        tensor = torch.tensor([data])
    else:
        # Try to convert using torch.tensor
        try:
            tensor = torch.tensor(data)
        except Exception as e:
            raise TypeError(f"Cannot convert {type(data)} to torch tensor: {e}")

    # Apply dtype conversion if specified
    if dtype is not None:
        tensor = tensor.to(dtype)

    # Apply device transfer if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def load_archetypes(x):
    with open("data/multidimensional_config_5_archetypes.json", "r") as f:
        data_config = json.load(f)

    archetypes = (
        torch.tensor(data_config.get("archetypes", None))
        .to(torch.float32)
        .to(x.device)  # [K, dx]
    )

    return archetypes
