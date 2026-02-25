import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from mixed_diffusion.models.get_model import get_model
from mixed_diffusion.train import train
from mixed_diffusion.utils import plot_loss_curve
from mixed_diffusion.main_utils import get_final_model_path, get_latest_checkpoint
from mixed_diffusion.config_manager import (
    save_model_config,
    save_training_results,
    save_training_data_info,
)
import numpy as np


class MixupDataset(Dataset):
    """Dataset that generates mixup samples on the fly between samples of the same label."""

    def __init__(
        self, x_data, labels, mixup_prob=0.5, mixup_alpha=1.0, n_mixup_points=2
    ):
        """
        Initialize MixupDataset.

        Args:
            x_data: Input data tensor [N, ...]
            labels: Label tensor [N]
            mixup_prob: Probability of applying mixup (default: 0.5)
            mixup_alpha: Beta distribution parameter for mixup (default: 1.0)
            n_mixup_points: Number of points to interpolate between (default: 2)
        """
        self.x_data = x_data
        self.labels = labels
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.n_mixup_points = max(2, n_mixup_points)  # Ensure at least 2 points

        # Group indices by label for efficient same-label sampling
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            label_item = label.item() if isinstance(label, torch.Tensor) else label
            if label_item not in self.label_to_indices:
                self.label_to_indices[label_item] = []
            self.label_to_indices[label_item].append(idx)

        # Convert to tensors for faster indexing
        for label in self.label_to_indices:
            self.label_to_indices[label] = torch.tensor(self.label_to_indices[label])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        label = self.labels[idx]
        label_item = label.item() if isinstance(label, torch.Tensor) else label

        # Apply mixup with probability mixup_prob
        if (
            np.random.random() < self.mixup_prob
            and len(self.label_to_indices[label_item]) >= self.n_mixup_points
        ):
            # Sample other indices with the same label
            same_label_indices = self.label_to_indices[label_item]
            # Remove current index to avoid mixing with itself
            other_indices = same_label_indices[same_label_indices != idx]

            if len(other_indices) >= self.n_mixup_points - 1:
                # Randomly select (n_mixup_points - 1) other samples with the same label
                selected_other_indices = other_indices[
                    torch.randperm(len(other_indices))[: self.n_mixup_points - 1]
                ]

                # Collect all samples to mix (including the current one)
                all_samples = [x]
                for other_idx in selected_other_indices:
                    all_samples.append(self.x_data[other_idx])

                # Generate mixing coefficients using Dirichlet distribution
                # This generalizes Beta distribution to N points
                if self.mixup_alpha > 0:
                    # Use Dirichlet distribution for N-way mixing
                    alphas = np.full(self.n_mixup_points, self.mixup_alpha)
                    mixing_coeffs = np.random.dirichlet(alphas)
                else:
                    # Equal mixing if alpha is 0
                    mixing_coeffs = np.full(
                        self.n_mixup_points, 1.0 / self.n_mixup_points
                    )

                # Mix the samples using the coefficients
                mixed_x = torch.zeros_like(x)
                for i, (sample, coeff) in enumerate(zip(all_samples, mixing_coeffs)):
                    mixed_x += coeff * sample

                return mixed_x, label

        # Return original sample if no mixup
        return x, label


def create_mixup_dataloader(
    x_data,
    labels,
    batch_size,
    mixup_prob=0.5,
    mixup_alpha=1.0,
    n_mixup_points=2,
    shuffle=True,
):
    """
    Create a DataLoader with mixup functionality.

    Args:
        x_data: Input data tensor
        labels: Label tensor
        batch_size: Batch size for DataLoader
        mixup_prob: Probability of applying mixup (default: 0.5)
        mixup_alpha: Beta distribution parameter for mixup (default: 1.0)
        n_mixup_points: Number of points to interpolate between (default: 2)
        shuffle: Whether to shuffle the data (default: True)

    Returns:
        DataLoader with mixup functionality
    """
    mixup_dataset = MixupDataset(
        x_data, labels, mixup_prob, mixup_alpha, n_mixup_points
    )
    return DataLoader(mixup_dataset, batch_size=batch_size, shuffle=shuffle)


def setup_and_train_model(
    config, x0_train, true_labels_train, device, args, train_data
):
    """Setup and train model from scratch"""
    model = get_model(config, x0_train.shape, device)

    # Print model parameter information
    print(f"\n🔧 Model Architecture: {config.get('model_type', 'Unknown')}")
    count_parameters(model, print_details=True)

    # Check if mixup dataloader should be used
    if config.get("use_mixup_dataloader", False):
        mixup_prob = config.get("mixup_prob", 0.5)
        mixup_alpha = config.get("mixup_alpha", 1.0)
        n_mixup_points = config.get("n_mixup_points", 2)

        print(
            f"Using mixup dataloader with prob={mixup_prob}, alpha={mixup_alpha}, n_points={n_mixup_points}"
        )
        train_dataloader = create_mixup_dataloader(
            x0_train,
            true_labels_train,
            batch_size=config["batch_size"],
            mixup_prob=mixup_prob,
            mixup_alpha=mixup_alpha,
            n_mixup_points=n_mixup_points,
            shuffle=True,
        )
    else:
        # Use standard dataloader
        train_data = TensorDataset(x0_train, true_labels_train)
        train_dataloader = DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True
        )

    model, config, results = train(config, train_dataloader, model, device, args)

    # Save model
    torch.save(model.state_dict(), f"{config['model_dir']}/checkpoints/model_final.pt")

    # Save configuration and results
    save_model_config(config)
    save_training_results(config, results)
    save_training_data_info(config, train_data, args)

    print("Model saved as model_final.pt")
    plot_loss_curve(
        results["training_losses"],
        "Loss curve",
        f"{config['model_dir']}/loss_curve.png",
    )

    return model, config, results


def load_existing_model(config, x0_shape, device, args):
    """Load existing model from checkpoint"""
    model = get_model(config, x0_shape, device)

    model_path = get_final_model_path(args.model_dir)
    if model_path is None:
        model_path = get_latest_checkpoint(args.model_dir)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    print(f"Model loaded from {model_path}")

    # Print model parameter information for loaded model
    print(f"\n📊 Loaded Model Architecture: {config.get('model_type', 'Unknown')}")
    count_parameters(model, print_details=False)  # Brief summary for loaded models

    return model


def count_parameters(model, print_details=True):
    """
    Count and print the total number of parameters in a PyTorch model.

    Args:
        model: PyTorch model
        print_details: Whether to print detailed parameter counts by layer

    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = 0
    trainable_params = 0

    if print_details:
        print("=" * 60)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 60)
        print(f"{'Layer Name':<30} {'Parameters':<15} {'Trainable':<10}")
        print("-" * 60)

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        if param.requires_grad:
            trainable_params += param_count
            trainable_status = "Yes"
        else:
            trainable_status = "No"

        if print_details:
            print(f"{name:<30} {param_count:<15,} {trainable_status:<10}")

    if print_details:
        print("-" * 60)
        print(f"{'TOTAL PARAMETERS':<30} {total_params:<15,}")
        print(f"{'TRAINABLE PARAMETERS':<30} {trainable_params:<15,}")
        print(
            f"{'NON-TRAINABLE PARAMETERS':<30} {(total_params - trainable_params):<15,}"
        )
        print("=" * 60)

        # Calculate memory usage (rough estimate)
        param_size_mb = (
            total_params * 4 / (1024 * 1024)
        )  # Assuming float32 (4 bytes per param)
        print(f"Estimated memory usage (parameters only): {param_size_mb:.2f} MB")
        print("=" * 60)
    else:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    return total_params, trainable_params
