import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from mixed_diffusion.data_loading.data_loading import get_data
from mixed_diffusion.train import load_noise_transform
from mixed_diffusion.utils import mkdir, save_data


def create_data_transform():
    """Create the standard data transformation"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def select_test_data(args, train_data, test_data):
    """Select data based on test_on flag"""
    if args.test_on == "train":
        y_test = train_data.tensors[0]
        true_labels_test = train_data.tensors[1]
        print(
            f"Using TRAINING data: {y_test.shape[0]} samples with shape {y_test.shape}"
        )
    elif args.test_on == "test":
        y_test = test_data.tensors[0]
        true_labels_test = test_data.tensors[1]
        print(f"Using TEST data: {y_test.shape[0]} samples with shape {y_test.shape}")
    elif args.test_on == "both":
        train_y = train_data.tensors[0]
        train_labels = train_data.tensors[1]
        test_y = test_data.tensors[0]
        test_labels = test_data.tensors[1]
        y_test = torch.cat([train_y, test_y], dim=0)
        true_labels_test = torch.cat([train_labels, test_labels], dim=0)
        print(
            f"Using BOTH train and test data: {y_test.shape[0]} samples with shape {y_test.shape}"
        )
        print(f"  - Train samples: {train_y.shape[0]}")
        print(f"  - Test samples: {test_y.shape[0]}")
    elif args.test_on == "one_type_each":
        true_labels_test = test_data.tensors[1]
        # 1. Sort labels and x_values based on the labels
        sorted_labels, sorted_indices = torch.sort(true_labels_test)
        sorted_values = test_data.tensors[0][sorted_indices]

        # 2. Create a boolean mask to identify the first occurrence of each unique label
        # We pad the sorted labels with a value at the beginning to handle the first element
        mask = torch.cat(
            [torch.tensor([True]), sorted_labels[1:] != sorted_labels[:-1]]
        )

        # 3. Apply the mask to filter the sorted tensors
        y_test = sorted_values[mask]
        true_labels_test = sorted_labels[mask]
        print(
            f"Using ONE TYPE EACH: {y_test.shape[0]} samples with shape {y_test.shape}"
        )
    else:
        raise ValueError(
            f"Invalid test_on value: {args.test_on}. Must be 'train', 'test', or 'both'"
        )

    # Apply subsampling if max_test_samples is specified
    if args.max_test_samples is not None and y_test.shape[0] > args.max_test_samples:
        print(
            f"Subsampling test data from {y_test.shape[0]} to {args.max_test_samples} samples"
        )

        # Stratified subsampling to maintain class distribution
        unique_labels, label_counts = torch.unique(true_labels_test, return_counts=True)
        total_samples = y_test.shape[0]

        # Calculate samples per class proportionally
        samples_per_class = (
            label_counts.float() / total_samples * args.max_test_samples
        ).int()

        # Handle rounding issues by distributing remaining samples
        remaining_samples = args.max_test_samples - samples_per_class.sum().item()
        if remaining_samples > 0:
            # Add remaining samples to classes with largest fractional parts
            fractional_parts = (
                label_counts.float() / total_samples * args.max_test_samples
            ) - samples_per_class.float()
            _, top_indices = torch.topk(fractional_parts, remaining_samples)
            samples_per_class[top_indices] += 1

        # Sample from each class
        generator = torch.Generator()
        generator.manual_seed(42)  # Fixed seed for reproducible subsampling
        selected_indices = []

        for label, n_samples in zip(unique_labels, samples_per_class):
            if n_samples > 0:
                class_mask = true_labels_test == label
                class_indices = torch.where(class_mask)[0]

                # Randomly select n_samples from this class
                perm = torch.randperm(class_indices.shape[0], generator=generator)
                selected_class_indices = class_indices[perm[:n_samples]]
                selected_indices.append(selected_class_indices)

        # Combine all selected indices
        indices = torch.cat(selected_indices)

        # Apply subsampling
        y_test = y_test[indices]
        true_labels_test = true_labels_test[indices]

        print(f"Subsampled test data shape: {y_test.shape}")
        print(
            f"Class distribution after stratified sampling: {torch.unique(true_labels_test, return_counts=True)}"
        )

    y_train = train_data.tensors[0]
    true_labels_train = train_data.tensors[1]

    return (y_test, true_labels_test, y_train, true_labels_train)


def prepare_observations(y, true_labels, args, config, device):
    """Prepare observations with noise and transformations"""

    y = y.to(device)
    true_labels = true_labels.to(device)

    # Create repeated samples if needed
    indices = torch.arange(y.shape[0], device=device).repeat_interleave(
        args.repeated_sampling_factor
    )
    print(f"True measurement (y) shape: {y.shape}")
    y_repeated = y[indices]
    print(f"Repeated measurements (y) shape: {y_repeated.shape}")

    # Add noise
    noise = torch.randn_like(y_repeated) * args.test_noise_level
    if args.same_noise:
        # use the same noise in all rows
        noise = noise[0:1].repeat(y_repeated.shape[0], 1)

    y_noised_repeated = y_repeated + noise

    true_labels_repeated = true_labels[indices]

    return y_noised_repeated, true_labels_repeated, y_repeated


def process_repeated_sampling(x_denoised, x0, args):
    """Process repeated sampling results based on the specified method"""
    if args.repeated_sampling_method == "mean":
        # Merge again
        x_denoised = x_denoised.view(
            x0.shape[0], args.repeated_sampling_factor, *x0.shape[1:]
        )
        x_denoised = x_denoised.mean(dim=1)  # Average over the repeated samples
        print(f"x_denoised shape: {x_denoised.shape}")
    elif args.repeated_sampling_method == "none":
        pass
    else:
        raise ValueError(
            f"Invalid repeated_sampling_method: {args.repeated_sampling_method}. "
        )

    return x_denoised
