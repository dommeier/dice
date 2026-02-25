import torch
from torchvision import datasets, transforms

from mixed_diffusion.data_loading.load_single_cell_data import load_single_cell_data

from .load_synthetic_data import load_synthetic_data


def get_data(config, transform):
    if config["dataset"] == "mnist":
        train_data = datasets.MNIST(
            root=config["data_file"], train=True, download=True, transform=transform
        )
        test_data = datasets.MNIST(
            root=config["data_file"], train=False, download=True, transform=transform
        )
    elif config["dataset"] == "cifar10":
        train_data = datasets.CIFAR10(
            root=config["data_file"], train=True, download=True, transform=transform
        )
        test_data = datasets.CIFAR10(
            root=config["data_file"], train=False, download=True, transform=transform
        )
    elif config["dataset"] == "celeba":
        transform = transforms.Compose(
            [
                transforms.CenterCrop(178),  # Recommended center crop for CelebA
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train_data = datasets.CelebA(
            root=config["data_file"], split="train", download=True, transform=transform
        )
        test_data = datasets.CelebA(
            root=config["data_file"], split="test", download=True, transform=transform
        )
    elif config["dataset"] == "lfw":
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train_data = datasets.LFWPeople(
            root=config.data_file, split="train", download=True, transform=transform
        )
        test_data = datasets.LFWPeople(
            root=config["data_file"], split="test", download=True, transform=transform
        )
    elif config["dataset"] == "celeba-hq":
        train_data = datasets.CelebAHQ(
            root=config["data_file"], split="train", download=True, transform=transform
        )
        test_data = datasets.CelebAHQ(
            root=config["data_file"], split="test", download=True, transform=transform
        )
    elif config["dataset"] == "single_cell":
        # Load single-cell data
        try:
            train_data, test_data, data_config = load_single_cell_data(config)
        except ValueError:
            raise ValueError(f"Dataset {config['dataset']} not supported.")
    else:
        try:
            train_data, test_data, data_config = load_synthetic_data(config)
        except ValueError:
            raise ValueError(f"Dataset {config.dataset} not supported.")

    # Add noise only for synthetic data (which returns TensorDataset)
    train_data = torch.utils.data.TensorDataset(
        train_data.tensors[0]
        + config["noise_level_for_training"] * torch.randn_like(train_data.tensors[0]),
        train_data.tensors[1],
    )

    return train_data, test_data, data_config
