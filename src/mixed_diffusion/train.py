from mixed_diffusion.training.generate_conditional_distribution import (
    generate_conditional_distribution,
)
from .helpers import get_beta_schedule
from .sampling import sample_images
from .utils import mkdir, save_data


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np


def load_noise_transform(config, sample_x, device):
    """Load noise transform matrix from config or create identity matrix."""
    noise_transform = torch.eye(sample_x.shape[-1]).to(device)

    if config.get("noise_transform", False):
        if config["noise_transform"].endswith(".npy"):
            noise_transform = (
                torch.tensor(np.load(config["noise_transform"]))
                .to(torch.float32)
                .to(device)
            )
        else:
            with open(config["noise_transform"], "r") as f:
                noise_transform = torch.tensor(json.load(f)).to(device)

        print(f"Loaded noise transform: {config['noise_transform']}")
        print(
            f"Noise will be generated in {noise_transform.shape[1]} dimensions, and mapped to {noise_transform.shape[0]} dimensions."
        )

    return noise_transform.to(device)


def add_noise(x0, alpha, noise_transform):

    # Draw noise in dimension dictated by noise_transform
    noise = torch.randn(x0.shape[0], *noise_transform.shape[1:]).to(x0.device)
    sqrt_alpha = torch.sqrt(alpha).to(x0.device).view(-1, *([1] * (x0.ndim - 1)))
    sqrt_one_minus_alpha = (
        torch.sqrt(1 - alpha).to(x0.device).view(-1, *([1] * (x0.ndim - 1)))
    )

    transformed_noise = torch.matmul(noise, noise_transform.T)

    return (
        sqrt_alpha * x0 + sqrt_one_minus_alpha * transformed_noise,
        transformed_noise,
    )


def train(config, dataloader, model, device, args):
    if config["model"] in ["TabularDiffusionMLP", "TabularDiffusionTransformer"]:
        config["hidden_dim"] = config["hidden_dim"]
        config["num_blocks"] = config["num_blocks"]

    config["model_dir"] = args.model_dir.rstrip("/")

    mkdir(f"{config['model_dir']}/checkpoints")
    mkdir(f"{config['model_dir']}/diff_samples")

    # input_dim = 28 * 28

    # diffusion_model = DiffusionModel(input_dim=input_dim)
    # optimizer = optim.Adam(diffusion_model.parameters(), lr=args.learning_rate)

    betas = get_beta_schedule(config)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] * len(dataloader),
        eta_min=0,
    )
    model.train()
    mse = nn.MSELoss()

    training_losses = []

    # Get a sample batch to determine the correct shape
    sample_batch = next(iter(dataloader))
    sample_x = sample_batch[0]

    noise_transform = load_noise_transform(config, sample_x, device)

    print(f"Shape of x0: {dataloader.dataset[0][0].shape}")

    try:
        for epoch in tqdm(
            range(config["epochs"]),
            desc=f"Epoch Progress",
            position=0,
            total=config["epochs"],
        ):
            total_loss = 0
            for i, (x0, label) in enumerate(dataloader):
                label = label.to(device)
                # Handle different conditioning methods
                if config.get("condition_training_method") == "archetype":
                    # Use archetype conditioning - pass labels directly to model
                    condition = None
                    archetype_labels = label
                elif config["condition_training_method"] == "one_hot":
                    one_hot_labels = torch.nn.functional.one_hot(
                        label, num_classes=config["num_classes"]
                    )
                    condition = one_hot_labels
                    archetype_labels = None
                elif config["condition_training_method"] == "dirichlet":
                    condition = generate_conditional_distribution(
                        label, num_classes=config["num_classes"]
                    )
                    archetype_labels = None
                else:
                    condition = None
                    archetype_labels = None

                # print(f"Condition: {condition}")

                x0 = x0.to(device)
                # print(f"x0 shape: {x0.shape}")
                t = torch.randint(0, config["noise_step"], (x0.shape[0],)).to(device)

                noisy_x, noise = add_noise(x0, alphas_cumprod[t], noise_transform)

                # print(f"noisy_x shape: {noisy_x.shape}")
                # print(f"noise shape: {noise.shape}")

                # print(
                #     f"Devices: x0={x0.device}, noisy_x={noisy_x.device}, noise={noise.device}, t={t.device}, condition={condition.device}"
                # )

                # Call model with appropriate conditioning
                if config.get("condition_training_method") == "archetype":
                    noise_pred = model(
                        noisy_x, t, condition=None, archetype_labels=archetype_labels
                    )
                else:
                    noise_pred = model(noisy_x, t, condition, archetype_labels=None)

                noise_pred = noise_pred.reshape_as(noise)
                loss = mse(noise_pred, noise)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            image_size = x0.shape[1:]
            training_loss = total_loss / len(dataloader)
            training_losses.append(training_loss)
            print(
                f"Epoch {epoch + 1} Loss: {training_loss} LR: {optimizer.param_groups[0]['lr']}"
            )

            # Only save checkpoints based on the checkpoints_to_keep flag
            if (
                args.checkpoints_to_keep == 0
                or (epoch + 1) % args.checkpoints_to_keep == 0
            ):
                torch.save(
                    model.state_dict(),
                    f"{config['model_dir']}/checkpoints/model_epoch_{epoch}.pt",
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted...\nGracefully exiting...")

    results = {
        "final_epoch": epoch + 1,
        "training_losses": training_losses,
    }

    return model, config, results
