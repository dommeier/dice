"""
CFG (Classifier-Free Guidance) sampling functionality for archetype conditioning.

This module provides CFG sampling capabilities that work with the archetype conditioning
system to enable controlled generation with different archetype labels.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List
from tqdm import tqdm


def ddpm_cfg_sampling(
    model: nn.Module,
    shape: tuple,
    num_steps: int,
    device: str,
    archetype_labels: torch.Tensor,
    cfg_scale: float = 1.5,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    noise_schedule: str = "linear",
) -> torch.Tensor:
    """
    DDPM sampling with Classifier-Free Guidance for archetype conditioning.

    Args:
        model: The trained diffusion model with archetype conditioning
        shape: Shape of the samples to generate (batch_size, ...)
        num_steps: Number of denoising steps
        device: Device to run on
        archetype_labels: Target archetype labels for generation
        cfg_scale: CFG scale w (typically 1.5-3.0)
        beta_start: Starting beta value for noise schedule
        beta_end: Ending beta value for noise schedule
        noise_schedule: Type of noise schedule ("linear" or "cosine")

    Returns:
        Generated samples
    """
    # Set up noise schedule
    if noise_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    elif noise_schedule == "cosine":
        betas = cosine_beta_schedule(num_steps, beta_start, beta_end, device=device)
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

    # Initialize with pure noise
    x = torch.randn(shape, device=device)

    # Ensure archetype_labels are on the correct device
    archetype_labels = archetype_labels.to(device)

    # Reverse diffusion process
    model.eval()
    with torch.no_grad():
        for i in tqdm(reversed(range(num_steps)), desc="CFG Sampling"):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            # Apply CFG
            if hasattr(model, "apply_cfg"):
                # Use model's built-in CFG method
                predicted_noise = model.apply_cfg(x, t, archetype_labels, cfg_scale)
            else:
                # Fallback: manual CFG implementation
                predicted_noise = apply_cfg_manual(
                    model, x, t, archetype_labels, cfg_scale
                )

            # DDPM update step
            if i > 0:
                # Add noise for next step
                noise = torch.randn_like(x)
                alpha_t = alphas_cumprod[i]
                alpha_t_prev = alphas_cumprod_prev[i]
                beta_t = betas[i]

                # Compute coefficients
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(
                    alpha_t
                )
                pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clamp to prevent extreme values

                # Compute mean of q(x_{t-1} | x_t, x_0)
                mean = (torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t)) * pred_x0 + (
                    torch.sqrt(1 - beta_t) * (1 - alpha_t_prev) / (1 - alpha_t)
                ) * x

                # Add noise
                if i > 0:
                    variance = (1 - alpha_t_prev) / (1 - alpha_t) * beta_t
                    x = mean + torch.sqrt(variance) * noise
                else:
                    x = mean
            else:
                # Final step - no noise
                alpha_t = alphas_cumprod[i]
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(
                    alpha_t
                )
                x = torch.clamp(pred_x0, -1, 1)

    return x


def apply_cfg_manual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    archetype_labels: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """
    Manual implementation of CFG when model doesn't have built-in CFG method.

    Args:
        model: The diffusion model
        x: Noisy input
        t: Timestep
        archetype_labels: Target archetype labels
        cfg_scale: CFG scale

    Returns:
        CFG-guided noise prediction
    """
    # Get unconditional prediction (null condition)
    if hasattr(model, "archetype_conditioning"):
        null_condition = model.archetype_conditioning.get_unconditional_condition(
            x.shape[0]
        )
        eps_uncond = model(x, t, condition=null_condition, archetype_labels=None)
    else:
        eps_uncond = model(x, t, condition=None, archetype_labels=None)

    # Get conditional prediction
    eps_cond = model(x, t, condition=None, archetype_labels=archetype_labels)

    # Apply CFG: ε̂ = ε_θ(x_t, t, ∅) + w(ε_θ(x_t, t, e) - ε_θ(x_t, t, ∅))
    eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    return eps_cfg


def cosine_beta_schedule(
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Cosine noise schedule for diffusion models.

    Args:
        timesteps: Number of timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        device: Device to create tensor on

    Returns:
        Beta values for cosine schedule
    """
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sample_with_archetype_guidance(
    model: nn.Module,
    num_samples: int,
    input_dim: int,
    archetype_labels: Union[torch.Tensor, List[int]],
    config: dict,
    device: str = "cuda",
    cfg_scale: float = 1.5,
) -> torch.Tensor:
    """
    High-level function to sample with archetype guidance.

    Args:
        model: Trained diffusion model with archetype conditioning
        num_samples: Number of samples to generate
        input_dim: Input dimension
        archetype_labels: Target archetype labels (can be tensor or list)
        config: Model configuration
        device: Device to run on
        cfg_scale: CFG scale

    Returns:
        Generated samples
    """
    # Convert archetype_labels to tensor if needed
    if isinstance(archetype_labels, list):
        archetype_labels = torch.tensor(archetype_labels, device=device)
    elif not isinstance(archetype_labels, torch.Tensor):
        archetype_labels = torch.tensor([archetype_labels] * num_samples, device=device)

    # Ensure we have the right number of labels
    if len(archetype_labels) != num_samples:
        if len(archetype_labels) == 1:
            archetype_labels = archetype_labels.repeat(num_samples)
        else:
            raise ValueError(
                f"Number of archetype labels ({len(archetype_labels)}) "
                f"must match number of samples ({num_samples})"
            )

    # Generate samples
    shape = (num_samples, input_dim)
    samples = ddpm_cfg_sampling(
        model=model,
        shape=shape,
        num_steps=config.get("noise_step", 1000),
        device=device,
        archetype_labels=archetype_labels,
        cfg_scale=cfg_scale,
        beta_start=config.get("beta_start", 0.0001),
        beta_end=config.get("beta_end", 0.02),
        noise_schedule=config.get("noise_schedule", "linear"),
    )

    return samples


def compare_archetype_generations(
    model: nn.Module,
    num_samples_per_archetype: int,
    input_dim: int,
    num_archetypes: int,
    config: dict,
    device: str = "cuda",
    cfg_scale: float = 1.5,
) -> dict:
    """
    Generate samples for all archetypes for comparison.

    Args:
        model: Trained diffusion model
        num_samples_per_archetype: Number of samples per archetype
        input_dim: Input dimension
        num_archetypes: Number of archetypes
        config: Model configuration
        device: Device to run on
        cfg_scale: CFG scale

    Returns:
        Dictionary with samples for each archetype
    """
    results = {}

    for archetype_id in range(num_archetypes):
        print(f"Generating samples for archetype {archetype_id}...")

        # Create labels for this archetype
        archetype_labels = torch.full(
            (num_samples_per_archetype,), archetype_id, device=device, dtype=torch.long
        )

        # Generate samples
        samples = sample_with_archetype_guidance(
            model=model,
            num_samples=num_samples_per_archetype,
            input_dim=input_dim,
            archetype_labels=archetype_labels,
            config=config,
            device=device,
            cfg_scale=cfg_scale,
        )

        results[f"archetype_{archetype_id}"] = samples

    return results


# Example usage
if __name__ == "__main__":
    # This is just for testing - would need actual model and config
    print("CFG sampling module loaded successfully!")
    print("Use sample_with_archetype_guidance() for high-level sampling")
    print("Use ddpm_cfg_sampling() for low-level control")
