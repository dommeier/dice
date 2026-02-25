import json
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

from mixed_diffusion.helpers import get_beta_schedule, load_archetypes
from mixed_diffusion.utils import mkdir, save_data, tensor_to_img, img_on_grid
from mixed_diffusion.wasserstein_distance import wasserstein_distance_from_samples


def log_prob(x, mu, sigma):
    """
    Log of the target probability distribution (Gaussian in this example).
    Args:
        x (np.ndarray): Point at which to evaluate the log probability.
        mu (np.ndarray): Mean of the Gaussian.
        sigma (np.ndarray): Covariance matrix of the Gaussian.

    Returns:
        float: Log probability of x.
    """
    d = len(mu)
    diff = x - mu
    return -0.5 * (
        np.log((2 * np.pi) ** d * np.linalg.det(sigma))
        + diff.T @ np.linalg.inv(sigma) @ diff
    )


def gradient_log_prob(x, mu, sigma):
    """
    Gradient of the log probability with respect to x.
    Args:
        x (np.ndarray): Point at which to evaluate the gradient.
        mu (np.ndarray): Mean of the Gaussian.
        sigma (np.ndarray): Covariance matrix of the Gaussian.

    Returns:
        np.ndarray: Gradient of the log probability at x.
    """
    return -np.linalg.inv(sigma) @ (x - mu)


def langevin_monte_carlo(args, mu, sigma):
    """
    Langevin Monte Carlo sampling.
    Args:
        mu (np.ndarray): Mean of the target Gaussian distribution.
        sigma (np.ndarray): Covariance matrix of the target Gaussian.
        num_samples (int): Number of samples to generate.
        step_size (float): Step size for the Langevin dynamics.
        burn_in (int): Number of iterations to discard as burn-in.

    Returns:
        np.ndarray: Generated samples.
    """
    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()

    d = mu.shape[0]
    samples = []

    # Initialize the chain at a random point
    x = np.random.randn(d)

    for i in range(args.num_samples + args.burn_in):
        # Compute the gradient of the log probability
        grad = gradient_log_prob(x, mu, sigma)

        # Langevin dynamics update
        x = (
            x
            + 0.5 * args.step_size * grad
            + np.sqrt(args.step_size) * np.random.randn(d)
        )

        # Store the sample after burn-in
        if i >= args.burn_in:
            samples.append(x)

    return np.array(samples)


def naive_sampling(args, mu, sigma):
    """
    Naive sampling from a Gaussian distribution.
    Args:
        mu (np.ndarray): Mean of the target Gaussian distribution.
        sigma (np.ndarray): Covariance matrix of the target Gaussian.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Generated samples.
    """
    # if mu is a 2d tensor, like (1, 28, 28), then mu = mu.reshape(-1)
    # Check device and move to CPU for sampling if using MPS
    device = mu.device
    cpu_fallback = device.type == "mps"

    if cpu_fallback:
        # print("Using CPU fallback for MultivariateNormal sampling")
        # Move to CPU for sampling
        mu_cpu = mu.cpu()
        sigma_cpu = sigma.cpu()

        mean_dim = mu_cpu.shape[-1]
        # Print shapes for debugging
        # print(f"mu shape: {mu.shape}")
        # print(f"sigma shape: {sigma.shape}")

        if mean_dim == 1:
            samples = torch.normal(
                mean=mu_cpu.squeeze(),  # Remove the last dimension
                std=torch.sqrt(sigma_cpu[0, 0]),  # Extract the variance value
            ).unsqueeze(-1)
        else:
            # Sample on CPU
            samples = torch.distributions.MultivariateNormal(
                loc=mu_cpu, covariance_matrix=sigma_cpu
            ).sample()

        # print(f"samples shape {samples.shape}")

        # Move back to original device
        return samples.to(device)
    else:
        return torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=sigma
        ).sample((args.num_samples,))


def sample_images(
    config,
    model,
    initial_x=None,
    num_samples=16,
    initial_step=None,
    image_size=(1, 28, 28),
    conditioning_vector=None,
    rho=None,
):
    if initial_step is None:
        initial_step = config["noise_step"] - 1

    betas = get_beta_schedule(config)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    model.eval()
    device = next(model.parameters()).device

    if rho:
        if alphas_cumprod[0] <= 1 / (1 + rho**2):
            initial_step = 1
        else:
            initial_step = torch.max(torch.nonzero(alphas_cumprod > 1 / (1 + rho**2)))
        # print(f"Initial step: {initial_step}")
        # print(f"Applied Factor: {torch.sqrt(alphas_cumprod[initial_step])}")
        initial_x = torch.sqrt(alphas_cumprod[initial_step]) * initial_x

    # print("Denoising images starting from step", initial_step, "......")
    with torch.no_grad():
        if initial_x is not None:
            x = initial_x.to(device)
        else:
            x = torch.randn((num_samples, *image_size)).to(device)
            # x = torch.randn((num_samples, 1, 28, 28)).to(device)
        for t in reversed(range(initial_step)):
            t_tensor = torch.tensor([t] * x.shape[0]).to(device)

            noise_pred = model(x, t_tensor, archetype_labels=conditioning_vector)
            # noise_pred = model(x, t_tensor, None)
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]

            # Update x_t to x_{t-1}
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred
            )
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(betas[t]) * noise

    model.train()

    return x, initial_step


def normalize_z(z, expected_range=(0, 1)):
    """
    Normalize z to match the expected range of the diffusion model.
    """
    z_min, z_max = z.min(), z.max()
    if expected_range == (-1, 1):
        z = 2 * (z - z_min) / (z_max - z_min + 1e-8) - 1  # Scale to [-1, 1]
    elif expected_range == (0, 1):
        z = (z - z_min) / (z_max - z_min + 1e-8)  # Scale to [0, 1]
    return z


def sample_z_given_x_y(args, y, x, rho, observation_transform):
    """
    Sample z from P(z|y,x) using elementwise Gaussian sampling while preserving spatial structure.
    """

    Sigma = torch.eye(y.shape[1], device=y.device)
    if Sigma.device.type == "mps":
        Sigma_cpu = Sigma.cpu()
        Sigma_inv_cpu = torch.inverse(Sigma_cpu)
        Sigma_inv = Sigma_inv_cpu.to(Sigma.device)
    else:
        Sigma_inv = torch.inverse(Sigma)

    # print(f"Sigma shape: {Sigma.shape}")
    # print(f"Sigma : {Sigma}")
    # print(f"Sigma ** -1 : {Sigma**-1}")
    # print(f"Observation transform shape: {observation_transform.shape}")
    # print(
    #     f"observation.t @ sigma**-1 shape: {(observation_transform.T @ (Sigma**-1)).shape}"
    # )
    # print(
    #     f"observation.t @ sigma**-1 @ observation shape: {(observation_transform.T @ (Sigma**-1) @ observation_transform).shape}"
    # )

    # print(f"torch.eye shape: {torch.eye(x.shape[1], device=y.device).shape}")
    # print(f"rho squared shape: {(rho**2).shape}")
    # print(
    #     f"torch eye / rho squared shape: {(torch.eye(x.shape[1], device=y.device) / (rho**2)).shape}"
    # )

    lambda_ = (
        observation_transform.T @ (Sigma_inv) @ observation_transform
    ) + torch.eye(x.shape[1], device=y.device) / (rho**2)

    lambda_inv = torch.inverse(lambda_)

    m_of_x = (y @ (observation_transform.T @ (Sigma_inv)).T + x / (rho**2)) @ (
        lambda_inv
    ).T

    # print(f"m_of_x shape: {m_of_x.shape}")
    # print(f"Lambda shape: {lambda_.shape}")
    # print(f"Lambda diag shape: {torch.diag(lambda_).shape}")

    # Compute the Cholesky factor of lambda_inv (i.e., covariance matrix)
    # Check if we need CPU fallback for MPS device
    if lambda_inv.device.type == "mps":
        print(
            "Using CPU fallback for Cholesky decomposition. You should speed this up!"
        )
        lambda_inv_cpu = lambda_inv.cpu()
        L_cpu = torch.linalg.cholesky(lambda_inv_cpu)
        L = L_cpu.to(lambda_inv.device)
    else:
        L = torch.linalg.cholesky(lambda_inv)  # shape [D, D]

    # Draw standard normal noise
    eps = torch.randn_like(m_of_x)  # shape [B, D]

    # Apply the transformation
    z = m_of_x + eps @ L.T  # shape [B, D]

    return z


def get_rho_schedule(args):
    """
    Generate a rho schedule for Gibbs sampling.
    """
    alpha = 0.9
    if args.rho_scheduling_type == "exponential":
        # This follows implementation in https://arxiv.org/pdf/2405.18782
        return torch.max(
            args.rho_start * alpha ** torch.arange(args.gibbs_iterations),
            torch.tensor(args.rho_end),
        )
    elif args.rho_scheduling_type == "linear":
        return torch.linspace(args.rho_start, args.rho_end, args.gibbs_iterations)
    else:
        raise ValueError("Invalid rho schedule type")


def map_y_back_to_x(y, observation_transform):
    """
    Map the observed y back to the latent x space using the inverse of the observation transform.
    Right now, we just use the transposed matrix.
    """
    return y @ observation_transform.to(y.device)


def initialize_x(args, y, observation_transform):
    if args.initial_x == "measurement":
        # Use the noisy measurement as the initial x
        x = y @ observation_transform
    elif args.initial_x == "random":
        # Use random noise as the initial x
        x = torch.randn_like(y @ observation_transform).to(y.device)
    else:
        raise ValueError(
            f"Invalid initial_x type: {args.initial_x}. Choose from 'measurement', 'true_x', or 'random'."
        )
    return x


def gibbs_sampling(args, y, model, config, observation_transform, data_config):
    num_iterations = args.gibbs_iterations

    x = initialize_x(args, y, observation_transform)

    # Rho scheduling with exponential or linear
    rho_schedule = get_rho_schedule(args)

    betas = get_beta_schedule(config)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    variance_schedule = 1 - alphas_cumprod

    if args.conditioning_vector:
        print(f"Loading conditioning vector from {args.conditioning_vector}")
        with open(args.conditioning_vector, "r") as f:
            conditioning_vector = torch.tensor(json.load(f)).to(x.device)
        print(f"Conditioning vector: {conditioning_vector}")
    else:
        print("Using uniform conditioning vector")
        conditioning_vector = torch.tensor([-1] * 5).to(x.device).to(torch.float32)

    pbar = tqdm(
        range(num_iterations),
        desc=f"Gibbs sampling with rho: {rho_schedule[0]:.2f}",
    )

    if args.save_trajectories:
        sampled_z = []
        sampled_x = [x]

    print("Starting Gibbs sampling...")
    for i in pbar:
        current_rho = rho_schedule[i]
        z = sample_z_given_x_y(args, y, x, current_rho, observation_transform)
        if args.save_trajectories:
            sampled_z.append(z)
        # print(f"z shape: {z.shape}, rho {rho_schedule[i]}")

        # repeat conditioning vector for each sample (in dim 2)
        repeated_conditioning_vector = None
        if config["condition"]:
            repeated_conditioning_vector = conditioning_vector.repeat(z.shape[0], 1)

        x, initial_step = sample_images(
            config,
            model,
            initial_x=z,
            num_samples=z.shape[0],
            initial_step=args.initial_step,
            image_size=y.shape[1:],
            rho=current_rho,
            conditioning_vector=repeated_conditioning_vector,
        )
        if args.save_trajectories:
            sampled_x.append(x)

        progress_bar_desc = f"Gibbs sampling with rho: {rho_schedule[i]:.2f} ｜initial step: {initial_step}"

        pbar.set_description(progress_bar_desc)

    if args.save_trajectories:
        torch.save(
            {
                "x": sampled_x,
                "z": sampled_z,
                "y": y,
                "rho_schedule": rho_schedule,
            },
            f"{args.result_dir}/trajectory.pkl",
        )

    return x


def log_likelihood_y_given_x(x, y, A, Sigma):
    """
    Compute log p(y | x) assuming y = Ax + ε, ε ~ N(0, Σ)

    Args:
        x: [B, dx] tensor of latent variables
        y: [B, dy] tensor of observations
        A: [dy, dx] linear observation matrix

    Returns:
        log_likelihood: [B] log-likelihood for each sample
    """
    mean = x @ A.T  # [B, dy]
    dist = MultivariateNormal(loc=mean, covariance_matrix=Sigma)
    return dist.log_prob(y)  # [B]


# def sample_images(model, num_samples=16, ):
#     model.eval()
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         x = torch.randn((num_samples, 1, 28, 28)).to(device)  # Start with pure noise
#         for t in reversed(range(args.noise_step)):
#             t_tensor = torch.tensor([t] * num_samples).to(device)
#             noise_pred = model(x, t_tensor)
#             alpha_t = alphas[t]
#             alpha_cumprod_t = alphas_cumprod[t]

#             # Update x_t to x_t-1
#             x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred)
#             if t > 0:
#                 noise = torch.randn_like(x)
#                 x += torch.sqrt(betas[t]) * noise

#         x = x.clamp(-1, 1)
#         return x


# def sample_z_given_x_y(y, x, epsilon=0.1):
#     return y + epsilon * np.random.randn(*x.shape)

# def sample_x_given_z(z, diffusion_model, epsilon=0.1):
#     z_tensor = torch.tensor(z, dtype=torch.float32)
#     with torch.no_grad():
#         predicted_noise = diffusion_model(z_tensor).numpy()
#     return z - epsilon * predicted_noise

# def gibbs_sampling(y, diffusion_model, num_samples=100, epsilon=0.1):
#     x_samples = np.random.randn(*y.shape)  # Initialize x
#     for _ in range(num_samples):
#         z_samples = sample_z_given_x_y(y, x_samples, epsilon)
#         x_samples = sample_x_given_z(z_samples, diffusion_model, epsilon)
#     return x_samples
