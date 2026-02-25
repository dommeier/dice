import torch
import torch.nn.functional as F


def generate_conditional_distribution(labels, num_classes, concentration=1.0):
    """
    Generate probability distributions that favor the target labels

    Args:
        target_labels: int, list, or torch.Tensor of target labels
                      Can be a single label or batch of labels
        num_classes: Total number of classes
        concentration: Controls how peaked the distributions are
                      - Higher values = more peaked around target
                      - Lower values = more uniform/spread out
        device: torch device to put tensors on

    Returns:
        torch.Tensor of shape (batch_size, num_classes) with probability distributions
    """

    batch_size = labels.shape[0]

    # Create base Dirichlet parameters (alpha values)
    # Shape: (batch_size, num_classes)
    alpha = torch.ones(batch_size, num_classes, device=labels.device) * concentration

    # Add extra weight to target classes
    # Sample random boosts for each target class
    boosts = (
        torch.distributions.Exponential(
            torch.ones(batch_size, device=labels.device) * 2.0
        ).sample()
        * concentration
    )

    # Add boosts to the target classes
    alpha[torch.arange(batch_size), labels] += boosts

    # Sample from Dirichlet distribution using Gamma sampling
    # Dirichlet(α) can be sampled as Gamma(α_i) / sum(Gamma(α_i))
    gamma_samples = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
    prob_dists = F.normalize(gamma_samples, p=1, dim=1)

    return prob_dists
