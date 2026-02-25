import torch


def wasserstein_distance_from_samples(
    samples_p: torch.tensor, samples_q: torch.tensor, p=1, blur=0.05
):
    from geomloss import SamplesLoss

    """
    Compute the Wasserstein distance between two sets of samples using GeomLoss.

    Parameters:
      samples_p : torch.Tensor (shape: [num_samples, dim])
          Samples from distribution P.
      samples_q : torch.Tensor (shape: [num_samples, dim])
          Samples from distribution Q.
      p : int
          The order of the Wasserstein distance (p=1 for Wasserstein-1, p=2 for Wasserstein-2).
      blur : float
          Regularization parameter for the Sinkhorn divergence.

    Returns:
      float
          The estimated Wasserstein distance.
    """
    # Everything except last dimension is treated as batch dimension
    # and the last dimension is treated as the feature dimension.
    samples_p = samples_p.view(-1, samples_p.shape[-1])
    samples_q = samples_q.view(-1, samples_q.shape[-1])
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    distance = loss(samples_p, samples_q)
    return distance.item()


def main():
    # Example: Generate samples from two distributions
    samples_p = torch.randn(
        1000, 2
    )  # 100 samples from a 2D normal distribution (mean=0, std=1)
    samples_q = (
        torch.randn(1000, 2) + 1
    )  # 100 samples from a shifted normal (mean=1, std=1)

    # Compute Wasserstein distance
    w_dist = wasserstein_distance_from_samples(samples_p, samples_q)
    print("Wasserstein Distance (samples_p, samples_q):", w_dist)

    w_dist = wasserstein_distance_from_samples(samples_p, samples_p)
    print("Wasserstein Distance (samples_p, samples_p):", w_dist)


if __name__ == "__main__":
    main()
