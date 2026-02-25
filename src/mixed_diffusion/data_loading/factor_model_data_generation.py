import numpy as np
import torch
import json


# Data generating function
# We generate x as a noise low_rank factor matrix, following x = XY, where X is a GMM and Y is a Rademacher [-1,+1]
def generate_factor_model_data(num_samples, X, Y, seed=None, **kwargs):

    if seed is not None:
        np.random.seed(seed)

    X = np.array(X)
    Y = np.array(Y)

    print(f"X.shape {X.shape}, Y.shape {Y.shape}")

    # Sample K from uniform
    K = np.random.randint(0, X.shape[0], num_samples)

    # Sample from the corresponding archetype
    X_archetypes = np.array([X[k] for k in K])

    # noise the data to make it a GMM
    noised_X = X_archetypes + np.random.normal(0, 1, X_archetypes.shape)

    print(f"noised_X.shape {noised_X.shape}")

    # Multiply it by Y
    data = noised_X @ Y

    return (
        torch.from_numpy(data).to(torch.float32),
        torch.from_numpy(K),
        {
            "X": X,
            "Y": Y,
            "num_samples": num_samples,
            "label_encoder": {i: i for i in range(X.shape[0])},
        },
    )


if __name__ == "__main__":
    # Generating X and Y
    np.random.seed(42)

    low_rank_dimension = 15
    n_archetypes = 10
    dimensions_to_map_to = 200

    X = np.random.normal(0, 1, (n_archetypes, low_rank_dimension))

    # Y transforms from low_rank_dimension to dimensions_to_map_to using rademacher +-1
    Y = (
        torch.randint(0, 2, (low_rank_dimension, dimensions_to_map_to)).float() * 2.0
        - 1.0
    )

    # Store X and Y in a config file
    config = {
        "X": X.tolist(),
        "Y": Y.tolist(),
        "low_rank_dimension": low_rank_dimension,
        "n_archetypes": n_archetypes,
        "dimensions_to_map_to": dimensions_to_map_to,
    }

    with open("factor_model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Saved X and Y to factor_model_config.json")
