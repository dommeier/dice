import numpy as np
import torch


# Very simple data generation process. We sample K "archetypes" in d dimensions, and then the data generating process is
# sampling K from uniform, and then sampling from the corresponding archetype
def generate_multidimensional_data(num_samples, dimensions, archetypes, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Generate K archetypes in d dimensions
    archetypes = np.array(archetypes)

    # Sample K from uniform
    K = np.random.randint(0, archetypes.shape[0], num_samples)

    # Sample from the corresponding archetype
    data = np.array([archetypes[k] for k in K])
    # data = np.array([archetypes[k] for k in K])

    return (
        torch.from_numpy(data).to(torch.float32),
        torch.from_numpy(K),
        {
            "archetypes": archetypes,
            "dimensions": dimensions,
            "num_samples": num_samples,
            "label_encoder": {i: i for i in range(archetypes.shape[0])},
        },
    )
