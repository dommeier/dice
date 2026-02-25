import argparse
from matplotlib.path import Path
import numpy as np
from matplotlib.textpath import TextPath
from sklearn.model_selection import train_test_split
import torch


def generate_circle_data(num_samples=5000, noise_level=0.2):
    # Generate points on a circle with some noise
    theta = torch.rand(num_samples) * 2 * np.pi
    radius = 1.0
    noise = noise_level * torch.randn(num_samples, 2)

    x = radius * torch.cos(theta) + noise[:, 0]
    y = radius * torch.sin(theta) + noise[:, 1]

    return torch.stack([x, y], dim=1)


def generate_letters_data(num_samples=50000, noise_level=0.2, sample_size=32):
    samples = []
    letters = "AEFHIKLMNVWXZ"
    for letter in letters:
        segments = path_to_segments(get_letter_path(letter))
        samples.extend(
            sample_from_segments(
                normalize_segments(np.array(segments)),
                (num_samples // len(letters)) * sample_size,
            )
        )

    return np.random.normal(samples, noise_level)


def get_letter_path(letter, font_size=100):
    return TextPath((0, 0), letter, size=font_size)


def normalize_segments(segments):
    # Flatten segments into vertices for easier computations
    vertices = segments.reshape(-1, 2)

    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    mean_x, mean_y = vertices[:, 0].mean(), vertices[:, 1].mean()

    if abs(max_x - min_x) < 1e-9:
        scale = 2.0
    else:
        scale = 2.0 / (max_x - min_x)

    # Normalize directly on segments array
    normalized_segments = np.zeros_like(segments)
    normalized_segments[:, :, 0] = (segments[:, :, 0] - mean_x) * scale
    normalized_segments[:, :, 1] = (segments[:, :, 1] - mean_y) * scale

    return normalized_segments


def path_to_segments(textpath):
    vertices = textpath.vertices
    codes = textpath.codes

    vertices = textpath.vertices
    codes = textpath.codes

    segments = [
        [vertices[i - 1], vertices[i]]
        for i in range(1, len(vertices))
        if codes[i] == Path.LINETO
        and not np.array_equal(vertices[i - 1], vertices[i])
        and not np.isnan(vertices[i - 1]).any()
    ]

    segments = [
        segment for segment in segments if segment[0][0] > 0.1 and segment[1][0] > 0.1
    ]

    return np.array(segments)


def sample_from_segments(segments, num_samples):
    # segments: array-like of shape (num_segments, 2, 2)
    edge_lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    cumulative_lengths = np.cumsum(edge_lengths)
    perimeter = cumulative_lengths[-1]

    sample_points = []
    for _ in range(num_samples):
        rand_dist = np.random.rand() * perimeter
        edge_idx = np.searchsorted(cumulative_lengths, rand_dist)

        local_dist = (
            rand_dist if edge_idx == 0 else rand_dist - cumulative_lengths[edge_idx - 1]
        )

        A, B = segments[edge_idx]
        t = local_dist / edge_lengths[edge_idx]
        point = A + t * (B - A)
        sample_points.append(point)

    return np.array(sample_points)


def generate_outlines_data(
    num_samples=5000, data_type="circle", noise_level=0.2, sample_size=512
) -> np.ndarray:
    """
    Generate synthetic 2D data for testing

    Args:
        n_samples: Number of samples to generate
        data_type: "circle", "spiral", "moons", or "gaussian_mixture"

    Returns:
        np.ndarray: Generated data of shape (n_samples, dimension)
    """
    if data_type == "circle":
        samples = generate_circle_data(num_samples * sample_size, noise_level)
        data = samples.view(-1, sample_size, 2)

    elif data_type == "ngons":
        for n in range(3, 8):
            segments = np.array(
                [
                    [
                        [
                            np.cos(2 * np.pi * i / n),
                            np.sin(2 * np.pi * i / n),
                        ],
                        [
                            np.cos(2 * np.pi * (i + 1) / n),
                            np.sin(2 * np.pi * (i + 1) / n),
                        ],
                    ]
                    for i in range(n)
                ]
            )
            samples = sample_from_segments(segments, num_samples * sample_size)
            if n == 3:
                all_samples = samples
            else:
                all_samples = np.concatenate([all_samples, samples])
        data = torch.from_numpy(all_samples).view(-1, sample_size, 2)

    elif data_type == "letters":
        data = torch.from_numpy(
            generate_letters_data(num_samples, noise_level, sample_size=sample_size)
        ).view(-1, sample_size, 2)

    elif len(data_type) > 7 and data_type[0:7] == "letter_":
        all_samples = []
        chars = data_type[7:]
        for char in chars:
            path = get_letter_path(char)
            samples = sample_from_segments(
                normalize_segments(path_to_segments(path)),
                num_samples * sample_size,
            )
            all_samples.append(samples)

        data = torch.from_numpy(np.random.normal(all_samples, noise_level)).view(
            -1, sample_size, 2
        )

    elif data_type == "gaussian_mixture":
        # Generate a mixture of 4 Gaussians
        means = [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]]

        std = 0.3

        points_per_cluster = num_samples * sample_size // 4
        data_list = []

        for mean in means:
            cluster = torch.randn(points_per_cluster, 2) * std
            cluster[:, 0] += mean[0]
            cluster[:, 1] += mean[1]
            data_list.append(cluster)

        x = torch.cat(data_list)
        data = x.view(-1, sample_size, 2)

    elif data_type == "1_d_gaussian_mixture":
        # Generate a mixture of 5 Gaussians
        means = [-2, -1, 0, 1, 2]

        std = 0.3

        points_per_cluster = num_samples // 5
        data_list = []

        for mean in means:
            cluster = torch.randn(points_per_cluster, 1) * std
            cluster[:, 0] += mean
            data_list.append(cluster)

        x = torch.cat(data_list)
        y = torch.zeros(x.shape[0])  # Dummy labels
        # Stack coordinates and create dummy labels
        data = torch.stack([x, y], dim=1)
        data = data.view(-1, sample_size, 1)

    elif data_type == "shapes":
        # Generate a mixture of circle, letters and different n-gons
        circle = generate_outlines_data(
            num_samples // 3, "circle", noise_level, sample_size=sample_size
        )
        letters = generate_outlines_data(
            num_samples // 3, "letters", noise_level, sample_size=sample_size
        )
        ngons = generate_outlines_data(
            num_samples // 3, "ngons", noise_level, sample_size=sample_size
        )
        data = torch.from_numpy(np.concatenate([letters, circle, ngons]))
    else:
        raise ValueError("Unkown data type")
    dummy_labels = torch.zeros(data.shape)

    return data.to(torch.float32), dummy_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--data_type", type=str, default="ngons")
    argsparse.add_argument("--num_samples", type=int, default=5000)
    argsparse.add_argument("--noise_level", type=float, default=0.0)
    args = argsparse.parse_args()

    data = generate_outlines_data(args.num_samples, args.data_type, args.noise_level)

    if data.shape[1] == 2:
        plt.scatter(data[:, 0], data[:, 1])
        plt.axis("equal")
        plt.show()
