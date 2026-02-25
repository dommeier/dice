from torchvision.utils import save_image
import torchvision
from PIL import Image
import numpy as np
import os
import torch

from matplotlib import pyplot as plt
import seaborn as sns


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def img_on_grid(img, axes, row, col, title=None):
    ax = axes[row, col]
    img = tensor_to_img(img)
    img = img.squeeze()
    img = img.permute(1, 2, 0)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, pad=10)


def tensor_to_img(tensor, normalize=True):
    if normalize:
        # print(f"before tensor stats: min: {tensor.min()}, max: {tensor.max()}, mean {tensor.mean()}")
        tensor = (tensor.clamp(-1, 1) + 1) / 2
        # print(f"tensor stats: min: {tensor.min()}, max: {tensor.max()}, mean {tensor.mean()}")
        tensor = (tensor * 255).type(torch.uint8)
    tensor = tensor.cpu().detach()

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor


def save_3d_tensor(tensor, output_prefix="denoised_image", normalize=True):
    tensor = tensor_to_img(tensor, normalize)
    for d in range(tensor.shape[0]):
        save_image(tensor[d], f"{output_prefix}_{d}.png")


def save_images(images, path, **kwargs):
    # grid = torchvision.utils.make_grid(images, **kwargs)
    # ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # print(ndarr.shape)
    # im = Image.fromarray(ndarr)
    # im.save(path)

    grid = torchvision.utils.make_grid(images, **kwargs)
    # Permute, move to CPU, convert to numpy array, scale, and change dtype
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    # ndarr = (ndarr * 255).astype(np.uint8)  # Scale and convert to uint8
    im = Image.fromarray(ndarr)
    im.save(path)


def save_tabular_tensor(tensor, output_prefix="denoised_tabular"):
    tensor = tensor.view(-1, tensor.shape[-1]).cpu().detach()
    plt.figure()
    plt.scatter(tensor[:, 0], tensor[:, 1])
    plt.savefig(f"{output_prefix}.png")
    plt.close()


def save_heatmap(tensor, output_prefix="denoised_heatmap"):
    """
    Save a 2D tensor as a heatmap image.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(tensor.cpu().numpy(), cmap="viridis")
    plt.title("Heatmap")
    plt.savefig(f"{output_prefix}.png")
    plt.close()


def save_data(data, output_prefix):
    """
    Save data dynamically based on its dimensions.
    - For image data (3D tensors), save as images.
    - For tabular data (2D tensors), save as 2d plot.
    """
    if data.ndim == 4:  # Image data (e.g., [batch, channels, height, width])
        save_3d_tensor(data, output_prefix)
    elif data.ndim == 3:  # Tabular data (e.g., [batch, features])
        save_tabular_tensor(data, output_prefix)
    elif data.ndim == 2:  # 2D data (e.g., [batch, features])
        save_heatmap(data, output_prefix)
    else:
        raise ValueError(
            f"Unsupported data dimensions for saving. data.ndim: {data.ndim}, data.shape: {data.shape}"
        )


def plot_loss_curve(
    losses, title="Training Loss", path="training_loss.png", show=False
):
    """
    Plot the training loss curve

    Args:
        losses: List of loss values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.savefig(path)
    if show:
        plt.show()
