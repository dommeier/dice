import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def get_ffhq_dataloader(root_dir, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the FFHQ dataset
    
    Args:
        root_dir (string): Path to the directory containing FFHQ images
        batch_size (int): Number of images per batch
        shuffle (bool): Whether to shuffle the dataset
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for FFHQ dataset
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Create dataset
    dataset = FFHQDataset(root_dir=root_dir, transform=transform)

    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=4,  # Adjust based on your system
        pin_memory=True  # Faster data transfer to GPU
    )

    return dataloader



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

class FaceDatasetLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load_labeled_faces_in_the_wild(self, batch_size=32):
        """
        Load Labeled Faces in the Wild (LFW) dataset
        
        Dataset characteristics:
        - Contains ~13,000 images
        - ~1680 people with 2+ images
        - Image sizes vary, so resizing is recommended
        - Good for face recognition tasks
        """
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.LFWPeople(
            root=self.root_dir, 
            split='train',
            transform=transform,
            download=True
        )

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )

    def load_celeba(self, batch_size=32):
        """
        Load CelebA dataset
        
        Dataset characteristics:
        - ~200K celebrity images
        - High-quality face images
        - Multiple attributes available
        - Consistent image quality
        """
        transform = transforms.Compose([
            transforms.CenterCrop(178),  # Recommended center crop for CelebA
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.CelebA(
            root=self.root_dir, 
            split='train',
            transform=transform,
            download=True
        )

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )

    def load_imagenet_faces(self, batch_size=32):
        """
        Load face-related subset from ImageNet
        
        Dataset characteristics:
        - Large variety of face-related images
        - High-resolution images
        - Requires manual download and organization
        """
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Note: This requires manually downloaded ImageNet face-related subset
        dataset = datasets.ImageFolder(
            root=os.path.join(self.root_dir, 'face_subset'),
            transform=transform
        )

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )

def main():
    # Replace with your dataset directory
    dataset_dir = './datasets'
    
    # Create loader
    face_loader = FaceDatasetLoader(dataset_dir)

    # Example: Load LFW dataset
    lfw_dataloader = face_loader.load_labeled_faces_in_the_wild()
    
    # Example: Load CelebA dataset
    celeba_dataloader = face_loader.load_celeba()

    # Print first batch info
    for name, dataloader in [('LFW', lfw_dataloader), ('CelebA', celeba_dataloader)]:
        for batch, labels in dataloader:
            print(f"{name} Dataset:")
            print(f"Batch shape: {batch.shape}")
            print(f"Labels shape: {labels.shape}")
            break

# if __name__ == "__main__":
#     main()




# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual path to FFHQ dataset
#     dataset_path = "/path/to/ffhq/images"
    
#     # Create dataloader
#     ffhq_loader = get_ffhq_dataloader(dataset_path)

#     # Iterate through the dataset
#     for batch in ffhq_loader:
#         # batch is a tensor of shape [batch_size, channels, height, width]
#         print(f"Batch shape: {batch.shape}")
#         break  # Just print the first batch