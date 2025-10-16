"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms

import params


def get_mnist(train=True, transform=None, batch_size=None):
    """Get MNIST dataset loader."""

    # Default preprocessing (expand to RGB if you want 3-channel input)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.Grayscale(num_output_channels=1),  
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

    if batch_size is None:
        batch_size = params.batch_size

    mnist_dataset = datasets.MNIST(
        root=params.data_root,
        train=train,
        transform=transform,
        download=True
    )

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return mnist_data_loader
