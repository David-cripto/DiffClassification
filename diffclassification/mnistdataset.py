from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader

PathLike = Path | str

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def get_dataset(img_dir: PathLike) -> tuple[Dataset, ...]:
    train_dataset = torchvision.datasets.MNIST(
        img_dir, download=True, train=True, transform=TRANSFORM
    )
    test_dataset = torchvision.datasets.MNIST(
        img_dir, download=True, train=True, transform=TRANSFORM
    )
    return train_dataset, test_dataset

def get_train_data_loader(path, batch_size=64, shuffle=True):
    train_dataset, test_dataset = get_dataset(path)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )