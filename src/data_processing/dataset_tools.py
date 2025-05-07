"""Data loading utilities."""
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from config.config import DataConfig
from config.enums import DatasetName


def get_dataset(dataset_name: str, data_dir: str):
    """Get training and validation datasets based on dataset name."""
    if dataset_name == DatasetName.MNIST:
        train_ds = datasets.MNIST(data_dir, train=True, download=True,
                                  transform=transforms.ToTensor())
        val_ds = datasets.MNIST(data_dir, train=False, download=True,
                                transform=transforms.ToTensor())
    elif dataset_name == DatasetName.FASHION_MNIST:
        train_ds = datasets.FashionMNIST(data_dir, train=True, download=True,
                                         transform=transforms.ToTensor())
        val_ds = datasets.FashionMNIST(data_dir, train=False, download=True,
                                       transform=transforms.ToTensor())
    elif dataset_name == DatasetName.CIFAR10:
        train_ds = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=transforms.ToTensor())
        val_ds = datasets.CIFAR10(data_dir, train=False, download=True,
                                  transform=transforms.ToTensor())
    elif dataset_name == DatasetName.CIFAR100:
        train_ds = datasets.CIFAR100(data_dir, train=True, download=True,
                                     transform=transforms.ToTensor())
        val_ds = datasets.CIFAR100(data_dir, train=False, download=True,
                                   transform=transforms.ToTensor())
    elif dataset_name == DatasetName.EUROSAT:
        train_ds = datasets.EuroSAT(data_dir, train=True, download=True,
                                    transform=transforms.ToTensor())
        val_ds = datasets.EuroSAT(data_dir, train=False, download=True,
                                  transform=transforms.ToTensor())
    elif dataset_name == DatasetName.CUSTOM:
        # TODO: consider keeping transforms in config
        train_ds = ImageFolder(f"{data_dir}/train",
                               transform=transforms.ToTensor())
        val_ds = ImageFolder(f"{data_dir}/val",
                             transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds


def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Return training and validation dataloaders using registry pattern."""

    data_dir = cfg.params['data_dir']
    batch_size = cfg.params['batch_size']
    num_workers = cfg.params['num_workers']

    train_ds, val_ds = get_dataset(cfg.name, data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
