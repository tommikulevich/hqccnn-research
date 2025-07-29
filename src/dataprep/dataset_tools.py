"""Data loading utilities."""
import inspect
import os
from typing import Tuple, Optional
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from config.schema import DataConfig


def count_dataloader_images_per_class(loader: DataLoader):
    """Count number of samples per target class in a DataLoader."""
    if loader is None:
        return {}

    cnt = Counter()
    for batch in loader:
        labels = batch[1]
        if isinstance(labels, torch.Tensor):
            labels = labels.view(-1).tolist()
        else:
            labels = list(labels)
        cnt.update(labels)

    return dict(cnt)


def get_dataloaders(cfg: DataConfig, dataset_cls: VisionDataset,
                    seed: Optional[int] = None) \
                    -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Return training, validation, and optional test dataloaders."""
    data_dir = cfg.params['data_dir']
    batch_size = cfg.params['batch_size']
    num_workers = cfg.params.get('num_workers', 0)

    # Split ratios
    split_cfg = cfg.params.get('split', {})
    train_ratio = split_cfg.get('train', 1.0)
    val_ratio = split_cfg.get('val', 0.0)
    test_ratio = split_cfg.get('test', 0.0)
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    # Build transforms
    def _build_transform(transform_list):
        if not transform_list:
            return transforms.ToTensor()

        tfs = []
        for entry in transform_list:
            name = entry.get('name')
            params = entry.get('params', {})
            if not hasattr(transforms, name):
                raise ValueError(f"Unknown transform: {name}")
            cls = getattr(transforms, name)
            tfs.append(cls(**params))

        return transforms.Compose(tfs)

    transforms_cfg = cfg.params.get('transforms', {})
    train_transform = _build_transform(transforms_cfg.get('train'))
    val_transform = _build_transform(transforms_cfg.get('val'))
    test_transform = _build_transform(transforms_cfg.get('test'))

    # Load datasets
    if type(dataset_cls) is ImageFolder:
        train_folder = os.path.join(data_dir, 'train')
        val_folder = os.path.join(data_dir, 'val')
        test_folder = os.path.join(data_dir, 'test')

        train_ds = dataset_cls(train_folder, transform=train_transform)
        val_ds = dataset_cls(val_folder, transform=val_transform)

        test_ds = None
        if os.path.isdir(test_folder):
            test_ds = dataset_cls(test_folder, transform=test_transform)
    else:
        init_sig = inspect.signature(dataset_cls.__init__)
        if 'train' in init_sig.parameters:
            ds_train_base = dataset_cls(root=data_dir, train=True,
                                        transform=None, download=True)
            ds_test_base = dataset_cls(root=data_dir, train=False,
                                       transform=None, download=True)
            base_ds = ConcatDataset([ds_train_base, ds_test_base])
        elif 'split' in init_sig.parameters:
            ds_train_base = dataset_cls(root=data_dir, split='train',
                                        transform=None, download=True)
            ds_test_base = dataset_cls(root=data_dir, split='test',
                                       transform=None, download=True)
            base_ds = ConcatDataset([ds_train_base, ds_test_base])
        else:
            base_ds = dataset_cls(root=data_dir, transform=None, download=True)

        labels = np.array([int(base_ds[i][1]) for i in range(len(base_ds))])

        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
        )
        train_idx, rest_idx = next(sss1.split(np.zeros(len(labels)), labels))

        val_idx, test_idx = [], []
        if val_ratio > 0 and test_ratio > 0:
            sss2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=seed,
            )

            sub_labels = labels[rest_idx]
            idx_a, idx_b = next(sss2.split(np.zeros(len(sub_labels)),
                                           sub_labels))

            val_idx = [rest_idx[i] for i in idx_a]
            test_idx = [rest_idx[i] for i in idx_b]
        elif val_ratio > 0:
            val_idx = rest_idx
        else:
            test_idx = rest_idx

        def make_concat(transform, download_flag=False):
            if 'train' in init_sig.parameters:
                return ConcatDataset([
                    dataset_cls(root=data_dir, train=True,
                                transform=transform, download=download_flag),
                    dataset_cls(root=data_dir, train=False,
                                transform=transform, download=download_flag)
                ])
            elif 'split' in init_sig.parameters:
                return ConcatDataset([
                    dataset_cls(root=data_dir, split='train',
                                transform=transform, download=download_flag),
                    dataset_cls(root=data_dir, split='test',
                                transform=transform, download=download_flag)
                ])
            else:
                return dataset_cls(root=data_dir, transform=transform,
                                   download=download_flag)

        full_train_ds = make_concat(train_transform, download_flag=True)
        train_ds = Subset(full_train_ds, train_idx)

        full_val_ds = make_concat(val_transform, download_flag=False)
        val_ds = Subset(full_val_ds,   val_idx)

        test_ds = None
        if test_idx:
            full_test_ds = make_concat(test_transform, download_flag=False)
            test_ds = Subset(full_test_ds, test_idx)

    # Build dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = None
    if 'test_ds' in locals() and test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
