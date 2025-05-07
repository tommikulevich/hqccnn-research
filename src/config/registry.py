"""Registries for models, optimizers, schedulers, searches, datasets etc."""
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss

from models.hqnn_parallel import HQNN_Parallel
from data_processing.dataset_tools import get_dataloaders
from config.enums import (DatasetName, ModelName, LossName,
                          OptimizerName, SearchMethod)


DATASET_REGISTRY = {
    DatasetName.MNIST: get_dataloaders,
    DatasetName.FASHION_MNIST: get_dataloaders,
    DatasetName.CIFAR10: get_dataloaders,
    DatasetName.CIFAR100: get_dataloaders,
    DatasetName.EUROSAT: get_dataloaders,
    DatasetName.CUSTOM: get_dataloaders,
}

MODEL_REGISTRY = {
    ModelName.HQNN_PARALLEL: HQNN_Parallel,
}

LOSS_REGISTRY = {
    LossName.CROSS_ENTROPY: CrossEntropyLoss,
}


OPTIMIZER_REGISTRY = {
    OptimizerName.ADAM: Adam,
    OptimizerName.SGD: SGD,
}


SCHEDULER_REGISTRY = {
}


SEARCH_REGISTRY = {
    SearchMethod.NONE: lambda cfg, fn: fn(cfg),
}
