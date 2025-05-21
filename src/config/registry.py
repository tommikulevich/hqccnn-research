"""Registries for models, optimizers, schedulers, searches, datasets etc."""
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torchvision import datasets

from models.hqnn_parallel.hqnn_parallel import HQNN_Parallel
from models.hqnn_parallel.cnn import HQNN_Parallel_Classic_CNN
from models.hqnn_quanv.hqnn_quanv import HQNN_Quanv
from models.hqnn_quanv.cnn import HQNN_Quanv_Classic_CNN
from models.hcqtcnn_resnet34.hcqtcnn_resnet34 import HCQTCNN_ResNet34
from search.grid import grid_search
from config.enums import (DatasetName, ModelName, LossName,
                          OptimizerName, SearchMethod)


DATASET_REGISTRY = {
    DatasetName.MNIST: datasets.MNIST,
    DatasetName.FASHION_MNIST: datasets.FashionMNIST,
    DatasetName.CIFAR10: datasets.CIFAR10,
    DatasetName.CIFAR100: datasets.CIFAR100,
    DatasetName.EUROSAT: datasets.EuroSAT,
    DatasetName.CUSTOM: datasets.ImageFolder,
}

MODEL_REGISTRY = {
    ModelName.HQNN_PARALLEL: HQNN_Parallel,
    ModelName.HQNN_PARALLEL_CLASSIC_CNN: HQNN_Parallel_Classic_CNN,
    ModelName.HQNN_QUANV: HQNN_Quanv,
    ModelName.HQNN_QUANV_CLASSIC_CNN: HQNN_Quanv_Classic_CNN,
    ModelName.HCQTCNN_RESNET34: HCQTCNN_ResNet34
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
    SearchMethod.GRID: grid_search,
}
