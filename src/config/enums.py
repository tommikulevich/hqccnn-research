"""Enums for models, optimizers, schedulers, searches, datasets etc."""
from enum import Enum


class DatasetName(str, Enum):
    MNIST = "mnist"
    FASHION_MNIST = "fashion-mnist"
    CIFAR10 = "cifar-10"
    CIFAR100 = "cifar-100"
    EUROSAT = "eurosat"
    SVHN = "svhn"
    CUSTOM = "custom"


class ModelName(str, Enum):
    HQNN_PARALLEL = "hqnn-parallel"
    HQNN_PARALLEL_CLASSIC_CNN = "hqnn-parallel-classic-cnn"
    HQNN_QUANV = "hqnn-quanv"
    HQNN_QUANV_CLASSIC_CNN = "hqnn-quanv-classic-cnn"


class LossName(str, Enum):
    CROSS_ENTROPY = "cross-entropy"


class OptimizerName(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class SchedulerName(str, Enum):
    NONE = "none"
    STEPLR = "steplr"
    EXPOTENTIALLR = "exponentiallr"


class SearchMethod(str, Enum):
    NONE = "none"
    GRID = "grid"
