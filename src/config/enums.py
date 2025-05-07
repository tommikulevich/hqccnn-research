"""Enums for models, optimizers, schedulers, searches, datasets etc."""
from enum import Enum


class DatasetName(str, Enum):
    MNIST = "mnist"
    IMAGEFOLDER = "imagefolder"


class ModelName(str, Enum):
    HQNN_PARALLEL = "hqnn-parallel"


class LossName(str, Enum):
    CROSS_ENTROPY = "cross-entropy"


class OptimizerName(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class SchedulerName(str, Enum):
    NONE = "none"


class SearchMethod(str, Enum):
    NONE = "none"
