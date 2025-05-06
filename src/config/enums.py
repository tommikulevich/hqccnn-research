"""Enums for models, optimizers, schedulers, searches, datasets etc."""
from enum import Enum


class DatasetName(str, Enum):
    MNIST = "mnist"
    IMAGEFOLDER = "imagefolder"


class ModelName(str, Enum):
    HQNNPARALLEL = "HQNN_Parallel"


class LossName(str, Enum):
    CROSSENTROPY = "CrossEntropy"


class OptimizerName(str, Enum):
    ADAM = "Adam"
    SGD = "SGD"


class SchedulerName(str, Enum):
    NONE = "none"


class SearchMethod(str, Enum):
    NONE = "none"
