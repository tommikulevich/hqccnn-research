"""Configuration loader and dataclasses."""
from dataclasses import dataclass
from typing import Any, Dict

from config.enums import (
    DatasetName, ModelName, OptimizerName,
    LossName, SchedulerName, SearchMethod
)


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and checkpoint configuration."""
    log_dir: str
    checkpoint_dir: str
    dashboard_dir: str
    save_interval: int


@dataclass(frozen=True)
class DataConfig:
    """Dataset configuration."""
    name: DatasetName
    params: Dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""
    name: ModelName
    params: Dict[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    """Training parameters."""
    epochs: int
    device: str


@dataclass(frozen=True)
class LossConfig:
    """Loss function configuration."""
    name: LossName
    params: Dict[str, Any]


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration."""
    name: OptimizerName
    params: Dict[str, Any]


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler configuration."""
    name: SchedulerName
    params: Dict[str, Any]


@dataclass(frozen=True)
class SearchConfig:
    """Hyperparameter search configuration."""
    method: SearchMethod
    params: Dict[str, Any]


@dataclass(frozen=True)
class Config:
    """Top-level experiment configuration."""
    version: str
    seed: int
    logging: LoggingConfig

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    search: SearchConfig
