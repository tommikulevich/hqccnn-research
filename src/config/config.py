"""Configuration loader and dataclasses."""
import yaml
from dataclasses import dataclass
from pathlib import Path
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


def load_config(path: str) -> Config:
    """Load YAML configuration into Config dataclass."""
    config_path = Path(path)
    with config_path.open('r') as f:
        cfg = yaml.safe_load(f)

    log_cfg = LoggingConfig(**cfg['logging'])
    data_cfg = DataConfig(
        name=DatasetName(cfg['data']['name']),
        params=cfg['data']['params'],
    )
    model_cfg = ModelConfig(
        name=ModelName(cfg['model']['name']),
        params=cfg['model']['params'],
    )
    train_cfg = TrainingConfig(
        epochs=cfg['training']['epochs'],
        device=cfg['training']['device'],
    )
    loss_cfg = LossConfig(
        name=LossName(cfg['loss']['name']),
        params=cfg['loss']['params'],
    )
    optimizer_cfg = OptimizerConfig(
        name=OptimizerName(cfg['optimizer']['name']),
        params=cfg['optimizer']['params'],
    )
    scheduler_cfg = SchedulerConfig(
        name=SchedulerName(cfg['scheduler']['name']),
        params=cfg['scheduler']['params'],
    )
    search_cfg = SearchConfig(
        method=SearchMethod(cfg['search']['method']),
        params=cfg['search']['params'],
    )

    return Config(
        version=cfg.get('version', 'UNDEFINED'),
        seed=cfg.get('seed'),
        data=data_cfg,
        model=model_cfg,
        loss=loss_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        training=train_cfg,
        logging=log_cfg,
        search=search_cfg,
    )
