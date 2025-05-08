"""Configuration loader and dataclasses."""
import yaml
from pathlib import Path

from config.schema import (
    Config, DataConfig, LoggingConfig, LossConfig, ModelConfig,
    OptimizerConfig, SchedulerConfig, SearchConfig, TrainingConfig)
from config.enums import (
    DatasetName, ModelName, OptimizerName,
    LossName, SchedulerName, SearchMethod
)


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
