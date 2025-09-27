"""Script to test HQNN_Parallel models."""
from dataprep.dataset_tools import get_dataloaders
from trainer.trainer import Trainer
from utils.seeds import set_seeds
from config.schema import Config
from config.enums import SchedulerName
from config.registry import (
    MODEL_REGISTRY, DATASET_REGISTRY,
    LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY,
)


def run_test_custom(cfg: Config, checkpoint_path: str, output_dir: str,
                    labels: list[str], extra: str = "") -> None:
    """Test model according to config and checkpoint path."""

    # Set seeds if provided
    if cfg.seed is not None:
        set_seeds(cfg.seed)

    # Data loaders via registry
    dataset_cls = DATASET_REGISTRY.get(cfg.data.name)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")

    loaders = get_dataloaders(cfg.data, dataset_cls, cfg.seed)
    if len(loaders) == 2:
        train_loader, val_loader = loaders
        test_loader = None
    elif len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        raise ValueError(f"Dataset loader returned {len(loaders)} loaders,\
            expected 2 or 3.")

    # Model instantiation via registry
    ModelClass = MODEL_REGISTRY.get(cfg.model.name)
    if ModelClass is None:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    # Determine in_channels from sample batch
    # TODO: fix input_shape (include resize)
    sample_data, _ = next(iter(train_loader))
    if ModelClass is None:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    if cfg.data.params['input_shape'][0] != sample_data.shape[1]:
        raise ValueError(f"Invalid provided input shape: num of channels \
            {cfg.data.params['input_shape'][0]} \
                != {sample_data.shape[1]}")
    model = ModelClass(
        in_channels=cfg.data.params['input_shape'][0],
        num_classes=cfg.data.params['num_classes'],
        input_size=cfg.data.params['input_shape'],
        **cfg.model.params,
    )

    # Build loss function via registry
    loss_cfg = cfg.loss
    loss_cls = LOSS_REGISTRY.get(loss_cfg.name)
    if loss_cls is None:
        raise ValueError(f"Unknown loss function: {loss_cfg.name}")
    loss_params = {} if loss_cfg.params in (None, "None", "none") \
        else loss_cfg.params
    loss_fn = loss_cls(**loss_params)

    # Build optimizer via registry
    opt_cfg = cfg.optimizer
    optim_cls = OPTIMIZER_REGISTRY.get(opt_cfg.name)
    if optim_cls is None:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
    optim_params = {} if opt_cfg.params in (None, "None", "none") \
        else opt_cfg.params
    optimizer = optim_cls(model.parameters(), **optim_params)

    # Build scheduler via registry
    sched_cfg = cfg.scheduler
    if sched_cfg.name != SchedulerName.NONE:
        sched_cls = SCHEDULER_REGISTRY.get(sched_cfg.name)
        if sched_cls is None:
            raise ValueError(f"Unknown scheduler: {sched_cfg.name}")
        sched_params = {} if sched_cfg.params in (None, "None", "none") \
            else sched_cfg.params
        scheduler = sched_cls(optimizer, **sched_params)
    else:
        scheduler = None

    trainer = Trainer(cfg, model, loss_fn, optimizer, scheduler,
                      train_loader, val_loader, test_loader,
                      resume_from=checkpoint_path)

    # Run custom test
    trainer.test_custom(output_dir, labels, extra=extra)
