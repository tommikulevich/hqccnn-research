"""Script to train HQNN_Parallel models."""
from dataprep.dataset_tools import get_dataloaders
from trainer.trainer import Trainer
from utils.seeds import set_seeds
from config.schema import Config
from config.enums import SchedulerName
from config.registry import (
    MODEL_REGISTRY, DATASET_REGISTRY, SEARCH_REGISTRY,
    LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY,
)


def run_train(config: Config, resume_from: str = None,
              dry_run: bool = False) -> None:
    """Train model according to config and optional checkpoint resume."""
    # Set seeds if provided
    if config.seed is not None:
        set_seeds(config.seed)

    # Data loaders via registry
    dataset_cls = DATASET_REGISTRY.get(config.data.name)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {config.data.name}")

    loaders = get_dataloaders(config.data, dataset_cls, config.seed)
    if len(loaders) == 2:
        train_loader, val_loader = loaders
        test_loader = None
    elif len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        raise ValueError(f"Dataset loader returned {len(loaders)} loaders, \
            expected 2 or 3.")

    # Model instantiation via registry
    ModelClass = MODEL_REGISTRY.get(config.model.name)
    if ModelClass is None:
        raise ValueError(f"Unknown model: {config.model.name}")

    # Determine in_channels from sample batch
    sample_data, _ = next(iter(train_loader))
    if ModelClass is None:
        raise ValueError(f"Unknown model: {config.model.name}")
    if config.data.params['input_shape'][0] != sample_data.shape[1]:
        raise ValueError(f"Invalid provided input shape: num of channels \
            {config.data.params['input_shape'][0]} != {sample_data.shape[1]}")
    model = ModelClass(
        in_channels=config.data.params['input_shape'][0],
        num_classes=config.data.params['num_classes'],
        **config.model.params,
    )

    def run_experiment(cfg: Config) -> None:
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
                          train_loader, val_loader, test_loader=test_loader,
                          resume_from=resume_from,
                          dry_run=dry_run)

        # Sanity check
        # batch = next(iter(train_loader))
        # sample = batch[0] if isinstance(batch, (list, tuple)) else batch
        # err = trainer.sanity_check(sample)
        # if err:
        #     raise err

        # Run training + validation + test
        trainer.run()

    # Hyperparameter search (or simple run)
    search_fn = SEARCH_REGISTRY.get(config.search.method)
    if search_fn is None:
        raise ValueError(f"Unknown search method: {config.search.method}")
    search_fn(config, run_experiment)
