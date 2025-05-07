"""Script to train HQNN_Parallel models."""
import argparse

from trainers.trainer import Trainer
from utils.seeds import set_seeds
from config.config import load_config, Config
from config.enums import SchedulerName
from config.registry import (
    MODEL_REGISTRY, DATASET_REGISTRY, SEARCH_REGISTRY,
    LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Environment for hybrid model training')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to YAML config')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='Path to checkpoint to resume')
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run')
    args = parser.parse_args()

    # Load config
    config: Config = load_config(args.config)

    # Set seeds
    set_seeds(config.seed)

    # Data loaders via registry
    loader_fn = DATASET_REGISTRY.get(config.data.name)
    if loader_fn is None:
        raise ValueError(f"Unknown dataset: {config.data.name}")
    train_loader, val_loader = loader_fn(config.data)

    # Model instantiation via registry
    model_name = config.model.name
    ModelClass = MODEL_REGISTRY.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Initialize model with params
    sample_data, _ = next(iter(train_loader))
    in_channels = sample_data.shape[1]
    model = ModelClass(
        in_channels=in_channels,
        num_classes=config.data.params['num_classes'],
        **config.model.params,
    )

    def run_experiment(cfg):
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
                          train_loader, val_loader,
                          resume_from=args.resume,
                          dry_run=args.dry_run)

        # Sanity check model before start training
        batch = next(iter(train_loader))
        sample = batch[0] if isinstance(batch, (list, tuple)) else batch
        model_check = trainer.sanity_check(sample)
        if model_check:
            raise model_check

        trainer.run()

    search_fn = SEARCH_REGISTRY.get(config.search.method)
    if search_fn is None:
        raise ValueError(f"Unknown search method: {config.search.method}")
    search_fn(config, run_experiment)


if __name__ == '__main__':
    main()
