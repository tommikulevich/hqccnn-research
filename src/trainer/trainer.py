"""Trainer module for training and evaluating models."""
import os
import zipfile
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, confusion_matrix

from config.schema import Config
from utils.common import flatten_dict
from utils.logger import setup_logging, get_mlflow_writer, CSVLogger


@dataclass(frozen=True)
class Metrics:
    """Performance metrics container."""
    loss: float
    accuracy: float

    precision: float
    precision_weighted: float

    recall: float
    recall_weighted: float

    f1: float
    f1_weighted: float


class Trainer:
    """Handles training, validation, checkpointing, and logging."""
    def __init__(self, config: Config,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 resume_from: Optional[str] = None,
                 dry_run: bool = False) -> None:
        self.config = config
        self.model = model.to(config.training.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dry_run = dry_run

        self.device = config.training.device

        # Logging
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        dashboard_dir = Path(config.logging.dashboard_dir).absolute()
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = get_mlflow_writer(
            self.config.model.name.value, dashboard_dir.as_uri(), ts)
        self.writer.add_params(flatten_dict(asdict(config)))

        self.log_dir = Path(config.logging.log_dir, ts)
        self.logger, self.log_file = setup_logging(ts, self.log_dir)

        self.train_csv = CSVLogger(self.log_dir / f'train_{ts}.csv')
        self.val_csv = CSVLogger(self.log_dir / f'val_{ts}.csv')

        # Loss function, optimizer and scheduler
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Useful log info
        self.logger.info(f"<<< RUN {ts} >>>")
        self.logger.info("> Model: \n%s", model)
        self.logger.info("> Total trainable parameters: %s",
                         sum(p.numel() for p in model.parameters()
                             if p.requires_grad))
        self.logger.info("> Parameters by top-level module: ")
        for name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            self.logger.info(f"  {name}: {n}")
        self.logger.info("> Parameters by individual tensor: ")
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.logger.info(f"  {name}: {param.numel()}")
        self.logger.info("> Loss function: \n%s", self.loss_fn)
        self.logger.info("> Optimizer: \n%s", self.optimizer)
        self.logger.info("> Scheduler: \n%s", self.scheduler)
        self.writer.add_text(f"{model}", f'model_{ts}.txt')

        # Checkpoint dir
        self.checkpoint_dir = Path(self.config.logging.checkpoint_dir, ts)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.start_epoch = 1

        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)

        # Save code and config snapshot
        self._save_code_snapshot()

    def _save_code_snapshot(self) -> None:
        """Save code and config snapshot to ZIP."""
        code_zip = self.log_dir / f'code_config_{self.log_dir.name}.zip'
        with zipfile.ZipFile(str(code_zip), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in Path('src').rglob('*.py'):
                zipf.write(file, arcname=str(file))
            for file in Path('configs').rglob('*.yaml'):
                zipf.write(file, arcname=str(file))
        self.writer.add_artifact(str(code_zip))
        self.logger.info(f'Code and config snapshot saved: {code_zip}')

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model, optimizer, scheduler states and epoch info."""
        path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pt'
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state':
                self.scheduler.state_dict() if self.scheduler else None,
            'config': asdict(self.config),
        }
        torch.save(state, path)
        self.writer.add_artifact(str(path))
        self.logger.info(f'Checkpoint saved: {path}')

    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint and restore states."""
        chk = torch.load(path, map_location=self.config.training.device,
                         weights_only=False)
        self.model.load_state_dict(chk['model_state'])
        self.optimizer.load_state_dict(chk['optimizer_state'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(chk['scheduler_state'])
        self.start_epoch = chk.get('epoch', 0) + 1
        self.logger.info(f"Resuming from checkpoint: {path}, "
                         f"epoch {self.start_epoch}")

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        """Compute accuracy, precision, recall, and F1 scores."""
        return {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds,
                                         average='macro',
                                         zero_division=0),
            'precision_weighted': precision_score(targets, preds,
                                                  average='weighted',
                                                  zero_division=0),
            'recall': recall_score(targets, preds,
                                   average='macro',
                                   zero_division=0),
            'recall_weighted': recall_score(targets, preds,
                                            average='weighted',
                                            zero_division=0),
            'f1': f1_score(targets, preds,
                           average='macro',
                           zero_division=0),
            'f1_weighted': f1_score(targets, preds,
                                    average='weighted',
                                    zero_division=0),
        }

    def sanity_check(self, sample: torch.Tensor) -> Optional[Exception]:
        sample = sample.to(self.device)
        try:
            with torch.no_grad():
                self.model(sample)
            return None
        except Exception as e:
            return e

    def train_epoch(self, epoch: int) -> dict:
        """Run one epoch of training."""
        self.model.train()

        losses = []
        all_preds, all_targets = [], []
        for batch_idx, (data, target) in enumerate(
                tqdm(self.train_loader, desc=f'Train Epoch {epoch}')):
            if self.dry_run:
                if batch_idx >= self.config.data.params['dry_run_batches']:
                    break

            data = data.to(self.config.training.device)
            target = target.to(self.config.training.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(target.cpu().numpy().tolist())

        stats = self._compute_metrics(np.array(all_preds),
                                      np.array(all_targets))

        return Metrics(loss=float(np.mean(losses)), **stats)

    def validate(self, epoch: int) -> dict:
        """Run validation."""
        self.model.eval()

        losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            loader = self.val_loader
            for batch_idx, (data, target) in enumerate(
                    tqdm(loader, desc=f'Val Epoch {epoch}')):
                data = data.to(self.config.training.device)
                target = target.to(self.config.training.device)

                output = self.model(data)

                loss = self.loss_fn(output, target)
                losses.append(loss.item())

                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(target.cpu().numpy().tolist())

        stats = self._compute_metrics(np.array(all_preds),
                                      np.array(all_targets))
        return Metrics(loss=float(np.mean(losses)), **stats)

    def test(self) -> tuple[Metrics, Any]:
        """Run testing and compute metrics and confusion matrix."""
        if self.test_loader is None:
            self.logger.warning("No test loader provided.")
            return

        self.model.eval()

        losses, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Test'):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.loss_fn(output, target)
                losses.append(loss.item())

                preds = output.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy().tolist())

        stats = self._compute_metrics(np.array(all_preds),
                                      np.array(all_targets))
        cm = confusion_matrix(all_targets, all_preds)

        return Metrics(loss=float(np.mean(losses)), **stats), cm

    def evaluate_random_sample(self) -> None:
        """Evaluate a single random sample from test set."""
        if self.test_loader is None:
            self.logger.warning("No test loader for sample evaluation.")
            return

        idx = np.random.randint(len(self.test_loader.dataset))
        sample, target = self.test_loader.dataset[idx]

        self.model.eval()
        with torch.no_grad():
            input_tensor = sample.to(self.device).unsqueeze(0)
            output = self.model(input_tensor)
            pred = output.argmax(dim=1).item()

        self.logger.info(f"Random sample eval - target: {target}, \
            prediction: {pred}")

    def run(self) -> None:
        """Execute full training and validation cycles."""
        self.logger.info(f'Configuration: {self.config}')
        for epoch in range(self.start_epoch, self.config.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            for k, v in asdict(train_metrics).items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            self.train_csv.log({'epoch': epoch, **asdict(train_metrics)})

            val_metrics = self.validate(epoch)
            for k, v in asdict(val_metrics).items():
                self.writer.add_scalar(f'val/{k}', v, epoch)
            self.val_csv.log({'epoch': epoch, **asdict(val_metrics)})

            self.logger.info(f"Epoch {epoch} | "
                             f"Train: {asdict(train_metrics)} | "
                             f"Val: {asdict(val_metrics)}")

            if self.scheduler is not None:
                self.scheduler.step()
                self.logger.info("Scheduler next LR = ",
                                 {self.scheduler.get_last_lr()})

            if epoch % self.config.logging.save_interval == 0:
                self._save_checkpoint(epoch)

            if self.dry_run:
                break

        if self.test_loader is not None and not self.dry_run:
            test_metrics, cm = self.test()
            for k, v in asdict(test_metrics).items():
                self.writer.add_scalar(f'test/{k}', v,
                                       self.config.training.epochs)
            self.logger.info(f"Test metrics: {asdict(test_metrics)}")
            cm_path = Path(self.log_dir) / 'confusion_matrix.npy'
            np.save(cm_path, cm)
            self.writer.add_artifact(str(cm_path))
            self.writer.add_text(np.array2string(cm),
                                 text_path='confusion_matrix.txt')
            self.evaluate_random_sample()

        self.writer.add_artifact(str(self.log_file))
        self.writer.add_artifact(str(self.train_csv.csv_path))
        self.writer.add_artifact(str(self.val_csv.csv_path))

        self.writer.close()
        self.train_csv.close()
        self.val_csv.close()
