"""Logging and monitoring utilities."""
import os
import csv
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import mlflow
from torch.utils.tensorboard import SummaryWriter


class CSVLogger:
    """Logger to write metrics to a CSV file."""
    def __init__(self, csv_path: str, fnames: Optional[list] = None) -> None:
        self.csv_path = csv_path
        self.fnames = fnames or []

        os.makedirs(Path(csv_path).parent, exist_ok=True)
        self.file = open(csv_path, mode='w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fnames)
        self.writer.writeheader()

    def log(self, data: Dict[str, Any]) -> None:
        """Write a row of metrics to CSV."""
        if not self.fnames:
            self.fnames = list(data.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fnames)
            self.writer.writeheader()
        self.writer.writerow(data)
        self.file.flush()

    def close(self) -> None:
        """Close CSV file."""
        self.file.close()


class MLflowWriter:
    """A thin wrapper around MLflow to mimic SummaryWriter"""
    def __init__(self, experiment_name: str, tracking_uri: str, ts: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.run = mlflow.start_run(run_name=f"run_{ts}",
                                    log_system_metrics=True)

    def add_scalar(self, tag: str, value: float, step: int = None):
        """Log a single scalar metric to MLflow."""
        if step is not None:
            mlflow.log_metric(tag, value, step=step)
        else:
            mlflow.log_metric(tag, value)

    def add_params(self, params: dict):
        """Log a batch of hyperparameters or settings."""
        mlflow.log_params(params)

    def add_artifact(self, local_path: str, artifact_path: str = None):
        """Log an arbitrary file (e.g. a plot, a model checkpoint)."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def add_text(self, text: str, text_path: str = None):
        """Log specific text to file."""
        mlflow.log_text(text, artifact_file=text_path)

    def close(self):
        """End the MLflow run."""
        mlflow.end_run()


def setup_logging(ts: str, log_dir: str) -> Tuple[logging.Logger, str]:
    """Configure root logger to log to console and file with timestamps."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / f'run_{ts}.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(log_path))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger, log_path


def get_summary_writer(ts: str, tb_dir: str) -> SummaryWriter:
    """Get TensorBoard SummaryWriter."""
    os.makedirs(tb_dir, exist_ok=True)
    log_dir = Path(tb_dir) / f'run_{ts}'
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer


def get_mlflow_writer(experiment_name: str, tracking_uri: str,
                      ts: str) -> MLflowWriter:
    """Factory to create an MLflowWriter."""
    return MLflowWriter(experiment_name, tracking_uri, ts)
