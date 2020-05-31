"""
Logger, that has an output consistent with DVC plots

https://dvc.org/doc/command-reference/plots


Ex:
    epoch, AUC, loss
    34, 0.91935, 0.0317345
    35, 0.91913, 0.0317829
    36, 0.92256, 0.0304632
    37, 0.92302, 0.0299015
"""

import csv

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path


class CsvLogger(LightningLoggerBase):
    def __init__(self, train_csv_path, val_csv_path, train_columns, val_columns):
        super().__init__()

        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path

        Path(self.train_csv_path.parent).mkdir(exist_ok=True, parents=True)
        Path(self.val_csv_path.parent).mkdir(exist_ok=True, parents=True)

        self.train_columns = train_columns
        self.val_columns = val_columns

        with open(self.train_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + train_columns)

        with open(self.val_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + val_columns)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if "train_loss" in metrics:
            fields = [step] + [metrics[x] for x in self.train_columns]
            filename = self.train_csv_path
        else:
            fields = [metrics["epoch"]] + [metrics[x] for x in self.val_columns]
            filename = self.val_csv_path
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    @property
    def experiment(self):
        pass

    def log_hyperparams(self, hparams):
        pass

    @property
    def version(self):
        pass

    @property
    def name(self):
        return "CSVLogger"
