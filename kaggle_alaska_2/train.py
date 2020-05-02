import argparse
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import apex
import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict
from pytorch_lightning.logging import NeptuneLogger
from torch.utils.data import DataLoader

from kaggle_alaska_2.dataloader import Alaska2Dataset

from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from kaggle_alaska_2.utils import get_samples, folder2label, idx2name
from sklearn.model_selection import KFold


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-r", "--resume", type=Path, help="Path to the checkpoint.")
    return parser.parse_args()


class Alaska2(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = object_from_dict(self.hparams["model"])

        if hparams["sync_bn"]:
            self.model = apex.parallel.convert_syncbn_model(self.model)

        self.loss = object_from_dict(self.hparams["loss"])

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.model(batch)

    def prepare_data(self):
        self.train_samples = []
        self.val_samples = []

        kf = KFold(n_splits=self.hparams["num_folds"], random_state=self.hparams["seed"], shuffle=True)

        for folder in sorted(list(folder2label.keys())):
            samples = np.array(get_samples(Path(self.hparams["data_path"]) / folder))

            for fold_id, (train_index, val_index) in enumerate(kf.split(samples)):
                if fold_id != self.hparams["fold_id"]:
                    continue

                self.train_samples += samples[train_index].tolist()
                self.val_samples += [samples[val_index].tolist()]

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])

        result = DataLoader(
            Alaska2Dataset(self.train_samples[:1000], train_aug),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        result = []

        for sample in self.val_samples:
            result += [
                DataLoader(
                    Alaska2Dataset(sample[:1000], val_aug),
                    batch_size=self.hparams["val_parameters"]["batch_size"],
                    num_workers=self.hparams["num_workers"],
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )
            ]

        return result

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"], params=filter(lambda x: x.requires_grad, self.model.parameters())
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        logits = self.forward(features)

        total_loss = self.loss(logits, batch["targets"])

        logs = {"train_loss": total_loss, "lr": self._get_current_lr()}

        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_idx, dataset_idx):
        features = batch["features"]
        targets = batch["targets"]

        logits = self.forward(features)

        total_loss = self.loss(logits, targets)

        return {f"val_loss_{idx2name[dataset_idx]}": total_loss}

    def validation_epoch_end(self, outputs: List) -> Dict[str, Any]:
        logs = {}
        val_loss = 0

        for output in outputs:
            for name, values in output[0].items():
                loss = values.mean()
                val_loss += loss

                logs[name] = loss

        return {"val_loss": val_loss, "log": logs}


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name=f"ternaus/kagglealaska2",
        experiment_name=f"{hparams['experiment_name']}",  # Optional,
        tags=["pytorch-lightning", "mlp"],  # Optional,
        upload_source_files=[],
    )

    pipeline = Alaska2(hparams)

    Path(hparams["checkpoint_callback"]["filepath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"], logger=logger, checkpoint_callback=object_from_dict(hparams["checkpoint_callback"])
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
