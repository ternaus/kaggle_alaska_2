import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import apex
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict

# from torch import nn
from iglovikov_helper_functions.config_parsing.utils import object_from_dict

# from sklearn.metrics import log_loss
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from pytorch_lightning.logging import NeptuneLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# from collections import defaultdict
from kaggle_alaska_2.dataloader import Alaska2Dataset
from kaggle_alaska_2.utils import get_samples, folder2label


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

    def forward(self, batch: Dict) -> torch.Tensor:  # skipcq: PYL-W0221
        return self.model(batch)

    def prepare_data(self):
        self.train_samples = []  # skipcq: PYL-W0201
        self.val_samples = []  # skipcq: PYL-W0201

        kf = KFold(n_splits=self.hparams["num_folds"], random_state=self.hparams["seed"], shuffle=True)

        for folder in sorted(list(folder2label.keys())):
            samples = np.array(get_samples(Path(self.hparams["data_path"]) / folder))

            for fold_id, (train_index, val_index) in enumerate(kf.split(samples)):
                if fold_id != self.hparams["fold_id"]:
                    continue

                self.train_samples += samples[train_index].tolist()
                self.val_samples += samples[val_index].tolist()

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])

        result = DataLoader(
            Alaska2Dataset(self.train_samples, train_aug),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        return DataLoader(
            Alaska2Dataset(self.val_samples, val_aug),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"], params=filter(lambda x: x.requires_grad, self.model.parameters()),
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]  # skipcq: PYL-W0201

        return self.optimizers, [scheduler]

    # skipcq: PYL-W0613, PYL-W0221
    def training_step(self, batch, batch_idx):
        features = batch["features"]
        logits = self.forward(features)

        total_loss = self.loss(logits, batch["targets"])

        logs = {"train_loss": total_loss, "lr": self._get_current_lr()}

        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()

    # skipcq: PYL-W0613, PYL-W0221
    # def validation_step(self, batch, batch_idx, dataset_idx):
    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        targets = batch["targets"]

        logits = self.forward(features)
        # probs = nn.Softmax(dim=1)(logits)[:, 1]
        #
        # return {str(idx2name[dataset_idx]): probs, "targets": targets}

        return {"val_loss": self.loss(logits, targets)}

    def validation_epoch_end(self, outputs: List) -> Dict[str, Any]:
        # result: defaultdict = defaultdict(dict)

        # for column in idx2name.values():
        #     result[column] = {"probs": [], "targets": []}
        #
        # for output in outputs:
        #     for o in output:
        # targets = o["targets"].cpu().numpy().tolist()
        # targets = o["targets"]
        # del o["targets"]
        # probs = list(o.values())[0].cpu().numpy().tolist()
        # key = list(o.keys())[0]
        #
        # result[key]["targets"] += targets
        # result[key]["probs"] += probs

        # probs_list: List[float] = []
        # target_list: List[float] = []
        #
        # # losses = {}
        # #
        # for key, value in result.items():
        #     probs_list += value["probs"]
        #     target_list += value["targets"]

        #     lg = cross_entropy(probs_list, target_list)
        #     losses[f"log_loss_{key}"] = lg

        # score = alaska_weighted_auc(target_list, probs_list)

        # print(target_list)
        # print(probs_list)

        # score = alaska_weighted_auc(target_list, probs_list)

        # score = alaska_weighted_auc(target_list, probs_list)
        # logs = {"val_score": score,
        #         "val_log_loss": log_loss(target_list, probs_list)
        #         }
        # # logs = {**logs, **losses}

        # print(outputs)
        score = find_average(outputs, "val_loss")
        logs = {"val_loss": score}

        return {"val_loss": score, "log": logs}


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name="ternaus/kagglealaska2",
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
