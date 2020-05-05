import argparse
from pathlib import Path
import yaml
from kaggle_alaska_2.train import Alaska2
from kaggle_alaska_2.dataloader import AlaskaTest2Dataset, Alaska2Dataset
from torch.utils.data import DataLoader
from albumentations.core.serialization import from_dict
from torch import nn
from typing import Dict
from kaggle_alaska_2.utils import load_checkpoint
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from kaggle_alaska_2.utils import get_samples, folder2label
from pytorch_toolbelt.inference import tta
from sklearn.metrics import log_loss
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-w", "--checkpoint_path", type=Path, help="Path to weights.", required=True)
    arg("-o", "--output_path", type=Path, help="Output_path.", required=True)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    model = Alaska2(hparams=hparams)

    # test_file_names = sorted(Path(hparams.test_image_path).glob("*.jpg"))
    #
    # test_aug = from_dict(hparams["train_aug"])
    #
    # dataloader = DataLoader(
    #     AlaskaTest2Dataset(test_file_names, test_aug),
    #     batch_size=hparams.test_parameters.batch_size,
    #     num_workers=hparams.num_workers,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    corrections: Dict[str, str] = {}

    checkpoint = load_checkpoint(file_path=args.checkpoint_path, rename_in_layers=corrections)  # type: ignore

    model.load_state_dict(checkpoint["state_dict"])

    model = nn.Sequential(model, nn.Softmax())

    model.eval()
    model = model.half()
    model.cuda()

    with torch.no_grad():
        print("Evaluate on validation.")
        val_aug = from_dict(hparams["val_aug"])
        val_samples = []

        kf = KFold(n_splits=hparams["num_folds"], random_state=hparams["seed"], shuffle=True)

        for folder in sorted(list(folder2label.keys())):
            samples = np.array(get_samples(Path(hparams["test_data_path"]) / folder))

            for fold_id, (_, val_index) in enumerate(kf.split(samples)):
                if fold_id != hparams["fold_id"]:
                    continue

                val_samples += samples[val_index].tolist()

        dataloader = DataLoader(
            Alaska2Dataset(val_samples, val_aug),
            batch_size=hparams["val_parameters"]["batch_size"],
            num_workers=hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        y_true = []
        y_pred = []

        for batch in tqdm(dataloader):
            features = batch["features"]
            targets = batch["targets"]

            preds = model(features.half().cuda())

            y_pred += preds[:, 1].cpu().numpy().tolist()
            y_true += targets.cpu().numpy().tolist()

        print("Val log loss = ", log_loss(y_true, y_pred))

        print("Evaluate on test.")
        test_aug = from_dict(hparams["val_aug"])

        test_file_names = sorted((Path(hparams["test_data_path"]) / "Test").glob("*.jpg"))

        dataloader = DataLoader(
            AlaskaTest2Dataset(test_file_names, test_aug),
            batch_size=hparams["test_parameters"]["batch_size"],
            num_workers=hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        y_pred = []
        file_ids = []

        for batch in tqdm(dataloader):
            file_ids += batch["image_id"]
            features = batch["features"]

            if hparams["test_parameters"]["tta"] == "d4":
                preds = tta.d4_image2label(model, features.half().cuda())
            else:
                preds = model(features.half().cuda())

            y_pred += preds[:, 1].cpu().numpy().tolist()

        submission = pd.DataFrame({"Id": file_ids, "Label": y_pred})

        submission["Id"] = submission["Id"] + ".jpg"

        submission.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
