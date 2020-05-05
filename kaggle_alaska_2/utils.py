from pathlib import Path
from typing import Union, Dict, List, Tuple
from sklearn import metrics
import numpy as np
import re
from typing import Optional, Any
import torch

folder2label = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 1, "UERD": 1}

idx2name = dict(zip(range(len(folder2label)), sorted(folder2label.keys())))


def get_id2_file_paths(file_path: Union[str, Path]) -> Dict[str, Path]:
    return {file.stem: file for file in Path(file_path).glob("*")}


def get_samples(image_path: Path) -> List[Tuple[Path, int]]:
    label = folder2label[image_path.name]
    result = [(x, label) for x in sorted(image_path.glob("*.jpg"))]
    return result


def alaska_weighted_auc(y_true: Union[np.ndarray, list], y_valid: Union[np.ndarray, list]) -> float:
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    y_true = np.array(y_true)
    y_valid = np.array(y_valid)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        competition_metric += submetric

    return competition_metric / normalization


def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(targets * np.log(predictions))


def load_checkpoint(file_path: Union[Path, str], rename_in_layers: Optional[dict] = None) -> Dict[str, Any]:
    """Loads PyTorch checkpoint, optionally renaming layer names.

    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint["state_dict"]

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)

            result[key] = value

        checkpoint["state_dict"] = result

    return checkpoint
