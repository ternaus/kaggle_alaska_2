from typing import Union

import numpy as np
from sklearn import metrics


def alaska_weighted_auc(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    if len(set(y_true)) == 1:
        return 0

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)

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
