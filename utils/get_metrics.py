from typing import Optional, Sequence

from sklearn.metrics import balanced_accuracy_score
import torch
import numpy as np


def get_iou_obj(pc_preds: torch.LongTensor, targets: torch.LongTensor, label: int = 1) -> float:
    """
    Calculates the Intersection over Union (IoU) for a specific label between predicted and target point clouds 
    using PyTorch tensors.

    The value is a fraction in [0, 1] (``get_iou_np`` returns a percentage). It is symmetric in
    its first two arguments. Used for the per-batch IoU logged during training.

    Args:
        pc_preds (torch.LongTensor): Predicted point cloud labels. Should be a PyTorch tensor of long integers.
        targets (torch.LongTensor): Ground truth point cloud labels. Should be a PyTorch tensor of long integers.
        label (int, optional): The label for which the IoU is calculated. Defaults to 1.

    Returns:
        float: IoU value for the specified label. Returns NaN if division by zero occurs,
        i.e. when the label is absent from both predictions and targets.
    """
    # get IoU
    corrects = pc_preds == targets
    gt_positive = torch.sum(targets == label).item()
    detected_positive = (pc_preds == label).sum().item()
    tp = (corrects & (pc_preds == label)).sum().item()
    fp = detected_positive - tp

    # Handle division by zero
    if gt_positive + fp == 0:
        iou_obj = torch.nan
    else:
        iou_obj = tp / (gt_positive + fp)
    return iou_obj


def get_iou_np(pc_preds: np.ndarray, targets: np.ndarray, label: int = 1) -> float:
    """
    Calculates the Intersection over Union (IoU) for a specific label between predicted and target point clouds.

    The value is a percentage in (0, 100]. Used by the evaluation scripts for the per-tile IoU
    written to the results CSV. An IoU of exactly 0 is reported as NaN, so it is left out of
    means computed with ``np.nanmean``.

    Args:
        pc_preds (numpy.ndarray): Predicted point cloud labels. Should be a NumPy array.
        targets (numpy.ndarray): Ground truth point cloud labels. Should be a NumPy array.
        label (int, optional): The label for which the IoU is calculated. Defaults to 1.

    Returns:
        float: IoU value for the specified label. Returns NaN if division by zero occurs or IoU is 0.
    """
    # get IoU
    corrects = pc_preds == targets
    gt_positive = np.sum(targets == label)
    detected_positive = np.count_nonzero(pc_preds == label)
    tp = np.count_nonzero(corrects & (pc_preds == label))
    fp = detected_positive - tp

    # Handle division by zero
    if gt_positive + fp == 0:
        iou_obj = np.nan
    else:
        iou_obj = tp / (gt_positive + fp)

    if iou_obj == 0.:
        iou_obj = np.nan

    return iou_obj*100


def get_accuracy(preds: torch.Tensor, targets: torch.Tensor, metrics: dict) -> dict:
    """
    Store the fraction of correctly classified points in ``metrics['accuracy']`` (0 if there are
    no points) and return ``metrics``.
    """
    corrects = torch.eq(preds.view(-1), targets.view(-1))
    if len(corrects) != 0:
        metrics['accuracy'] = (corrects.sum().item() / len(corrects))
    else:
        metrics['accuracy'] = 0
    return metrics


def get_weights_effective_num_of_samples(n_of_classes: int, beta: float,
                                         samples_per_cls: Sequence[float]) -> np.ndarray:
    """
    Class weights from the effective number of samples, (1 - beta) / (1 - beta^n), normalised
    to sum to 1.

    The authors suggest experimenting with different beta values: 0.9, 0.99, 0.999, 0.9999.
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights4class = (1.0 - beta) / np.array(effective_num)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_inverse_num_of_samples(n_of_classes: int, samples_per_cls: Sequence[float],
                                       power: float = 1.0) -> np.ndarray:
    """Class weights proportional to 1 / n^power, normalised to sum to 1."""
    weights4class = 1.0 / np.array(np.power(samples_per_cls, power))  # [0.03724195 0.00244003]
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_sklearn(n_of_classes: int, samples_per_cls: Sequence[float]) -> np.ndarray:
    """Class weights total / (n_classes * n), as scikit-learn's 'balanced' mode, normalised to sum to 1."""
    weights4class = np.sum(samples_per_cls) / np.multiply(n_of_classes, samples_per_cls)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights4class(weighing_method: str, n_classes: int, samples_per_cls: Sequence[float],
                      beta: Optional[float] = None) -> Optional[torch.Tensor]:
    """
       Class weights computed from the number of samples per class.

       :param weighing_method: str, options available: "EFS" (effective number of samples),
           "INS" (inverse number of samples), "ISNS" (inverse square root of the number of
           samples), "sklearn"
       :param n_classes: int, representing the total number of classes in the entire train set
       :param samples_per_cls: A python list of size [n_classes]
       :param beta: float, only used by "EFS"

       :return weights4class: float torch.Tensor of size [n_classes], or None for an unknown method
    """
    if weighing_method == 'EFS':
        weights4class = get_weights_effective_num_of_samples(n_classes, beta, samples_per_cls)
    elif weighing_method == 'INS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls)
    elif weighing_method == 'ISNS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls, 0.5)  # [0.9385, 0.0615]
    elif weighing_method == 'sklearn':
        weights4class = get_weights_sklearn(n_classes, samples_per_cls)
    else:
        return None

    weights4class = torch.tensor(weights4class).float()
    return weights4class
