"""Label mappings and metrics, checked on hand-built inputs."""
import math

import pytest
import torch
import numpy as np

from src.datasets import CAT3Dataset, DalesDataset
from utils.get_metrics import get_accuracy, get_iou_np, get_iou_obj


def _window(classes):
    """Minimal point cloud with the class code in column 3."""
    pc = np.zeros((len(classes), 12), dtype=np.float32)
    pc[:, 3] = classes
    return pc


def test_cat3_labels_three_classes():
    ds = CAT3Dataset(number_of_points=8, files=[], use_windturbine=False)
    # ground, low/med/high vegetation, building, power line, tower, other tower
    classes = [2, 3, 4, 5, 6, 14, 15, 18]
    labels = ds.get_labels_segmen(_window(classes), is_prod=False, use_windturbine=False, default_class=0)
    assert labels.tolist() == [0, 0, 0, 0, 0, 2, 1, 0]


def test_cat3_labels_with_wind_turbine():
    ds = CAT3Dataset(number_of_points=8, files=[], use_windturbine=True)
    classes = [2, 5, 14, 15, 18, 19, 29]
    labels = ds.get_labels_segmen(_window(classes), is_prod=False, use_windturbine=True, default_class=0)
    assert labels.tolist() == [0, 0, 2, 1, 3, 3, 3]


def test_dales_labels():
    raw = np.arange(9)  # 0 undefined, 1 ground, 2 veg, 3 cars, 4 trucks, 5 lines, 6 fences, 7 poles, 8 buildings
    assert DalesDataset.get_labels(raw).tolist() == [0, 1, 0, 0, 0, 3, 0, 2, 0]
    assert DalesDataset.get_all_labels(raw).tolist() == [-1, 0, 3, 5, 5, 2, -1, 1, 4]


def test_iou():
    targets = np.array([0, 0, 0, 1, 1, 2])
    preds = np.array([0, 0, 1, 1, 1, 0])
    assert get_iou_np(preds, targets, 0) == pytest.approx(50.0)
    assert get_iou_np(preds, targets, 1) == pytest.approx(200 / 3)
    # a class that is present but never predicted correctly is reported as NaN, not 0
    assert math.isnan(get_iou_np(preds, targets, 2))
    # a class that is absent from both arrays is NaN
    assert math.isnan(get_iou_np(preds, targets, 3))

    t, p = torch.from_numpy(targets), torch.from_numpy(preds)
    assert get_iou_obj(p, t, 0) == pytest.approx(0.5)
    assert get_iou_obj(p, t, 1) == pytest.approx(2 / 3)
    assert get_iou_obj(p, t, 2) == 0
    assert math.isnan(get_iou_obj(p, t, 3))


def test_accuracy():
    metrics = get_accuracy(torch.tensor([0, 1, 1, 2]), torch.tensor([0, 1, 2, 2]), {})
    assert metrics['accuracy'] == 0.75
