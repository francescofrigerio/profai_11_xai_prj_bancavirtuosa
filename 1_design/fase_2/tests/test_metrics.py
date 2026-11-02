"""metrics.py: known cases, worked out by hand."""
import numpy as np
import pytest

from metrics import (accuracy, confusion, macro_f1, quadrant_indices, quadrants, seconds_per_image)


def test_accuracy_f1_and_confusion_on_a_hand_case():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    assert abs(accuracy(y_true, y_pred) - 4 / 6) < 1e-12
    cm = confusion(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # per-class F1: class0 = 2*(1/2)(1/2)/(1) = .5, class1 = 2*(2/3)(1)/(5/3) = .8, class2 = 2*(1)(.5)/(1.5) = 2/3
    assert abs(macro_f1(y_true, y_pred, 3) - (0.5 + 0.8 + 2 / 3) / 3) < 1e-9


def test_quadrants_partition_the_images():
    y = np.array([0, 1, 2, 3, 4, 5])
    d = np.array([0, 1, 9, 9, 4, 5])        # DenseNet wrong on images 2, 3
    l = np.array([0, 9, 2, 9, 9, 5])        # LR wrong on images 1, 3, 4
    q = quadrants(y, d, l)
    assert q == {"both_correct": 2, "only_densenet": 2, "only_lr": 1, "both_wrong": 1}
    assert sum(q.values()) == len(y)
    idx = quadrant_indices(y, d, l)
    assert {k: len(v) for k, v in idx.items()} == q
    assert sorted(np.concatenate(list(idx.values())).tolist()) == list(range(6))


def test_seconds_per_image_counts_the_synchronization():
    calls = []
    t = seconds_per_image(lambda: sum(range(10000)), 10, sync=lambda: calls.append(1))
    assert t > 0 and len(calls) == 2
    with pytest.raises(ValueError):
        seconds_per_image(lambda: None, 0)
