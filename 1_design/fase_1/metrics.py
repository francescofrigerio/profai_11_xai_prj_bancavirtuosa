"""Classification metrics of the two models and their comparison (numpy only)."""
from __future__ import annotations

import time

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro",
                          zero_division=0))


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def quadrants(y_true, pred_densenet, pred_lr) -> dict:
    """Counts of the four cases of the comparison; they always sum to the number of images."""
    y = np.asarray(y_true)
    d_ok = np.asarray(pred_densenet) == y
    l_ok = np.asarray(pred_lr) == y
    return {
        "both_correct": int((d_ok & l_ok).sum()),
        "only_densenet": int((d_ok & ~l_ok).sum()),
        "only_lr": int((~d_ok & l_ok).sum()),
        "both_wrong": int((~d_ok & ~l_ok).sum()),
    }


def quadrant_indices(y_true, pred_densenet, pred_lr) -> dict:
    """The same four cases, as arrays of positions (used to pick the gallery images)."""
    y = np.asarray(y_true)
    d_ok = np.asarray(pred_densenet) == y
    l_ok = np.asarray(pred_lr) == y
    return {
        "both_correct": np.where(d_ok & l_ok)[0],
        "only_densenet": np.where(d_ok & ~l_ok)[0],
        "only_lr": np.where(~d_ok & l_ok)[0],
        "both_wrong": np.where(~d_ok & ~l_ok)[0],
    }


def seconds_per_image(fn, n_images: int, sync=None) -> float:
    """Wall time of fn() divided by the images it processes. `sync` (e.g. cuda synchronize)
    is called before and after, so GPU work is really counted."""
    if n_images <= 0:
        raise ValueError("n_images must be > 0")
    if sync:
        sync()
    t0 = time.perf_counter()
    fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / n_images
