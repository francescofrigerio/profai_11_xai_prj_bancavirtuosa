"""
    Function to calc metrics for models
"""

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np

def accuracy(y_true, y_pred) -> float:
    """
        accuracy function
    """
    return accuracy_score(y_true, y_pred)

def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """
        macro_f1 function
    """
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro",
                          zero_division=0))


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    """
        Rows = true class, columns = predicted class.
    """

    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def quadrants(y_true, pred_densenet, pred_lr) -> dict:
    """
        Counts of the four cases of the comparison; they always sum to the number of images.
    """
    y = np.asarray(y_true)
    d_ok = np.asarray(pred_densenet) == y
    l_ok = np.asarray(pred_lr) == y
    return {
        "both_correct": int((d_ok & l_ok).sum()),
        "only_densenet": int((d_ok & ~l_ok).sum()),
        "only_lr": int((~d_ok & l_ok).sum()),
        "both_wrong": int((~d_ok & ~l_ok).sum()),
    }
