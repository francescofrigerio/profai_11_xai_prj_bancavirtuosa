"""
    Function to calc metrics for models
"""
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from utility import save_dict_to_json

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

def quadrant_indices(y_true, pred_densenet, pred_lr) -> dict:
    """
        The same four cases, as arrays of positions (used to pick the gallery images).
    """
    y = np.asarray(y_true)
    d_ok = np.asarray(pred_densenet) == y
    l_ok = np.asarray(pred_lr) == y
    return {
        "both_correct": np.where(d_ok & l_ok)[0],
        "only_densenet": np.where(d_ok & ~l_ok)[0],
        "only_lr": np.where(~d_ok & l_ok)[0],
        "both_wrong": np.where(~d_ok & ~l_ok)[0],
    }

def print_final_metrics(out_path , file_name="xai_metrics_all.csv") -> dict:
    """
        Print general final metrics:
        Hot to read the single metrics:

        mask_share and focus_ratio: 
        1- mask_share represents the attribution share within the digit. 
        
        2- focus_ratio divides this by the area of ​​the digit; thus, a value of 1 indicates 
        uniform distribution, while values ​​significantly greater than 1 signify 
        that attention is concentrated on the stroke itself. 
        A CNN that makes an incorrect prediction may still exhibit a high focus value—
        it is simply focusing on the wrong part of the stroke.
        
        3- del_gain: the difference between the drop in probability resulting 
        from random masking and that resulting from map-guided masking. 
        A positive value indicates that the map is more faithful than random chance. 
        A value near 0 suggests that the map does not accurately describe 
        what the model is actually using, even if the visualization itself appears compelling. 
        For signed maps, only the positive component—representing evidence in favor of 
        the class—is considered.
        
        4- agreement: the mean rank correlation between different techniques. 
        A low value for cases where both methods fail (both_ko) combined with a high value 
        for cases where both succeed (both_ok) would be an interesting finding: 
            it would suggest that explanations are consistent when the model is confident, 
            but diverge when the model’s prediction is ambiguous.
        _true grid: for instances categorized as lr_ok_cnn_ko 
        (where the linear model is correct but the CNN is incorrect) 
        and both_ko (where both models are incorrect), this compares the attribution 
        map for the true class against the map for the predicted class.
    """

    df = pd.read_csv(out_path / file_name)
    query_out = (df[df.target == "pred"].groupby(["group", 
                                                  "technique"])
                                    [["mask_share", "focus_ratio", "del_gain", "agreement"]]
                                    .mean().round(3))
    print(query_out)

