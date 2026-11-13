"""
    Function to calc metrics for models
"""
import numpy as np

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

def create_quadrant_lists( cfg,
                         logger, 
                          y_true, 
                          y_pred_lr, 
                          y_pred_cnn, 
                          idx,
                          classes, 
                          out_path, 
                          n_per_class=1 ):
    """
        Save quadrants on json
        Quadrants: both_ok, lr_ko_cnn_ok, lr_ok_cnn_ko, both_ko (same split)
    """
    
    y_true, y_pred_lr, y_pred_cnn, idx = map(np.asarray, (y_true, y_pred_lr, y_pred_cnn, idx))
    
    lr_ok, cnn_ok = y_pred_lr == y_true, y_pred_cnn == y_true

    groups = { "both_ok": lr_ok & cnn_ok,
               "lr_ko_cnn_ok": ~lr_ok & cnn_ok,
               "lr_ok_cnn_ko": lr_ok & ~cnn_ok,
               "both_ko": ~lr_ok & ~cnn_ok }

    # random but reproducible, not the "first" ones
    rng = np.random.default_rng(cfg.SEED)       
    for name, mask in groups.items():
        items = []
        for c in classes:
            cand = np.where(mask & (y_true == c))[0]
            pick = np.sort(rng.choice(cand, 
                                      size=min(len(cand), n_per_class), 
                                      replace=False)) if len(cand) else []
            
            items += [{"class": int(c), 
                       "id_dataset": int(idx[p]), 
                       "y_true": int(y_true[p]),
                       "y_pred": int(y_pred_cnn[p]),          # compat. con load_images_by_class
                       "y_pred_lr": int(y_pred_lr[p])
                       } for p in pick]
        save_dict_to_json(out_path / f"images_{name}.json", items)
        logger.debug("%s: %d candidati, %d salvati", name, int(mask.sum()), len(items))
    
