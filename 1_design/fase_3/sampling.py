"""Which test images get explained (numpy only, deterministic given the seed).

  - Quantitative sample (E07): N images, stratified by the class PREDICTED by the DenseNet, N/10
    per class; the explained class is the predicted one.
  - Gallery (E08): GALLERY_SIZE images for the human eye:
        one correct image per class (DenseNet right, drawn at random, not picked for beauty)
      + the most confident DenseNet errors
      + two images from each of the four quadrants (both right, only DenseNet right, only LR
        right, both wrong), never repeating an image.
    For a DenseNet error both the predicted and the true class are explained.

Nothing shrinks silently: if the data cannot supply what is asked, a ValueError says why.
"""
from __future__ import annotations

import numpy as np

from metrics import quadrant_indices

QUADRANT_ORDER = ("both_correct", "only_densenet", "only_lr", "both_wrong")


def explanation_targets(true_label, predicted) -> list:
    """The classes to explain for one image: the predicted one, plus the true one if different."""
    p, t = int(predicted), int(true_label)
    return [p] if p == t else [p, t]


def select_quant_sample(pred, n: int, num_classes: int, seed: int) -> np.ndarray:
    """n positions in the test set, n/num_classes for each PREDICTED class, ordered by class."""
    if n % num_classes:
        raise ValueError(f"n={n} must be a multiple of the number of classes ({num_classes})")
    per_class = n // num_classes
    pred = np.asarray(pred)
    rng = np.random.RandomState(seed)
    picks = []
    for c in range(num_classes):
        pool = np.where(pred == c)[0]
        if len(pool) < per_class:
            raise ValueError(f"only {len(pool)} images are predicted as class {c}, {per_class} needed")
        picks.append(np.sort(rng.choice(pool, per_class, replace=False)))
    return np.concatenate(picks)


def select_gallery(y_true, pred_densenet, pred_lr, conf_densenet, num_classes: int, seed: int,
                   gallery_size: int, n_errors: int = 6, per_quadrant: int = 2) -> list:
    """List of dicts {index, role, true, pred, confidence, targets}, `gallery_size` long."""
    expected = num_classes + n_errors + per_quadrant * len(QUADRANT_ORDER)
    if gallery_size != expected:
        raise ValueError(f"GALLERY_SIZE={gallery_size} but the composition ({num_classes} correct + "
                         f"{n_errors} errors + {per_quadrant} x {len(QUADRANT_ORDER)} quadrants) is {expected}")
    y, pd, pl, cf = (np.asarray(a) for a in (y_true, pred_densenet, pred_lr, conf_densenet))
    rng = np.random.RandomState(seed)
    chosen, used = [], set()

    def take(index, role):
        index = int(index)
        used.add(index)
        chosen.append({"index": index, "role": role, "true": int(y[index]), "pred": int(pd[index]),
                       "confidence": float(cf[index]), "targets": explanation_targets(y[index], pd[index])})

    for c in range(num_classes):                              # 1. one correct image per class
        pool = np.where((y == c) & (pd == c))[0]
        if len(pool) == 0:
            raise ValueError(f"no image of class {c} is correctly classified by the DenseNet")
        take(rng.choice(pool), "correct")

    errors = np.where(pd != y)[0]                             # 2. the most confident errors
    errors = errors[np.lexsort((errors, -cf[errors]))]        # confidence descending, then index
    if len(errors) < n_errors:
        raise ValueError(f"only {len(errors)} DenseNet errors, {n_errors} needed")
    for i in errors[:n_errors]:
        take(i, "confident_error")

    quadrants = quadrant_indices(y, pd, pl)                   # 3. two per quadrant, no repeats
    for name in QUADRANT_ORDER:
        pool = [int(i) for i in quadrants[name] if int(i) not in used]
        if len(pool) < per_quadrant:
            raise ValueError(f"quadrant {name!r} has only {len(pool)} unused images, {per_quadrant} needed")
        for i in sorted(rng.choice(pool, per_quadrant, replace=False)):
            take(i, f"quadrant:{name}")
    return chosen
