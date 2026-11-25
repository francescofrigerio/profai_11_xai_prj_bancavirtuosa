"""Deletion curves (D15): do the pixels a technique calls important really matter to the model?

Pixels inside the region are switched off (set to black) from the most to the least important
according to the map, and the probability of the explained class is followed. A faithful map makes
the probability fall quickly: LOWER area under the curve = better. A random order is the reference.

It only runs when COMPUTE_FAITHFULNESS is True (it costs one batch of DELETION_STEPS+1 forward
passes per image and technique). The model is reached through `predict_proba_fn`, a plain function
(B, 1, 28, 28) float array in [0, 1] -> (B, K) probabilities, so this module needs no torch.
"""
from __future__ import annotations

import numpy as np

from comparison import apply_sign_mode


def deletion_curve(predict_proba_fn, image28, attribution28, mask, target: int, steps: int,
                   order: str = "attribution", rng=None) -> np.ndarray:
    """Probability of `target` after deleting 0, 1/steps, ..., 1 of the pixels of the region."""
    img = np.asarray(image28, dtype=np.float64)
    positions = np.flatnonzero(np.asarray(mask).ravel())
    if order == "attribution":
        scores = np.asarray(attribution28, dtype=np.float64).ravel()[positions]
        ranked = positions[np.argsort(-scores, kind="stable")]
    elif order == "random":
        ranked = (rng or np.random.RandomState(0)).permutation(positions)
    else:
        raise ValueError(f"unknown order {order!r}")
    batch = np.repeat(img.reshape(1, 1, *img.shape), steps + 1, axis=0)
    for s in range(1, steps + 1):
        n_deleted = int(round(len(ranked) * s / steps))
        flat = batch[s, 0].reshape(-1)
        flat[ranked[:n_deleted]] = 0.0
    return np.asarray(predict_proba_fn(batch))[:, target]


def deletion_auc(curve) -> float:
    """Area under the curve over the deleted fraction in [0, 1] (trapezoid rule)."""
    c = np.asarray(curve, dtype=np.float64)
    return float(np.sum((c[1:] + c[:-1]) / 2) / (len(c) - 1))


def deletion_for_techniques(cfg, predict_proba_fn, maps28: dict, images28, targets, masks, seed: int):
    """name -> mean AUC (plus 'random'), or None when COMPUTE_FAITHFULNESS is False."""
    if not cfg.COMPUTE_FAITHFULNESS:
        return None
    n_img = len(masks)
    result = {}
    for name, maps in maps28.items():
        aucs = [deletion_auc(deletion_curve(
            predict_proba_fn, images28[n], apply_sign_mode(maps[n], cfg.SIGN_MODE), masks[n],
            targets[n], cfg.DELETION_STEPS)) for n in range(n_img)]
        result[name] = float(np.mean(aucs))
    rng = np.random.RandomState(seed)
    result["random"] = float(np.mean([deletion_auc(deletion_curve(
        predict_proba_fn, images28[n], None, masks[n], targets[n], cfg.DELETION_STEPS,
        order="random", rng=rng)) for n in range(n_img)]))
    return result
