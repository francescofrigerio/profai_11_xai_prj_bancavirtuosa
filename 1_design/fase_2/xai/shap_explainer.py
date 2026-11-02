"""SHAP (reference package `shap`): expected gradients over a fixed background of training images."""
from __future__ import annotations

import numpy as np

from data import to_densenet_input
from xai.contract import Explainer


def make_background(X28, idx_train, cfg, device):
    """SHAP_NUM_BACKGROUND training images, always the same for a given seed."""
    import torch
    rng = np.random.RandomState(cfg.SEED)
    pick = rng.choice(np.asarray(idx_train), cfg.SHAP_NUM_BACKGROUND, replace=False)
    return to_densenet_input(X28[torch.as_tensor(pick)], cfg.DENSENET_SIZE).to(device)


def pick_shap_target(sv, target: int, num_classes: int) -> np.ndarray:
    """shap returns a list per class (older versions) or an array with the classes last (newer)."""
    if isinstance(sv, list):
        arr = np.asarray(sv[target])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 5 and arr.shape[-1] == num_classes:
            arr = arr[..., target]
        elif arr.ndim == 5 and arr.shape[0] == num_classes:
            arr = arr[target]
        else:
            raise ValueError(f"unexpected shap output shape {arr.shape}")
    return arr[0].sum(axis=0)                          # (3, S, S) -> (S, S)


class Shap(Explainer):
    name = "shap"
    signed = True

    def __init__(self, model, cfg, background):
        super().__init__(model, cfg)
        self.background = background

    def _attribute(self, x1, target):
        import shap
        from training import set_seed
        set_seed(self.cfg.SEED)                        # shap draws with numpy AND torch
        values = shap.GradientExplainer(self.model, self.background).shap_values(
            x1, nsamples=self.cfg.SHAP_NUM_SAMPLES)
        return pick_shap_target(values, target, self.cfg.NUM_CLASSES)
