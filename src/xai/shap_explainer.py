"""
     Captum Explainer
"""

import torch
import numpy as np
import shap

from training import set_seed
from xai import Explainer

def make_background(X28, idx_train, cfg, device):
    """
        SHAP_NUM_BACKGROUND training images, always the same for a given seed.
    """
    
    rng = np.random.RandomState(cfg.SEED)
    pick = rng.choice(np.asarray(idx_train), cfg.SHAP_NUM_BACKGROUND, replace=False)
    return to_densenet_input(X28[torch.as_tensor(pick)], cfg.DENSENET_SIZE).to(device)


def pick_shap_target(sv, input_label: int, num_classes: int) -> np.ndarray:
    """
        shap returns a list per class (older versions) or an array with the classes last (newer).
    """
    if isinstance(sv, list):
        arr = np.asarray(sv[input_label])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 5 and arr.shape[-1] == num_classes:
            arr = arr[..., input_label]
        elif arr.ndim == 5 and arr.shape[0] == num_classes:
            arr = arr[input_label]
        else:
            raise ValueError(f"unexpected shap output shape {arr.shape}")
    # (3, S, S) -> (S, S)
    return arr[0].sum(axis=0) 


class Shap(Explainer):
    """
        Shap explainer
    """
    name = "shap"
    signed = True

    def __init__(self, model, cfg, background):
        super().__init__(model, cfg)
        self.background = background

    def attribute(self, input_img, input_label):
        
        set_seed(self.cfg.SEED) 
        # shap draws with numpy AND torch                       
        values = shap.GradientExplainer(self.model, 
                                        self.background).shap_values(input_img, 
                                                                     nsamples=self.cfg.SHAP_NUM_SAMPLES)
        return pick_shap_target(values, input_label, self.cfg.NUM_CLASSES)