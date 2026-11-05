"""
     Captum Explainer
"""

import torch
import numpy as np
import shap

from training import set_seed
from xai import Explainer
from utility import to_densenet_input

def make_background(input_images, idx_train, cfg, device):
    """
        This function select the SHAP background
        because GradientExplainer needs background dataset.
        This background is a benchmark distribution 
        against which SHAP evaluates how much the image being explained 
        contributes to the prediction.
        SHAP_NUM_BACKGROUND images, always the same for a given seed.

        input_images	Tutte le immagini trainval, shape (N, 1, 28, 28)
        idx_train	Indici del training set all'interno di trainval

    """
    
    rng = np.random.RandomState(cfg.SEED) # pylint: disable=no-member
    
    pick = rng.choice(np.asarray(idx_train), cfg.SHAP_NUM_BACKGROUND, replace=False)
    return to_densenet_input(input_images[torch.as_tensor(pick)], cfg.DENSENET_SIZE).to(device)


def pick_shap_target(shap_values, input_label: int, num_classes: int) -> np.ndarray:
    """
        SHAP has already been calculated. 
        This function serves to:
        Interpret the output format returned by the SHAP library.
        Select the attributions for the class to explain.
        Reduce the channel dimension from (3, H, W) to (H, W).

        The format returned by SHAP has changed between different versions of the library.

        shap returns a list for class (older versions) 
        or an array with the classes last (newer version).
    """
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[input_label])
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 5 and arr.shape[-1] == num_classes:
            arr = arr[..., input_label]
        elif arr.ndim == 5 and arr.shape[0] == num_classes:
            arr = arr[input_label]
        else:
            raise ValueError(f"unexpected shap output shape {arr.shape}")
    # Convert to (3, S, S) -> (S, S)
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
                                        self.background).shap_values(input_img, nsamples=self.cfg.SHAP_NUM_SAMPLES)
        return pick_shap_target(values, input_label, self.cfg.NUM_CLASSES)