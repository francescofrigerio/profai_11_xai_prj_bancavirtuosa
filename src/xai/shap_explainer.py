"""
     Captum Explainer
"""

import torch
import numpy as np
import shap

from training import set_seed
from xai import Explainer
from utility import to_densenet_input

def make_background(transform_train, cfg,logger, device):
    """
        This function obtain all the MNIST images 
        as tensors with a shape of (N, 1, 28, 28) and with values 
        normalized between 0 and 1, and so we pass transform_train that
        is already with this format.

        This function select the SHAP background
        because GradientExplainer needs background dataset.
        This background is a benchmark distribution 
        against which SHAP evaluates how much the image being explained 
        contributes to the prediction.
        SHAP_NUM_BACKGROUND images, always the same for a given seed.

        Details:
        pick: Selection process:
        1. Select a subset of images: 
        extracting SHAP_NUM_BACKGROUND (recommended values: 100, 50) indices from idx_train, 
        which identify the images to be used as the SHAP background. 
        Consequently, the entire training set is not used.
        2. Ensure a reproducible selection: the random number generator (rng) is initialized
        using cfg.SEED; therefore, given the same indices and parameters, 
        the selection will always be identical.
        3. replace=False ensures that no duplicate images are selected.

        The function selects SHAP_NUM_BACKGROUND distinct images from those contained in idx_train.
        Thus, the "pick" step serves not only to set the seed—making the selection repeatable—
        but also to choose the background images.

        input_images	Tutte le immagini trainval, shape (N, 1, 28, 28)



    """
    logger.info(f" transform_train len {len(transform_train)}")
    rng = np.random.RandomState(cfg.SEED) # pylint: disable=no-member
    pick = rng.choice(len(transform_train),
                     cfg.SHAP_NUM_BACKGROUND,
                     replace=False
                    )
    # Load images selected with pick
    input_images = torch.stack([transform_train[int(i)][0] for i in pick])
    logger.info(f" input_images shape {input_images.shape}")
    # not run other select with pick and other normalization 
    # return to_densenet_input(input_images[torch.as_tensor(pick)], cfg.DENSENET_SIZE).to(device)
    return input_images.to(device)

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