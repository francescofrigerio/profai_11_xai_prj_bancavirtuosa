"""
"""

from pathlib import Path
import numpy as np

from captum.attr import LayerGradCam
from xai.explainer import Explainer


class GradCamCaptum(Explainer):
    """
        get Captum GradCam map
    """
    name = "gradcam"
    # never negative for Relu
    signed = False                      # relu_attributions=True: only what pushes the class UP

    def __init__(self, model, cfg, layer_name=None):
        """
        """
        super().__init__(model, cfg)
        self.layer_name = layer_name or cfg.GRADCAM_LAYER
        modules = dict(model.named_modules())
        if self.layer_name not in modules:
            raise ValueError(f"Grad-CAM layer {self.layer_name} is not a module of the model")
        self.layer = modules[self.layer_name]

    def attribute(self, input_img, input_label):
        """
            Using hook on the feature map and gradients of the model.
        """
        gd_attr = LayerGradCam(self.model, 
                           self.layer).attribute(input_img, target=input_label, relu_attributions=True)

        return gd_attr[0, 0].detach().cpu().numpy()