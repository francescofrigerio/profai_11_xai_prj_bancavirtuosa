"""Grad-CAM (Captum) on a convolutional layer: coarse map (S/32 on norm5, S/16 on denseblock3)."""
from __future__ import annotations

from xai.contract import Explainer


class GradCam(Explainer):
    name = "gradcam"
    signed = False                      # relu_attributions=True: only what pushes the class UP

    def __init__(self, model, cfg, layer_name=None):
        super().__init__(model, cfg)
        self.layer_name = layer_name or cfg.GRADCAM_LAYER
        modules = dict(model.named_modules())
        if self.layer_name not in modules:
            raise ValueError(f"Grad-CAM layer {self.layer_name!r} is not a module of the model")
        self.layer = modules[self.layer_name]

    def _attribute(self, x1, target):
        from captum.attr import LayerGradCam
        att = LayerGradCam(self.model, self.layer).attribute(x1, target=target, relu_attributions=True)
        return att[0, 0].detach().cpu().numpy()
