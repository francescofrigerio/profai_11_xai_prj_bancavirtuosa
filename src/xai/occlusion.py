"""
    Occlusion (Captum): slide a black window over the image and see how the logit drops.

    The window covers the three channels together and overlapping windows are averaged by Captum
    (phase 0, check 11), so every pixel gets one value; the three identical channel maps are averaged.
"""
from __future__ import annotations

from data import black_normalized
from xai import Explainer
from captum.attr import Occlusion as CaptumOcclusion

class Occlusion(Explainer):
    name = "occlusion"
    signed = True

    def attribute(self, x1, target):
        w, s = self.cfg.OCCLUSION_WINDOW, self.cfg.OCCLUSION_STRIDE
        att = CaptumOcclusion(self.model).attribute(
            x1, sliding_window_shapes=(3, w, w), strides=(3, s, s),
            baselines=black_normalized(self.cfg.DENSENET_SIZE, x1.device), target=target,
            perturbations_per_eval=32, show_progress=False)
        return att[0].mean(0).detach().cpu().numpy()