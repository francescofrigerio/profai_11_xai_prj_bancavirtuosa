"""
    Saliency (Captum): absolute gradient of the logit with respect to each pixel, 
    summed over channels.

    The simplest gradient technique and the noisiest: phase 0 measured that only ~23 % of its mass
    falls on the digit (a random map would give ~22 %). It is kept as the didactic starting point.
"""
from __future__ import annotations
from captum.attr import Saliency
from xai.explainer import Explainer


class SaliencyCaptum(Explainer):
    """
            get Captum Occlusion map
            Could be exclude this captum on production
    """
    name = "saliency"
    signed = False

    def attribute(self, input_img, input_label):
        att = Saliency(self.model).attribute(input_img.clone().requires_grad_(True),
                                             target=input_label, abs=True)
        return att[0].sum(0).detach().cpu().numpy()
