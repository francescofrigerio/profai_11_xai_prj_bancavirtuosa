"""Saliency (Captum): absolute gradient of the logit with respect to each pixel, summed over channels.

The simplest gradient technique and the noisiest: phase 0 measured that only ~23 % of its mass
falls on the digit (a random map would give ~22 %). It is kept as the didactic starting point.
"""
from __future__ import annotations

from xai.contract import Explainer


class Saliency(Explainer):
    name = "saliency"
    signed = False

    def _attribute(self, x1, target):
        from captum.attr import Saliency as CaptumSaliency
        att = CaptumSaliency(self.model).attribute(x1.clone().requires_grad_(True), target=target, abs=True)
        return att[0].sum(0).detach().cpu().numpy()
