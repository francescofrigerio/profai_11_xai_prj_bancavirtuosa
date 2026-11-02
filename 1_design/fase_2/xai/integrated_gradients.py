"""Integrated Gradients (Captum) from the black image, summed over the colour channels.

The attribution is exactly zero wherever the image equals the baseline (the black background), and
its sum must equal f(x) - f(baseline) (completeness): the error is reported in the metadata.
"""
from __future__ import annotations

import torch

from data import black_normalized
from xai.contract import Explainer


class IntegratedGradients(Explainer):
    name = "integrated_gradients"
    signed = True

    def _attribute(self, x1, target):
        from captum.attr import IntegratedGradients as CaptumIG
        att = CaptumIG(self.model).attribute(
            x1, baselines=black_normalized(self.cfg.DENSENET_SIZE, x1.device), target=target,
            n_steps=self.cfg.IG_STEPS, internal_batch_size=50)
        self._last_sum = att.sum().item()
        return att[0].sum(0).detach().cpu().numpy()

    def _meta(self, x1, target):
        base = black_normalized(self.cfg.DENSENET_SIZE, x1.device)
        with torch.no_grad():
            delta = (self.model(x1)[0, target] - self.model(base)[0, target]).item()
        error = abs(self._last_sum - delta) / max(abs(delta), 1e-6)
        return {"completeness_error": error, "f_x_minus_f_baseline": delta, "steps": self.cfg.IG_STEPS}
