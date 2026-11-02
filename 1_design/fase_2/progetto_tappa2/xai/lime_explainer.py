"""LIME (reference package `lime`) on an 8x8 grid of square superpixels, with black as fill colour.

LIME explains the PROBABILITY of the class (the convention of the package), not the logit; the
metadata says so. The grid is used instead of quickshift: phase 0 showed quickshift cuts the
background into 100-130 tiny segments.
"""
from __future__ import annotations

import inspect

import numpy as np

from data import denormalize, normalize
from xai.contract import Explainer


def grid_segments(size: int, cells: int) -> np.ndarray:
    """Segment label (0 .. cells**2-1) of every pixel of a size x size image, in equal square cells."""
    if size % cells:
        raise ValueError(f"size {size} is not a multiple of {cells} cells")
    rows = np.arange(size) // (size // cells)
    return (rows[:, None] * cells + rows[None, :]).astype(np.int64)


class Lime(Explainer):
    name = "lime"
    signed = True
    output_space = "probability"

    def _attribute(self, x1, target):
        import torch
        import torch.nn.functional as F
        from lime import lime_image
        cfg, model, device = self.cfg, self.model, x1.device
        size = x1.shape[-1]
        img = denormalize(x1)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float64)
        segments = grid_segments(size, cfg.LIME_GRID_CELLS)

        def classifier_fn(batch):                    # LIME works on images in the ORIGINAL [0, 1] domain
            t = normalize(torch.from_numpy(batch).float().permute(0, 3, 1, 2)).to(device)
            out = []
            with torch.no_grad():
                for i in range(0, len(t), 64):
                    out.append(F.softmax(model(t[i:i + 64]), dim=1).cpu())
            return torch.cat(out).numpy()

        explainer = lime_image.LimeImageExplainer(random_state=cfg.SEED)
        kwargs = dict(labels=(target,), top_labels=None, hide_color=0, num_samples=cfg.LIME_NUM_SAMPLES,
                      batch_size=64, segmentation_fn=lambda im: segments, random_seed=cfg.SEED)
        if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
            kwargs["progress_bar"] = False           # absent in lime 0.2.0.1
        explanation = explainer.explain_instance(img, classifier_fn, **kwargs)
        out = np.zeros((size, size))
        for segment, weight in dict(explanation.local_exp[target]).items():
            out[segments == segment] = weight
        return out
