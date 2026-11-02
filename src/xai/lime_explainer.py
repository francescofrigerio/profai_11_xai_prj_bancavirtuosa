"""
    Lime Explainer
"""
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image

from utility import denormalize_imagenet, normalize_imagenet
from xai import Explainer


def grid_segments(size: int, cells: int) -> np.ndarray:
    """
        Segment label (0 .. cells**2-1) of every pixel of a size x size image, 
        in equal square cells.
    """
    if size % cells:
        raise ValueError(f"size {size} is not a multiple of {cells} cells")
    rows = np.arange(size) // (size // cells)
    return (rows[:, None] * cells + rows[None, :]).astype(np.int64)


class Lime(Explainer):
    """
    """
    name = "lime"
    signed = True
    output_space = "probability"

    def attribute(self, input_img, input_label):
       
        cfg, model, device = self.cfg, self.model, input_img.device
        size = input_img.shape[-1]
        img = denormalize_imagenet(input_img)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float64)
        segments = grid_segments(size, cfg.LIME_GRID_CELLS)

        def classifier_fn(batch):                    # LIME works on images in the ORIGINAL [0, 1] domain
            t = normalize_imagenet(torch.from_numpy(batch).float().permute(0, 3, 1, 2)).to(device)
            out = []
            with torch.no_grad():
                for i in range(0, len(t), 64):
                    out.append(F.softmax(model(t[i:i + 64]), dim=1).cpu())
            return torch.cat(out).numpy()

        explainer = lime_image.LimeImageExplainer(random_state=cfg.SEED)
        kwargs = dict(labels=(input_label,), top_labels=None, hide_color=0, num_samples=cfg.LIME_NUM_SAMPLES,
                      batch_size=64, segmentation_fn=lambda im: segments, random_seed=cfg.SEED)
        if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
            kwargs["progress_bar"] = False           # absent in lime 0.2.0.1
        explanation = explainer.explain_instance(img, classifier_fn, **kwargs)
        out = np.zeros((size, size))
        for segment, weight in dict(explanation.local_exp[input_label]).items():
            out[segments == segment] = weight
        return out
