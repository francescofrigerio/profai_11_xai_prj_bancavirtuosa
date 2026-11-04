"""
"""

from pathlib import Path
import numpy as np
import torch
from matplotlib.figure import Figure
from data import debug_indices, to_densenet_input
from xai.gradcam import GradCam

from captum.attr import LayerGradCam

from xai.explainer import Explainer

def make_gradcam_debug_callback(data, cfg, device, out_dir, log=print):
    """
        Grad-CAM debug images of the first and last training epoch.
        The callback explains the PREDICTED class of the 10 fixed validation images (one per class), so a
        broken training (maps that do not move, or that look at the background) 
        is visible at a glance.
    """
    out_dir = Path(out_dir)
    idx = debug_indices(data.y_trainval.numpy(), 
                        data.split.idx_val, 
                        cfg.NUM_DEBUG_IMAGES)
    
    x28 = data.X_trainval[torch.as_tensor(idx)]
    labels = data.y_trainval[torch.as_tensor(idx)].numpy()

    def callback(model, epoch):
        model.eval()
        explainer = GradCam(model, cfg)
        fig = Figure(figsize=(2 * len(idx), 4.4))
        axes = fig.subplots(2, len(idx))
        for k in range(len(idx)):
            input_img = to_densenet_input(x28[k:k + 1], 
                                   cfg.DENSENET_SIZE).to(device)
            with torch.no_grad():
                pred = int(model(input_img).argmax(1))
            result = explainer.explain(input_img, pred)
            axes[0, k].imshow(x28[k, 0], cmap="gray")
            axes[0, k].set_title(f"true {labels[k]} / pred {pred}", fontsize=8)
            axes[1, k].imshow(result.map, cmap="viridis", interpolation="bilinear")
            for row in (0, 1):
                axes[row, k].axis("off")
        fig.suptitle(f"Grad-CAM debug, epoch {epoch}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gradcam_debug_epoch{epoch}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        log(f"debug Grad-CAM saved: {path}")

    return callback


class GradCam(Explainer):
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