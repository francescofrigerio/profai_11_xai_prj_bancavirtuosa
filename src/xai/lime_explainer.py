"""
    Lime Explainer
"""
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image

from utility import denormalize_imagenet, normalize_imagenet , normalize_imagenet_np
from xai import Explainer


def grid_segments(size: int, cells: int) -> np.ndarray:
    """
        Segment labels of a cells x cells grid over a size x size image.
        This function divides the image into a regular grid of cells × cells 
        squares and assigns a numerical ID (0 .. cells**2-1) to each square.
        size : image matrix size
        cell : block grid cell size

        LIME creates perturbed versions of an image by deactivating certain regions and observing how the model's prediction changes.

        defines which pixels belong to the same region. 
        So, each region is simply a square within a grid, 
        rather than a region identified by a segmentation algorithm.
    """
    if size % cells:
        raise ValueError(f"size {size} is not a multiple of {cells} cells")

    step = size // cells
    rows = np.arange(size) // step
    # The function return an size × size matrix 
    # in which each cells × cells block has a label:
    return (rows[:, None] * cells + rows[None, :]).astype(np.int64)


class Lime(Explainer):
    """
        Lime Explainer
    """
    name = "lime"
    signed = True
    output_space = "probability"

    def attribute(self, input_img, input_label):

        if self.cfg.LIME_METHOD == "quickshift":
            return self.attribute_quickshift(input_img, input_label)

        elif self.cfg.LIME_METHOD == "grid":
            return self.attribute_grid(input_img, input_label)

        else:
            raise ValueError(
                f"Unknown LIME method: {self.cfg.LIME_METHOD}"
            )

    def attribute_quickshift(self, input_img, input_label):

        cfg, model, device = self.cfg, self.model, input_img.device
        size = input_img.shape[-1]
        
        # Extract NumPy Image in the format (H, W, 3)
        img = denormalize_imagenet(input_img)[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy().astype(np.float64)
        segments = grid_segments(size, cfg.LIME_GRID_CELLS)
        
        def classifier_fn(batch):
            norm_images = normalize_imagenet_np(batch)
            # Permute into (B, C, H, W) and send to device
            tensor_in = torch.tensor(norm_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                logits = model(tensor_in)
                probs = F.softmax(logits, dim=1)

            return probs.cpu().numpy()

        explainer = lime_image.LimeImageExplainer(random_state=cfg.SEED)

        # default segemnt based on Quickshift to generate superpixel.
        # generate num_samples sample perturbet .
        # labels=(input_label,): explain selected class.
        # hide_color=0: superpixel disable will be sostitute with 0=black 
        # (zero in the visible image).   
        # top_labels=None: not work with top classes. We work with single class, 
        explanation = explainer.explain_instance(image=img,
        
                                                classifier_fn=classifier_fn,
                                                labels=(int(input_label),),
                                                top_labels=None,
                                                hide_color=0,
                                                num_samples=cfg.LIME_NUM_SAMPLES,
                                                random_seed=cfg.SEED,
                                            )

        segments = explanation.segments
        weights = dict(explanation.local_exp[int(input_label)])

        out = np.zeros(segments.shape, dtype=np.float32)

        for segment_id, weight in weights.items():
            out[segments == segment_id] = weight

        return out

    def attribute_grid(self, input_img, input_label):
        """
            get Captum GradCam map
        """
        cfg, model, device = self.cfg, self.model, input_img.device
        size = input_img.shape[-1]

        # Extract NumPy Image in the format (H, W, 3)
        img = denormalize_imagenet(input_img)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float64)
        segments = grid_segments(size, cfg.LIME_GRID_CELLS)

        def classifier_fn(batch): 
            """
                Predict Function for LIME:
                Receive perturbate images from LIME (in [0, 1]), 
                and apply ImageNet norm before pass the image to DenseNet
            """                  
            # LIME works on images in the ORIGINAL [0, 1] domain
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
