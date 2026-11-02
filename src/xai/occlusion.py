"""
    Occlusion (Captum): slide a black window over the image and see how the logit drops.

    Occlusion scroll black window (patch) over the image
    and measure How much decrease classe target's prob 
    when a zone is covered .

    The window covers the three channels together and overlapping windows are averaged by Captum
    , so every pixel gets one value; the three identical channel maps are averaged.

    Occlusion get for eache zone Occluse, 
    the difference of the output's model compare to the output with original image
    Semplify we can write ΔF = F(x)-F(x_{occluded})]
    This difference could be distribuited upon
    the positions of the occluse window. 
    The windows overlap together so a single pixels
    could recerive contribute  from different pertubation windows.
"""
from __future__ import annotations
from captum.attr import Occlusion 

from utility import black_normalized
from xai import Explainer




class OcclusionCaptum(Explainer):
    """
        get Captum Occlusion map
    """
    name = "occlusion"
    signed = True

    def attribute(self, input_img, input_label):
        """
            get tensors map attribution with shape :(B,C,H,W)
        """
        win, stride = self.cfg.OCCLUSION_WINDOW, self.cfg.OCCLUSION_STRIDE
        occ_attr = Occlusion(self.model).attribute(
                            input_img, sliding_window_shapes=(3, win, win), strides=(3, stride, stride),
                            baselines=black_normalized(self.cfg.DENSENET_SIZE, input_img.device), 
                            target=input_label,
                            perturbations_per_eval=32, 
                            show_progress=False)

        # 1. Take abs to consider the magnitude of the element
        # 2. Select element of the batch (1, C, H, W) -> (C, H, W)
        # 3. get sum along the channels 
        # (difference compare to IG : 
        # ls -lrattributes has already detect an output 
        # difference for each window and has distribuited on the occlus position)
        # 4. detach from compute graph
        # 5. force on cpu
        # 6. Convert to NumPy 2D
        return occ_attr.abs()[0].sum(0).detach().cpu().numpy()
