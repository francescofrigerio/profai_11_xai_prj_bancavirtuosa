"""
   Integrated Gradients (Captum) from the black image, summed over the colour channels.

   The attribution is exactly zero wherever the image equals the baseline (the black background), and
   its sum must equal f(x) - f(baseline) (completeness): the error is reported in the metadata.
"""
from __future__ import annotations
import torch
from captum.attr import IntegratedGradients as ig

from utility import black_normalized
from xai import Explainer


class IntegratedGradientsCaptum(Explainer):
    """
            get Captum IntegratedGradients map
    """
    name = "integrated_gradients"
    signed = True
    # if near 0 then the approssimate with n_steps was been accurate
    conv_delta = 0.0
    last_sum = 0.0

    def attribute(self, input_img, input_label):
        """
            Computes Integrated Gradients and stores the convergence delta.
            get Integrated Gradients and returns :
            - ig_attr: pixels attributions.
            - self.conv_delta: How much Integrated Gradients approssimate
                               satisfies the completeness property.

            How Much ΣIG_i(x) approximates F(x)-F(x')
            where:
            - (x) image to explain
            - (x') baseline here blach and  normalized image;
            - (F(x)) logit of target class 
            - (IG_i(x)) attribution of element i.

            convergence delta of Captum measure 
            error delta approximates between the attribution sum and the difference
            between the model's output on input img F(x) and baseline F(x')  .

            Captum provides the convergence delta to diagnose 
            the numerical approximation of Integrated Gradients (IG); 
            
            meta() function calculates the difference between the model output 
            and the sum of the attributions.
            These are two complementary checks for the same theoretical property.
        """
        ig_attr, self.conv_delta = ig(self.model).attribute(input_img, baselines=black_normalized(self.cfg.DENSENET_SIZE, 
                                                                                    input_img.device), 
                                        target=input_label,
                                        n_steps=self.cfg.IG_STEPS, 
                                        internal_batch_size=self.cfg.IG_BATCH_SIZE,
                                        return_convergence_delta=True )
        
        self.last_sum = ig_attr.sum().item()
        # 1. Take abs to consider the magnitude of the element
        # 2. Select element of the batch 
        # 3. Sum along the channels to reduce the shape at (H, W)
        # ( difference compare to Occlusion : 
        # every channel contributes to the value attributes at pixel)
        # 4. detach from compute graph
        # 5. Convert to NumPy 2D
        # return ig_attr[0].sum(0).detach().cpu().numpy()
        return ig_attr[0].sum(0).detach().cpu().numpy()

    def meta(self, input_img, input_label):
        """
            This function is used to valid the map
            differenza effettiva tra il logit dell'immagine originale e quello del baseline.

            delta is the benchmark to check the attribution completness
        """
        base = black_normalized(self.cfg.DENSENET_SIZE, input_img.device)
        with torch.no_grad():
            delta = (self.model(input_img)[0, input_label] - self.model(base)[0, input_label]).item()
        # error relative and norm
        error = abs(self.last_sum - delta) / max(abs(delta), 1e-6)
        return {"completeness_error": error, "f_x_minus_f_baseline": delta, "steps": self.cfg.IG_STEPS}
