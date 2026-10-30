# from xai.contract import Explainer
from xai.explainer import Explainer
from captum.attr import LayerGradCam

class GradCam(Explainer):
    """
    """
    name = "gradcam"
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

    def attribute(self, x_data, target):
        """
        """
        att = LayerGradCam(self.model, 
                           self.layer).attribute(x_data, target=target, relu_attributions=True)
        return att[0, 0].detach().cpu().numpy()