"""
"""

from pathlib import Path
import numpy as np

from data import to_densenet_input
from captum.attr import LayerGradCam
from xai.explainer import Explainer

def make_gradcam_debug_callback(cfg,logger,dataset, device, out_path , list_images):
    """
        Grad-CAM debug images of the first and last training epoch.
        The callback explains the PREDICTED class of the 10 fixed validation images (one per class), so a
        broken training (maps that do not move, or that look at the background) 
        is visible at a glance.
    """
    import torch
    from matplotlib.figure import Figure
    from data import to_densenet_input
    from xai.gradcam import GradCam

    # list_images = load_images_by_class(logger,out_path / input_file)
    
    def callback(model, epoch):
           model.eval()
           explainer = GradCam(model, cfg)

           n_images = len(list_images)
           fig = Figure(figsize=(2 * n_images, 4.4))
           axes = fig.subplots(2, n_images)
           for i, id_class in enumerate(sorted(list_images.keys())):
               
               item = list_images[id_class]
               idx_dataset = item["id_dataset"]
               
               # The function to_densenet_input() has to receive a batch with 4 dim. 
               # So we add the size batch here 
               # Original Image MNIST: (1, 28, 28)
               input_image, input_label = dataset[idx_dataset]
               
               # Add size of the batch: (1, 1, 28, 28) to the input
               x28 = input_image.unsqueeze(0)

               # Convert to Input DenseNet: (1, 3, 128, 128)
               input_img = to_densenet_input(x28,cfg.DENSENET_SIZE).to(device)

               with torch.no_grad():
                   # pred = int(model(input_img).argmax(1))
                   pred = int(model(input_img).argmax(dim=1).item())
               
               result = explainer.explain(input_img, pred)
               
               # Original Image
               axes[0, i].imshow(input_image.squeeze(0),cmap="gray")
               axes[0, i].set_title(f"true {input_label} / pred {pred}",fontsize=8)
               
               # Grad-CAM
               axes[1, i].imshow(result.map, cmap="viridis", interpolation="bilinear")
               
               for row in (0, 1):
                   axes[row, i].axis("off")
           
           fig.suptitle(f"Grad-CAM debug, epoch {epoch}")
           out_path.mkdir(parents=True, exist_ok=True)
           path = out_path / f"gradcam_debug_epoch{epoch}.png"
           fig.savefig(path, dpi=100, bbox_inches="tight")
           logger(f"debug Grad-CAM saved: {path}")

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