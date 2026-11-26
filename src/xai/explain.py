"""
    To ensure that the five saliency maps (Grad-CAM, Saliency, 
    Integrated Gradients,Occlusion, and SHAP/LIME) 
    can be compared using correlation and IoU metrics,
    Min-Max normalization to the range [0, 1] 
    must be applied to each map.

    This produces five homogeneous 2D matrices 
    with values in the [0, 1] range,
    ready to be used in Spearman correlation or IoU 
    calculations.

    Saliency Map Normalization (Min-Max [0, 1]):
    Apply this normalization after the XAI algorithm 
    has computed the pixel
    attributions, in order to make the maps generated 
    by different techniques
    homogeneous and directly comparable 
    (e.g., Grad-CAM and Integrated Gradients
    can produce raw attribution values 
    on very different scales).

    Normalization should be performed before passing the maps to agreement metrics
    such as IoU, or before visualizing them with Matplotlib.

    Differences in Scale Across the Five XAI Techniques:
    The five XAI techniques used compute feature importance in mathematically
    different ways, resulting in raw attribution values with incompatible scales:

    - Grad-CAM: Produces non-negative values in the range [0, +infinity),
        derived from convolutional activations and a ReLU operation.
    - Integrated Gradients: Produces attributions based on the integral of the
          gradients multiplied by the input difference (x - x_0), often yielding
          small values (e.g., between 10^-4 and 10^-2).
    - SHAP / LIME: Produces Shapley values or coefficients from a linear surrogate
          model, respectively.
    - Occlusion: Produces attribution values based on the change in the output
          score or probability when parts of the input are occluded
          (e.g., values between -1.0 and +1.0).

    Comparing raw maps using IoU or correlation metrics without appropriate
    normalization may lead to misleading results due to differences in scale.

    After Min-Max normalization, the least important pixel in the map is assigned
    a value of 0.0, while the most important pixel is assigned a value of 1.0.
    All five techniques therefore produce maps scaled to the [0.0, 1.0] range,
    ready for use in Spearman correlation, IoU calculations, and Matplotlib
    visualization.
    
    Technical note: Min-Max normalization makes the numerical ranges comparable, 
    but it does not by itself make all attribution maps semantically equivalent. 
    In particular, signed attributions (such as those from Integrated Gradients or 
    Occlusion) may require a consistent policy for handling positive 
    and negative contributions before computing IoU.
"""

import inspect
import numpy as np
from captum.attr import LayerGradCam
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import Occlusion
import shap
from lime import lime_image

from xai.integrated_gradients import IntegratedGradientsCaptum
from xai.shap_explainer import ShapExplainer

from utility import normalize_imagenet , normalize_imagenet_np , set_seed
from utility import denormalize_imagenet , black_normalized , to_grid28

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_ERROR = None
except Exception as exc:                     # reported by check 1, not a crash
    torch = nn = F = None
    _TORCH_ERROR = exc

from xai.metrics_xai import deletion_curve_for_faithfulness 
from xai.metrics_xai import spearman_agreement_mean , deletion_metrics_from_maps



def explain_gradcam(model, input_img, input_label, layer_name, relu=True):
    """
        input_img (B, C, H, W) (1,3,128,128)
        target = input_label
    """
    
    layer = dict(model.named_modules())[layer_name]
    gradcam_attr = LayerGradCam(model, layer).attribute(input_img, 
                                                        target=int(input_label), 
                                                        relu_attributions=relu)
    return gradcam_attr.squeeze().detach().cpu().numpy()
      
def explain_integrated_gradients(cfg,model, input_img, input_label):
    """
        x1 = input_img (B, C, H, W) (1,3,128,128)
        target = input_label
            
        n_steps parameters=50 gets integral decision.s 
        for test/debug n_steps=20 to be faster

        attr->get tensor of shape (1, C, H, W))
    """
    ig = IntegratedGradients(model)
    # Baseline nera della stessa dimensione dell'input (stesso device)
    baseline = torch.zeros_like(input_img)
        
    # Calcolo dell'attribuzione (ritorna un tensore di forma (1, C, H, W))attribuzione (ritorna un tensore di forma (1, C, H, W))attribuzione (ritorna un tensore di forma (1, C, H, W))
    ig_attr, delta_attr = ig.attribute( input_img, 
                                  baselines=baseline, 
                                  target=int(input_label), 
                                  n_steps=cfg.IG_STEPS, 
                                  return_convergence_delta=True
                              )
                              
    print(f"check: delta {delta_attr}"
           "if near 0 then the approssimate with n_steps was been accurate ")
    # 1. Take abs to consider the magnitude of the element
    # 2. Select element of the batch (1, C, H, W) -> (C, H, W)
    # 3. Sum along the channels to reduce the shape at (H, W)
    # 4. Convert to NumPy 2D
    attr_2d = ig_attr[0].sum(0).detach().cpu().numpy()
        
    return attr_2d
    
    
def explain_ig(model, input_img, input_label, cfg, steps=None):
    """
            x1 = input_img (B, C, H, W) (1,3,128,128)
            target = input_label
                
            n_steps parameters=50 gets integral decision.s 
            for test/debug n_steps=20 to be faster
    
            attr->get tensor of shape (1, C, H, W))
    """

    baseline = black_normalized(input_img.shape[-1], input_img.device)
    ig_attr ,delta_attr = IntegratedGradients(model).attribute(input_img, 
                                                   baselines=black_normalized, 
                                                   target=int(input_label),
                                                   n_steps=cfg.IG_STEPS, 
                                                   # it avoid issue of Out of Memory
                                                   # procede gradually step by step
                                                   internal_batch_size=cfg.IG_BATCH_SIZE,
                                                   return_convergence_delta=True 
                                                   )
    print(f"check: delta {delta_attr}"
            "if near 0 then the approssimate with n_steps was been accurate ")
    # 1. Take abs to consider the magnitude of the element
    # 2. Select element of the batch (1, C, H, W) -> (C, H, W)
    # 3. Sum along the channels to reduce the shape at (H, W)
    # 4. Convert to NumPy 2D
    attr_2d = ig_attr[0].sum(0).detach().cpu().numpy()

    return attr_2d

    
def explain_saliency(model, input_img, input_label):
    """
        input_img (B, C, H, W) (1,3,128,128)
        target = input_label
    """
    
    sal_att = Saliency(model).attribute(input_img.clone().requires_grad_(True), target=int(input_label), abs=True)
    # sum over channels, >= 0
    return sal_att[0].sum(0).detach().cpu().numpy()  
        
def explain_occlusion(cfg,model, input_img, input_label,device):
    """
        input_img (B, C, H, W) (1,3,128,128)
        target = input_label
            
        Occlusion scroll black window (patch) over the image
        and measure How much decrease classe target's prob 
        when a zone is covered .
    """   
    

    occlusion = Occlusion(model)
        
    o_w, o_s = cfg.OCCLUSION_WINDOW, cfg.OCCLUSION_STRIDE
    o_size = cfg.DENSENET_SIZE
    # ones, zeros = torch.ones(1, 3, o_size, o_size, device=device), torch.zeros(1, 3, o_size, o_size, device=device)
    zeros = torch.zeros(1, 3, o_size, o_size, device=device)

    # Calcolo dell'attribuzione (ritorna un tensore di forma (1, C, H, W))
    occ_attr = occlusion.attribute( input_img,
                                        strides=(3, o_s, o_s), # scroll step (C, H, W)
                                        sliding_window_shapes=(3,o_w,o_w), # patch size (C, H, W)
                                        target=int(input_label),
                                        baselines=zeros # evita problemi sul device
                                    )
        
    # 1. Take abs to consider the magnitude of the element
    # 2. Select element of the batch (1, C, H, W) -> (C, H, W)
    # 3. Sum along the channels to reduce the shape at (H, W)
    # 4. Convert to NumPy 2D
    attr_2d = occ_attr[0].sum(0).detach().cpu().numpy()
        
    return attr_2d
        


def explain_shap_lib_1(cfg,model, input_img, input_label, n_evals=200):
    """
    Wrapper for SHAP with manage correctly of the normalization .
    input_img:  PyTorch Tensor (1, 3, H, W)
    """
    model.eval()
    # take the same device of the model
    device = next(model.parameters()).device

    # Extract the image with shape HWC from [1,H,W,C]
    img_hwc_norm = input_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # DENORM: Convert the image into the pixel view range [0, 1]
    img_hwc_clean = denormalize_imagenet(img_hwc_norm)

    # Predict Function: Riceive Images in the range [0, 1],
    #  and apply ImageNet norm e predict
    def predict_fn(np_images):
        # Ripristina la normalizzazione ImageNet per la rete: (x - mean) / std
        # norm_images = (np_images - IMAGENET_MEAN) / IMAGENET_STD
        norm_images = normalize_imagenet_np(np_images)
        
        # Convert into (B, C, H, W) PyTorch Tensor
        tensor_in = torch.tensor(norm_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            logits = model(tensor_in)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # Initialize the Shap Explainer using clean image
    masker = shap.maskers.Image("inpaint_telea", img_hwc_clean.shape)
    # Explainer is the class of the object explainer
    explainer = shap.Explainer(predict_fn, masker)

    # Insert the batch of clean image (1, H, W, 3)
    img_batch = np.expand_dims(img_hwc_clean, axis=0)

    # get the SHAP values
    # as Object Explanation calling explainer .
    # Object explanation is a struct container with
    # data that can be slicing
    shap_values = explainer(img_batch, max_evals=n_evals, batch_size=cfg.SHAP_BATCH_SIZE)

    # Extract 2d map (abs magintude over 3 channels)
    attr_3d = shap_values.values[0, :, :, :, int(input_label)] # pylint: disable=no-member
    attr_2d = np.abs(attr_3d).sum(axis=-1)

    return attr_2d
     
def explain_shap_lib_2(cfg,model, input_img, input_label, n_evals=200):
    """
    Wrapper for SHAP using official library 'shap'.
    input_img: PyTorch Torch of shape (1, 3, H, W) 
    input_label: int, target class to explain
    """
    model.eval()
    # take device from the model
    device = next(model.parameters()).device

    
    def predict_fn(np_images):
        """
            Predict SHAP Function : 
            input as NumPy (B, H, W, C) 
            return Prob (B, NumClasses)
        """
        # Converts from (B, H, W, C) to PyTorch tensor (B, C, H, W)
        tensor_in = torch.tensor(np_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            logits = model(tensor_in)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()
       
    # Prepare the image for SHAP in the format NHWC (H, W, 3)
    img_nhwc = input_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    # add the batch size (1, H, W, 3)
    img_batch = np.expand_axis(img_nhwc, axis=0) # (1, H, W, 3)

    # 3. Initialize the Shap Explainer 
    # # (con masker neutral black for the perturbation)
    masker = shap.maskers.Image("inpaint_telea", img_nhwc.shape)
    explainer = shap.Explainer(predict_fn, masker)

    # 4. get the SHAP values
    # get the SHAP values
    # as Object Explanation calling explainer .
    # Object explanation is a struct container with
    # data that can be slicing
    shap_values = explainer(img_batch, max_evals=n_evals, batch_size=cfg.SHAP_BATCH_SIZE)

    # shap_values.values has format: (1, H, W, C, NumClasses)
    # Select explain for the target class: (H, W, C)
    attr_3d = shap_values.values[0, :, :, :, int(input_label)] # pylint: disable=no-member

    # abs value and sum along 3 channels of color -> 2D (H, W)
    attr_2d = np.abs(attr_3d).sum(axis=-1)

    return attr_2d
    
    
def pick_shap_target(sv, input_label, num_classes):
    """
        shap returns a list per class (older) or an array with classes last (newer).
    """
    if isinstance(sv, list):
        arr = np.asarray(sv[input_label])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 5 and arr.shape[-1] == num_classes:
            arr = arr[..., input_label]
        elif arr.ndim == 5 and arr.shape[0] == num_classes:
            arr = arr[input_label]
        else:
            raise ValueError(f"unexpected shap output shape {arr.shape}")
    return arr[0].sum(axis=0)


def explain_shap_pkg(model, input_img, input_label, background, cfg):
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    explainer = shap.GradientExplainer(model, background)
    
    sv = explainer.shap_values(input_img, nsamples=cfg.SHAP_NUM_SAMPLES)
    return pick_shap_target(sv, int(input_label), cfg.NUM_CLASSES)
    

def grid_segments(size: int, cells: int) -> np.ndarray:
    """
        Segment labels of a cells x cells grid over a size x size image.
    """
    step = size // cells
    rows = np.arange(size) // step
    return (rows[:, None] * cells + rows[None, :]).astype(np.int64)

def explain_lime_auto(cfg,model, input_img, input_label, num_samples=500):
    """
        This function run lime with Automatic Segmentation 
        input_img shape (1, 3, H, W)  # batch, channels, h, w
    """
    model.eval()
    # take device from the model
    device = next(model.parameters()).device

    # Extract NumPy Image in the format (H, W, CHANNELS)
    img_hwc_norm = input_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # DENORM: Take the image in the pixel view range [0, 1] for LIME superpixel
    # denorm LIME works on visibile image with  pixel range [0, 1], 
    # LIME not work on image norm for ImageNet.
    img_hwc_clean = denormalize_imagenet(img_hwc_norm)

    def predict_fn(np_images):
        """
            Predict Function acts as an adapter for LIME:
            1. It receives the perturbed images from LIME([0, 1]).
            2. It normalizes them according to ImageNet standards.
            3. It converts the format from NHWC to NCHW.
            4. It passes them to DenseNet.
            5. It returns the probabilities for each class.
        """
       
        # Rollback to the ImageNet normalize for the cnn net : (x - mean) / std
        # norm_images = (np_images - IMAGENET_MEAN) / IMAGENET_STD
        norm_images = normalize_imagenet_np(np_images)
        
        # Permute into (B, C, H, W) and send to device
        tensor_in = torch.tensor(norm_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            logits = model(tensor_in)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

    # 4. Initialize LIME and run explainer on denorm image
    explainer = lime_image.LimeImageExplainer()

    # LIME receive a visibile and clean image[0, 1]
    # LIME uses probabilities return from predict_fn
    # to build the interpretable local model.

    # default segemnt based on Quickshift to generate superpixel.
    # generate num_samples sample perturbet .
    # labels=(input_label,): explain selected class.
    # hide_color=0: superpixel disable will be sostitute with 0=black 
    # (zero in the visible image).   
    # top_labels=None: not work with top classes. We work with single class, 
    explanation = explainer.explain_instance(image=img_hwc_clean.astype(np.float64),                                       
                                             classifier_fn=predict_fn,
                                             top_labels=None,
                                             labels=(int(input_label),),
                                             hide_color=0,
                                             num_samples=num_samples
                                        )

    # Extract the weights map of lime superpixel 
    # This function selects the most important superpixels from the local explanation.
    # positive_only=False: considers both positive and negative contributions.
    # hide_rest=False: keeps all pixels visible.
    # num_features allows for up to n superpixels in the visualization.
    image, mask = explanation.get_image_and_mask(label=int(input_label),
                                                positive_only=False,
                                                hide_rest=False,
                                                num_features=cfg.LIME_NUM_SAMPLES
                                            )

    dict_exp = dict(explanation.local_exp[int(input_label)])
    attr_2d = np.zeros_like(mask, dtype=np.float32)

    # For each segment:
    # segment_id is the superpixel ID.
    # weight is the local contribution estimated by LIME’s linear model.
    for segment_id, weight in dict_exp.items():
        # Assign the absolute value of the weight to all pixels within the segment.
        attr_2d[mask == segment_id] = abs(weight)

    return attr_2d

def explain_lime_man(model, input_img, input_label, cfg, device):
    """
         This function run lime with Manual Segmentation 
    """
    # denorm image 
    shape_last_size = input_img.shape[-1]
    img = denormalize_imagenet(input_img)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float64)
    # build the grid with 64(8x8) regular segments.
    segments = grid_segments(shape_last_size, cfg.LIME_GRID_CELLS)

    def classifier_fn(batch):
        """
            Predict Function acts as an adapter for LIME:
            1. It receives the perturbed images from LIME([0, 1]).
            2. It normalizes them according to ImageNet standards.
            3. It converts the format from NHWC to NCHW.
            4. It passes them to DenseNet.
            5. It returns the probabilities for each class.
        """
        t = normalize_imagenet(torch.from_numpy(batch).float().permute(0, 3, 1, 2)).to(device)
        probs = []
        with torch.no_grad():
            # process the sample with BATCH_SIZE
            for i in range(0, len(t), cfg.LIME_BATCH_SIZE):
                probs.append(F.softmax(model(t[i:i + cfg.LIME_BATCH_SIZE]), dim=1).cpu())
            return torch.cat(probs).numpy()

    
    explainer = lime_image.LimeImageExplainer(random_state=cfg.SEED)
    # segmentation_fn : function that receive the image and return a segments matrix.
    kwargs = dict(labels=(int(input_label),), top_labels=None, hide_color=0,
                  num_samples=cfg.LIME_NUM_SAMPLES, batch_size=cfg.LIME_BATCH_SIZE,
                  segmentation_fn=lambda im: segments, random_seed=cfg.SEED)
    if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
        kwargs["progress_bar"] = False            

    # run explainer
    expl = explainer.explain_instance(img, classifier_fn, **kwargs)
    # get the weights
    out = np.zeros((shape_last_size, shape_last_size))
    for seg, weight in dict(expl.local_exp[int(input_label)]).items():
        # positive weight → the segment favors the class;
        # negative weight → the segment disfavors the class.
        out[segments == seg] = weight
    return out

def explain_target( model, 
                    techniques, 
                    input_img, 
                    target, 
                    mask,
                    fractions, 
                    baseline, 
                    rng, 
                    cfg ):
    """
        maps + quantitative metrics for ONE target class

        Metric	Question answered
        mask_share	How much importance is attributed to the digit?
        mask_frac	How much space does the digit occupy?
        focus_ratio	How concentrated is the importance on the digit relative to its area?
        del_auc	    How high does the probability remain when important pixels are removed?
        del_gain	How much better is the technique compared to random selection?
        agreement_spearman	To what extent do different techniques agree?

        We are measuring three different properties of the explanations:
        - spatial localization (mask_share, focus_ratio);
        - fidelity to the CNN's behavior (del_auc, del_gain);
        - agreement between techniques (agreement_spearman).
        These metrics are not interchangeable, and a given technique may perform 
        well in one dimension while performing worse in another.

        Faithfulness seeks to measure whether an XAI explanation truly 
        represents the model's behavior.

        We have defined two families of measures:
        -Mask-based measures: examines where importance 
        is concentrated in relation to the region occupied by the digit.

        -Deletion-based measures: examines what happens to the prediction 
        when the most important pixels are removed.

        Focus Ratio Interpretation:
        - focus_ratio > 1: importance concentrated on the figure, 
        to a greater extent than would be expected from a uniform distribution;
        - focus_ratio = 1: concentration proportional to the area;
        - focus_ratio < 1: importance relatively concentrated outside the figure.

        DelGai Interpretation:
        DelGain     Meaning
        Positive	XAI performs better than random
        Zero	    XAI performs the same as random
        Negative	XAI performs worse than random

        In the deletion test, a lower curve is generally better than higher.

    """
    # This next runs explaination over the different technique
    # his output is a generation maps
    # Each technique generate his explanation
    maps = {name: technique.explain(input_img, target) for name, technique in techniques.items()}

    # Here, we perform a deletion test using a random importance map.
    # This serves as a benchmark against which XAI techniques can be compared.
    # The random baseline does not represent a random image; rather, 
    # it represents a random ordering of the pixels to be deleted.
    rand_curve = deletion_curve_for_faithfulness(model, 
                                                input_img, 
                                                target,
                                                rng.random((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                                                fractions, 
                                                baseline)
    
    # Calculate the average of the probabilities obtained along the random curve.
    rand_auc = float(np.mean(rand_curve))

    # This next get 
    # the Percentage of area occupied by the digit   
    # The mask contains Boolean values.
    # Therefore, the mean corresponds 
    # to the percentage of pixels identified as belonging to the digit.
    frac_mask = float(mask.mean())

    # This is one of the most interesting metrics.
    # It measures how much of the map’s total importance falls 
    # within the mask of the digit.
    # The formula is: Mask Share = Σiε-mask |Mi| /  Σi |Mi|
    # For example Esempio:
    # Total importance of the map: 100
    # Total importance within the digit: 80
    # Mask Share = 80/100 = 0.80
    # It means that 80% is focused on the digit figure
    metrics = {}
    for name, res in maps.items():
        m = np.asarray(res.map)
        # g = map_to_28(np.abs(m), cfg.IMAGE_SIZE)
        g = to_grid28(np.abs(m), cfg.DENSENET_SIZE)

        mask_share = float(g[mask].sum() / max(g.sum(), 1e-12))

        # Here we manage signed maps 
        # If the map is signed, consider only the positive contributions—that is, 
        # those that provide evidence in favor of the target class.
        # If no positive contributions exist, use the absolute value.
        # This avoids performing the deletion test on an entirely negative map.
        imp = np.maximum(m, 0) if res.signed else m   # evidence FOR the class
        if imp.max() <= 0:
            imp = np.abs(m)

        # Calculate the deletion curve for each XAI technique and compute its mean.
        #  This metric is referred to as `del_auc`.
        # this is an arithmetic mean of the curve values, not a trapezoidal integration. 
        # If the fractions are equally spaced, it serves as a consistent indicator 
        # for comparing the curves, though it does not represent the exact geometric area.
        curve = deletion_curve_for_faithfulness(model, 
                                                input_img, 
                                                target, 
                                                imp, 
                                                fractions, 
                                                baseline)
        
        # This is the metric that compares the XAI technique with random selection.
        # DelGain = AUC_random - AUC_XAI
        auc = float(np.mean(curve))

        metrics[name] = { "mask_share": mask_share,
                          "mask_frac": frac_mask,
                          # This metric compares the concentration of importance with 
                          # the surface area occupied by the figure.
                          # FocusRatio = MaskShare / MaskFraction
                          "focus_ratio": mask_share / max(frac_mask, 1e-12),
                          "del_auc": auc,
                          "del_gain": rand_auc - auc,      # > 0: better than random
                          "del_curve": curve }
    
    meta = { "agreement_spearman": spearman_agreement_mean(maps),
             "del_auc_random": rand_auc,
             "del_curve_random": rand_curve
    }
    
    return maps, metrics, meta


def explain_target_1( techniques, 
                    input_img, 
                    target, 
                    mask,
                    cfg ):
    """
        maps + quantitative metrics for ONE target class

        Metric	Question answered
        mask_share	How much importance is attributed to the digit?
        mask_frac	How much space does the digit occupy?
        focus_ratio	How concentrated is the importance on the digit relative to its area?
        agreement_spearman	To what extent do different techniques agree?

        We are measuring three different properties of the explanations:
        - spatial localization (mask_share, focus_ratio);
        - agreement between techniques (agreement_spearman).
        These metrics are not interchangeable, and a given technique may perform 
        well in one dimension while performing worse in another.

        We have defined a families of measures:
        -Mask-based measures: examines where importance 
        is concentrated in relation to the region occupied by the digit.

        Focus Ratio Interpretation:
        - focus_ratio > 1: importance concentrated on the figure, 
        to a greater extent than would be expected from a uniform distribution;
        - focus_ratio = 1: concentration proportional to the area;
        - focus_ratio < 1: importance relatively concentrated outside the figure.

    """
    # This next runs explaination over the different technique
    # his output is a generation maps
    # Each technique generate his explanation
    maps = {name: technique.explain(input_img, target) for name, technique in techniques.items()}

    # This next get 
    # the Percentage of area occupied by the digit   
    # The mask contains Boolean values.
    # Therefore, the mean corresponds 
    # to the percentage of pixels identified as belonging to the digit.
    frac_mask = float(mask.mean())

    # This is one of the most interesting metrics.
    # It measures how much of the map’s total importance falls 
    # within the mask of the digit.
    # The formula is: Mask Share = Σiε-mask |Mi| /  Σi |Mi|
    # For example Esempio:
    # Total importance of the map: 100
    # Total importance within the digit: 80
    # Mask Share = 80/100 = 0.80
    # It means that 80% is focused on the digit figure
    metrics = {}
    for name, res in maps.items():
        m = np.asarray(res.map)
        # g = map_to_28(np.abs(m), cfg.IMAGE_SIZE)
        g = to_grid28(np.abs(m), cfg.DENSENET_SIZE)

        mask_share = float(g[mask].sum() / max(g.sum(), 1e-12))

        metrics[name] = { "mask_share": mask_share,
                          "mask_frac": frac_mask,
                          # This metric compares the concentration of importance with 
                          # the surface area occupied by the figure.
                          # FocusRatio = MaskShare / MaskFraction
                          "focus_ratio": mask_share / max(frac_mask, 1e-12) }
    
    meta = { "agreement_spearman": spearman_agreement_mean(maps)}
    
    return maps, metrics, meta



def explain_target_2( score_fn,
                      input_img,
                      maps_for_target,
                      fractions,
                      baseline,
                      rng,
                    ):
    
    return deletion_metrics_from_maps( score_fn=score_fn,
                                        input_img=input_img,
                                        maps_for_target=maps_for_target,
                                        fractions=fractions,
                                        baseline=baseline,
                                        rng=rng,
                                    )


def explain_shap_vs_ig( org_idx,model,background, input_img, target_class, score_fn ,cfg,logger ):
    """
         compare shap and ig for debug
    """
    shap_explainer = ShapExplainer(model, cfg, background)
    ig_explainer = IntegratedGradientsCaptum(model, cfg)

    # Inspect raw output of GradientExplainer before pick_shap_target
    set_seed(shap_explainer.cfg.SEED)
    raw_values = shap.GradientExplainer(shap_explainer.model, shap_explainer.background).shap_values(input_img, 
                                                                                                    nsamples=shap_explainer.cfg.SHAP_NUM_SAMPLES)

    logger.info(f"target_class:{target_class} org_idx {org_idx}" )
    logger.info(f"type(raw_values): {type(raw_values)}")
    if isinstance(raw_values, list):
        logger.info(f"len(raw_values):{len(raw_values)} | shapes: {[np.asarray(v).shape for v in raw_values]}")
    else:
        logger.info(f"np.asarray(raw_values).shape:{np.asarray(raw_values).shape}")

    # Compare 2D maps build by SHAP and IG
    m_shap = shap_explainer.attribute(input_img, target_class)
    m_ig = ig_explainer.attribute(input_img, target_class)

    for name, m in [("shap", m_shap), ("ig", m_ig)]:
        m = np.asarray(m)
        pos_frac = float((m > 0).mean())
        neg_frac = float((m < 0).mean())
        logger.info(f"[{name}] shape={m.shape} min={m.min():.6f} max={m.max():.6f} "
                   f"mean={m.mean():.6f} pos%={pos_frac*100:.1f}% neg%={neg_frac*100:.1f}%"
                    )

    # 3. Check initial score and importance value used into deletion
    logger.info(f"score_fn(input_img) iniziale: {float(score_fn(input_img))}")


