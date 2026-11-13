"""
     This module implement run function with Xai code
"""

# import os
import json
import time
import copy

import torch
import torch.nn.functional as F

import numpy as np
from matplotlib.figure import Figure
# from matplotlib.pyplot import plt
from IPython.display import display

from scipy.ndimage import binary_dilation
from IPython.display import Image, display

from config import TITLES
from config import IMAGENET_MEAN_TRANSFORM
from config import IMAGENET_STD_TRANSFORM

from utility import to_densenet_input
from xai import GradCamCaptum , build_techniques
from xai.shap_explainer import make_background


def digit_mask(input_img28, threshold, dilation):
    """
        This function creates a binary mask of the image.

        It compares each pixel with a threshold:
        - pixel above the threshold: True;
        - pixel at or below the threshold: False.
        In this way, it approximately identifies 
        the pixels belonging to the digit,
        and so identify the area occupied by the digit

        In the short:
        This function is used to distinguish the digit area 
        from the background, potentially including 
        a small margin around the strokes(tratti).

        Limits:THRESHOLD must be consistent 
        with the pixel values in X_train. 
        If the images are normalized or standardized, 
        the threshold must be chosen accordingly.
    """
    mask = np.asarray(input_img28) > threshold

    # expands the mask, slightly enlarging the identified area.
    # The dilation parameter determines how many times 
    # this expansion is performed. If it is set to zero, 
    # the mask remains unchanged.
    return binary_dilation(mask, 
                           structure=np.ones((3, 3), bool), 
                           iterations=dilation
                           ) if dilation else mask



def img28(pos,X_data,size ):
    """
        flat vector (784,) -> image (28, 28) for mask and imshow
    """
    return X_data[pos].reshape(size, size)

def to_grid28(map2d, size, side=28):
    """
        This Function convert XAI 2dmap 
        to mnist risolution 28x28 to compare
        xai map with the digit mask

        This is the step:
    """

    # - Convert map2d to tensor and Adds two dimensions—batch 
    # and channel—resulting 
    # in the shape (1, 1, H, W).
    t = torch.as_tensor(np.asarray(map2d), 
                        dtype=torch.float32)[None, None]
    if t.shape[-1] < size:
        # If the map is smaller than the specified size, 
        # it scales it up to size × size.
        t = F.interpolate(t, 
                          size=(size, size), 
                          mode="bilinear", 
                          align_corners=False)

    # Resizes it to 28 × 28 using area interpolation.
    # and removes the batch and channel dimensions 
    # and returns a NumPy array.  
    return F.interpolate(t, 
                         size=(side, side), 
                         mode="area")[0, 0].numpy()

def make_gradcam_callback(cfg,dataloader, device, out_path , list_images , logger):
    """
        Grad-CAM debug images of the first and last training epoch.
        The callback explains the PREDICTED class of the 10 fixed validation images 
        (one per class), so a
        broken training (maps that do not move, or that look at the background) 
        is visible at a glance.

        Mind the difference between idx_dataset and idx position of image.
        It’s not about the shuffle. In data.load_dataset function, 
        the validation loader already has shuffle=False (only the training loader has 
        shuffle=True). 
        Furthermore, dataloader.dataset[k] is unaffected by the shuffle, 
        which only changes the order in which the DataLoader iterates through the data.

        The issue lies with the Subset :
        TransformedDataset.__getitem__(idx)interprets idx as a position within the 
        Subset` (i.e., `self.dataset[idx]).
        However, the third element it returns—which ends up in the JSON files as
        id_dataset —is the original index from trainval_base (original_idx).

        For the test set, the two indices coincide (original_indices=None), so it works. 
        For the validation set, they do not: the callback performs `dataset[idx_dataset]
        using an original index (e.g., 48213) on a dataset that is only 6,000 items long;
        consequently, it either retrieves the wrong image or triggers an IndexError.

        The solution is to convert the original index into its corresponding position
        which is precisely why we created idx_dataset_position
    """
    dataset = dataloader.dataset
    # built once: original idx (saved in the json) -> position in the dataset
    idx_to_pos = dataset.idx_dataset_position()

    def callback(model, epoch):
        logger.debug("Start Callback Grad-CAM epoch:%d",epoch)
        model.eval()
        explainer = GradCamCaptum(model, cfg)

        n_images = len(list_images)
        fig = Figure(figsize=(2 * n_images, 4.4))
        axes = fig.subplots(2, n_images)
        for i, id_class in enumerate(sorted(list_images.keys())):
            # load_images_by_class return a list not a dict 
            # so we add [0]    
            item = list_images[id_class][0]
            original_idx = item["id_dataset"]
               
            logger.debug("before read dataset on: %s", original_idx)
            logger.debug("before read dataset on: %s", type(dataloader))
            
            # Dataset: allows access to an element using dataset[idx].
            # DataLoader: returns batches of elements when iterated over.
            # dataset = dataloader.dataset
            # input_image, input_label, idx = dataset[idx_dataset]
            if original_idx not in idx_to_pos:
                raise KeyError(
                    f"id_dataset {original_idx} not found in this dataset: "
                    "json list and dataloader must use the same split (val/test)")
            position = idx_to_pos[original_idx]

            input_image, input_label, idx = dataset[position]
            # This alerts if the JSON list and the dataloader 
            # do not belong to the same split.
            assert idx == original_idx   # sanity check

            # Add batch dimension: (3, 128, 128) -> (1, 3, 128, 128)
            input_img = input_image.unsqueeze(0).to(device)
            logger.debug("input_image:%s",input_img.shape)
               
            with torch.no_grad():
                # pred = int(model(input_img).argmax(1))
                pred = int(model(input_img).argmax(dim=1).item())

            # Thus, in the case of a 3 classified as an 8, the map highlights 
            # the regions that support the prediction of an 8. 
            # It does not necessarily show the regions that would 
            # have allowed for the correct identification of the 3.
            # Furthermore,  the brighter areas indicate greater activation relevant 
            # to the chosen class—not a precise segmentation of the digit, 
            # nor causal evidence of the decision-making process.
            result = explainer.explain(input_img, pred)
            
            # Original Image
            # axes[0, i].imshow(input_image.squeeze(0),cmap="gray")
            # Reverse the ImageNet normalization 
            # before displaying the image. 
            # We don't should display the normalized tensor directly, 
            # as the pixel scale will not correspond to that of a normal image.
            mean = torch.tensor(IMAGENET_MEAN_TRANSFORM,
                                device=input_image.device).view(3, 1, 1)

            std = torch.tensor(IMAGENET_STD_TRANSFORM,
                                device=input_image.device ).view(3, 1, 1)

            image_display = (input_image * std + mean).clamp(0, 1)

            # Display original grayscale image
            axes[0, i].imshow(image_display.mean(dim=0).cpu().numpy(),cmap="gray")
            axes[0, i].set_title(f"true {input_label} / pred {pred}",fontsize=8)
               
            # Grad-CAM
            # pylint: disable=no-member
            axes[1, i].imshow(result.map, cmap="viridis", interpolation="bilinear")
               
            for row in (0, 1):
                axes[row, i].axis("off")
           
        fig.suptitle(f"Grad-CAM debug On Train, epoch {epoch+1}")
        out_path.mkdir(parents=True, exist_ok=True)
        
       
        path_png = out_path / f"gradcam_debug_epoch{epoch}.png"
        try:
            fig.savefig(path_png, dpi=100, bbox_inches="tight")
        
            try:
                # display(fig)
                display(Image(str(path_png)))
            except Exception:
                logger.exception("Impossibile mostrare %s", path_png)
        
        finally:
            # Figure delete refernce without pyplot
            fig.clear()
            del fig, axes

        logger.debug("End Callback Grad-CAM saved: %s",path_png)

    return callback

# disable=too-many-arguments, too-many-positional-arguments
def run_exp( cfg, logger, model, lr, device , xai_param_dict) -> None :   
    """
    This function implements Hook that are a tool 
    for observing the internal 
    workings of a network during processing.
    
    A hook is a function that allows you to attach 
    to a specific point during the execution 
    of a PyTorch model, enabling you to intercept 
    and observe information—such as during the 
    forward or backward pass—without 
    having to directly modify the model's code.

    In the context of XAI (Explainable AI), 
    hooks are primarily used to retrieve layer 
    activations and their corresponding gradients, 
    which are necessary for calculating explanations.
    """
    
    rows, ig_errors = [], []

    t_start = xai_param_dict["t_start"]
    X_data = xai_param_dict["X_data"]
    y_data = xai_param_dict["y_data"]
    info = xai_param_dict["info"]
    list_images = xai_param_dict["list_img"]
    out_path = xai_param_dict["out_path"]
    transform_cnn = xai_param_dict["transform_cnn"] 
    dataloader_cnn = xai_param_dict["dataloader_cnn"]
    flag_error_pred_cnn = xai_param_dict["flag_error_pred_cnn"]

    dataset = dataloader_cnn.dataset
    # original idx (saved in the json) -> position in the dataset
    idx_to_pos = dataset.idx_dataset_position()
   
    # pass all dataset train , already processed and normalized
    # make_background don't need train_idx parameter
    background = make_background(transform_cnn,cfg,logger, device)
    techniques = build_techniques(model, cfg, background)

    for i, id_class in enumerate(sorted(list_images.keys())):
        logger.debug("Start Loop %d %d " , i ,id_class)
        # load_images_by_class return a list not a dict 
        # so we add [0]    
        item = list_images[id_class][0]
        original_idx = item["id_dataset"]
                       
        logger.debug("before read dataset on: %s", original_idx)
        logger.debug("before read dataset on: %s", type(dataloader_cnn))
                    
        # Dataset: allows access to an element using dataset[idx].
        # DataLoader: returns batches of elements when iterated over.
        # dataset = dataloader.dataset
        # input_image, input_label, idx = dataset[idx_dataset]
        if original_idx not in idx_to_pos:
            raise KeyError(f"id_dataset {original_idx} not found in this dataset: "
                            "json list and dataloader must use the same split (val/test)")
        position = idx_to_pos[original_idx]
        
        input_image, input_label, idx = dataset[position]
        # This alerts if the JSON list and the dataloader 
        # do not belong to the same split.
        assert idx == original_idx   # sanity check
        logger.debug("assert original_idx :%d",original_idx)

        assert int(input_label) == int(y_data[position])
        logger.debug("assert position :%d",position)

        # assert idx_to_pos[position] == original_idx  
        # Add batch dimension: (3, 128, 128) -> (1, 3, 128, 128)
        input_img = input_image.unsqueeze(0).to(device)
        logger.debug("input_image:%s",input_img.shape)
                       
        with torch.no_grad():
            # pred = int(model(input_img).argmax(1))
            pred = int(model(input_img).argmax(dim=1).item())

        maps = {}
        for name, technique in techniques.items():
            logger.debug(f"Start Xai {name} {technique}")
            result = technique.explain(input_img, pred)
            maps[name] = result
            
            if name == "integrated_gradients":
                ig_errors.append(result.meta["completeness_error"])
            logger.debug(f"End Xai {name} {technique}")

        # substiutr i with idx_dataset
        # original_idx idc original (log/json)
        # dataset position X_data[idx]
        rows.append({"index": original_idx, 
                     "pos": position,
                     "true": int(y_data[position]), 
                     "pred": pred, 
                     "maps": maps,
                     # local_map runs reshape(28,28)
                     "lr_map": lr.local_map(X_data[position], pred)})
    
        logger.debug(f"End Loop {i} image {original_idx} {position} (true {y_data[position]}, predicted {pred}): ")

    # This selects the first index to
    # compare explanation getting from 
    # this first with that already in rows[0] 
    first_idx = rows[0]["pos"]
    
    # These select and transform the first image
    # already 3x128x128 and normlaized because we begin from transform
    first, _ , _ = dataset[first_idx]
    # convert shape from [3,28,28] to [28,28]
    first = first.unsqueeze(0).to(device)
    
    # LIME and SHAP can have stochastic components: 
    # a new execution might produce a slightly different map.                         
    for name in ("lime", "shap"):
        logger.debug(f"Start Again Calculate Xai {name} {techniques[name]}")
        # This gets the new calculated map;
        again = techniques[name].explain(first, rows[0]["pred"]).map
        # This gest the previously saved calculated map, 
        ref = rows[0]["maps"][name].map

        diff = float(np.abs(again - ref).max())
        logger.debug("%s check difference: %s, max difference %.2e",name, diff <= cfg.RELU_MAX_REF_DIFF, diff)     
        logger.debug(f"End Again Calculate Xai {name} {techniques[name]}")
       
    # This model's independent copy is used to perform checks 
    # without affecting the original trained model
    patched = copy.deepcopy(model).eval()

    # Saliency Map Normalization (Min-Max [0, 1]):
    # in order to make the maps generated 
    # by different techniques
    # homogeneous and directly comparable 
    # (Grad-CAM and Integrated Gradients
    #     can produce raw attribution values 
    #     on very different scales).

    # Just verify that the recalculation using the copy 
    # of the model produces results sufficiently close 
    # to the original ones.
    # it is a consistency check
    # We can cancel this test to have a more simple code
    check_names = ["gradcam", "integrated_gradients"]

    if "saliency" in techniques:
        check_names.append("saliency")

    for name in check_names:
        # getting the map calculate in the main loop
        a = rows[0]["maps"][name].map
        # getting the map recalculated on the same image, 
        # using the copied model.
        b = build_techniques(patched, cfg, names=(name,))[name].explain(first, rows[0]["pred"]).map
        
        # it measures the maximum absolute difference 
        # between the two maps, normalized by the 
        # maximum absolute value of the second map.
        diff = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
        logger.debug("%s check on train: %s, relative difference %.2e",name, diff <= cfg.RELU_MAX_REF_DIFF, diff)

    shares = []

    # Here we calculate two values
    # 1) It represents the proportion 
    # of the total saliency that falls within the digit mask.
    # If first value = 0.85, 
    # it would mean that 85% of the total saliency 
    # under consideration falls within the area 
    # identified as the digit.

    # 2) It represents the fraction of the 
    # image occupied by the mask space digit.
    # shares contains a pair of indicators for each image, 
    # rather than the maps themselves.
    shares = []
    if "saliency" in techniques:
        for row in rows:
            # mask = digit_mask(X_data[row["pos"], 0].numpy(), cfg.MASK_THRESHOLD, cfg.MASK_DILATION)
            mask = digit_mask(img28(row["pos"],X_data,size=cfg.IMAGE_SIZE), 
                          cfg.MASK_THRESHOLD, 
                          cfg.MASK_DILATION)
    
            g = to_grid28(np.abs(row["maps"]["saliency"].map), 
                           cfg.DENSENET_SIZE)

            share = float(g[mask].sum() / max(g.sum(), 1e-12))
            shares.append((share,float(mask.mean())))
        
    # figure plot for all techniques
    names = list(techniques)
    fig = Figure(figsize=(2.1 * (len(names) + 2), 2.1 * len(rows)))

    fig.suptitle("XAI - IMMAGINI PREDETTE CON ERRORE DA DENSENET" 
                 if flag_error_pred_cnn else "XAI - IMMAGINI PREDETTE CON ERRORE DA LR",
                fontsize=14,
                fontweight="bold",
                y=1.02
                )
    
    axes = fig.subplots(len(rows), len(names) + 2)
    for r, row in enumerate(rows):
        # original image
        # axes[r, 0].imshow(X_data[row["pos"], 0], cmap="gray")
        axes[r, 0].imshow(img28(row["pos"],X_data,size=cfg.IMAGE_SIZE), cmap="gray")

        axes[r, 0].set_ylabel(f"true {row['true']}\npred {row['pred']}", fontsize=8)
        # Prepare the cell's maps for visualization:
        # col1 original map
        # col2 for each XAI technique;
        # col3 for the local map of the Logistic Regression
        cells = [(row["maps"][n].map, 
                  row["maps"][n].signed) for n in names] + [(row["lr_map"], True)]

        for c, (m, signed) in enumerate(cells, start=1):
            # Graphical normalization is performed using 
            # the maximum absolute value of each individual map. 
            # This makes the maps easier to read, 
            # but be careful: it does not mean that 
            # the colors can be directly compared 
            # in quantitative terms across different techniques.
            a = float(np.abs(m).max()) or 1.0
            
            # seismic for maps with both positive and negative value
            # viridis for those that are non-negative.
            axes[r, c].imshow(m, cmap="seismic" if signed else "viridis", 
                              vmin=-a if signed else 0, 
                              vmax=a,
                              interpolation="nearest" if m.shape[0] <= 8 else "bilinear")
            
        for c in range(len(names) + 2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, title in enumerate(["image"] + [TITLES[n] for n in names] + ["LR local\n(w * x)"]):
        axes[0, c].set_title(title, fontsize=9)

    if flag_error_pred_cnn:
        path_png = out_path / "maps_grid_ko_cnn.png"
    else:
        path_png = out_path / "maps_grid_ko_lr.png"

    try:
        fig.savefig(path_png, dpi=90, bbox_inches="tight")

        try:
            display(Image(str(path_png)))
        except Exception:
            logger.exception("Impossibile mostrare %s", path_png)

    finally:
        # Figure delete refernce without pyplot
        fig.clear()
        del fig, axes

    # save result to json file
    results = {
        "model": {k: info[k] for k in ("action", "reason", "device")},
        "ig_completeness_errors": ig_errors,
        "images": [{"index": r["index"], "true": r["true"], "pred": r["pred"],
                    "native_shapes": {n: list(r["maps"][n].map.shape) 
                                      for n in names}} for r in rows],
        "total_seconds": time.time() - t_start,
    }
    (out_path / "stage2_results.json").write_text(json.dumps(results, indent=2))
    
    logger.info(f"All tecniques run correctly. Total time {time.time() - t_start:.0f}s")
    return results
