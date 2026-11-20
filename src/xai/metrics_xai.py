"""
     Utility Xai 
"""
import csv
import numpy as np
import torch
import torch.nn.functional as F
from utility import to_grid28
from config import CONFIG


def signed_diags(m):
    m = np.asarray(m, dtype=float)

    pos = np.maximum(m, 0)
    neg = np.minimum(m, 0)

    return {
        "positive_fraction": float(np.mean(m > 0)),
        "negative_fraction": float(np.mean(m < 0)),
        "positive_sum": float(pos.sum()),  # somma dei contributi positivi
        "negative_sum": float(neg.sum()),  # somma con segno, quindi <= 0
        "negative_abs_sum": float(np.abs(neg).sum()),  # magnitudine dei contributi negativi
        "net_sum": float(m.sum()),
        "absolute_sum": float(np.abs(m).sum()),
    }

def compute_lr_metrics(lr_map, mask):
    """
         compute metrics for lr
         lr_map and mask have to be strictly 28 x 28
    """
    importance = np.abs(np.asarray(lr_map, dtype=np.float64))
    mask = np.asarray(mask, dtype=bool)

    if importance.shape != mask.shape:
        raise ValueError(f"Shape mismatch: LR map {importance.shape}, mask {mask.shape}")

    mask_frac = float(mask.mean())
    total_importance = float(importance.sum())

    mask_share = (float(importance[mask].sum() / total_importance)
                    if total_importance > 0
                    else 0.0
                )

    focus_ratio = ( mask_share / mask_frac
                    if mask_frac > 0
                        else 0.0
                    )

    return { "mask_share": mask_share,
             "mask_frac": mask_frac,
            "focus_ratio": focus_ratio,
           }

def to_valid_float(x):
    """
        float JSON-safe (nan/inf -> None)

        If the value is:
        - finite: it returns the number;
        - NaN, +Inf, or -Inf: it returns None.
        This is used to avoid invalid numeric value in json results
    """
    x = float(x)
    return x if np.isfinite(x) else None

    
def mean_spearman_agreement(maps,cfg=CONFIG):
    """
        mean pairwise rank correlation between techniques (on |map| at 28x28)

        This function measures the agreement between different XAI techniques.
        Each one(Grad Cam , Lime , Shap , Integrated Gradients , Occlusion ) 
        produces an importance map.

        The question is:
        Do the techniques identify the same pixels as being the most important ?

        Note regarding the code: 
        - ABS so it Consider the magnitude, regardless of the sign and 
        do not distinguish between positive and negative evidence.

        - to_grid28 function : Different techniques can produce maps of varying dimensions.
        Grad-CAM: 4×4 , Integrated Gradients: 128×128 and so on.
        This function map them all onto the same grid, so we can compare them pixel by pixel.

        - np.argsort(np.argsort(map.ravel())) : Transformation into ranks 
        so no longer compare the absolute values of importance,
        but rather their position in the ranking.

        - Mind Attention Note: 
        This metric measures the agreement between techniques, 
        not their faithfulness with respect to the CNN. 
        Two techniques may be in high agreement while both being incorrect.
    """
    # Calling twice np.argsort 
    # we gets the ranking position of the item that we have in this cell?"
    ranks = [np.argsort(np.argsort( 
                         # flat numpy array return from to_grid28
                         # abs to avoid sign impact on this metrics
                         to_grid28(np.abs(r.map), cfg.DENSENET_SIZE).ravel())).astype(float)
                         for r in maps.values()]
    

    if len(ranks) < 2:
        return None
    # Here we Calculate the correlation matrix between the ranks.
    c = np.corrcoef(ranks)

    # Take only the correlations between distinct pairs of techniques 
    # and calculate their average.
    # The interpretation is as follows:
    # Spearman	Interpretation
    # +1	    Perfect agreement
    # 0	        No rank correlation
    # -1	    Perfect inverse agreement
    return to_valid_float(np.nanmean(c[np.triu_indices(len(ranks), 1)]))


@torch.no_grad()
def deletion_curve_for_faithfulness(model, 
                                    input_img, 
                                    target, 
                                    importance, 
                                    fractions, 
                                    baseline ):
    """
       Its goal is to verify whether removing the most important pixels 
       decreases the probability of the explained class.

        So we get the Probability of the target class while the most important pixels
        are replaced with black. 
        Faithful map -> probability drops fast.

        Deletion flow
        1. Original image
   
        The CNN recognizes the digit and assigns a probability to the target class.
        2. Ranking of pixels
        Rank the pixels from most important to least important according to the XAI map.
        
        3. Progressive deletion
        Replace an increasing percentage of the important pixels with a baseline value.
        
        4. New prediction
        Recalculate the probability for the target class.
        
        5. Curve construction
        6. Record the probability obtained for each deletion percentage.

        It requires a baseline with a shape compatible with (1, 3, k), 
        because it replaces selected pixels via:
        x.view(1, 3, -1)[:, :, order[:k]] = baseline

        A faithful map should lead to a more rapid decrease in probability 
        compared to random selection.

    """
    # It recover high and width of the image
    h, w = input_img.shape[-2:]

    # Bring the importance map to the same resolution as the image.
    # The added dimensions [None, None] are used to obtain 
    # the shape required by interpolate [1,1,H,W]
    imp = F.interpolate(torch.as_tensor(np.asarray(importance), dtype=torch.float32)[None, None],
                        size=(h, w), mode="bilinear", 
                        align_corners=False)[0, 0]

    # pixel order
    order = imp.flatten().argsort(descending=True).to(input_img.device)
    probs = []

    # Here we transform the map into a vector and sort the indices 
    # from the most important pixel to the least important.
    for frac in fractions:
        # For each percentage, we create a copy of the original image 
        # and replace the selected pixels.
        x = input_img.clone()
        k = int(round(frac * order.numel()))
        if k > 0:
            # replace selected pixel
            x.view(1, 3, -1)[:, :, order[:k]] = baseline

        # Calculate the probability of the target class after deletion.
        # The result is a list of probabilities, one for each fraction.
        probs.append(float(F.softmax(model(x), dim=1)[0, target]))
    return probs


