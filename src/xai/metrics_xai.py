"""
     Utility Xai 
"""
import csv
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

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





def spearman_agreement_matrix(maps, cfg=CONFIG):
    """
    Calculate the Spearman matrix for the maps produced by the techniques.

    Use abs(map), then compare the ranking of the attribution magnitudes,
    ignoring their signs.

    Returns:
      - techniques: names corresponding to the rows and columns;
      - matrix: an NxN matrix, with None for undefined correlations;
      - mean_off_diagonal: the mean of the correlations between distinct pairs.
    """
    techniques = list(maps)
    n_techniques = len(techniques)

    if n_techniques == 0:
        return { "techniques": [],
                    "matrix": [],
                    "mean_off_diagonal": None,
                }

    ranks = []

    for name in techniques:
        res = maps[name]
        grid = np.asarray(to_grid28(np.abs(np.asarray(res.map)), cfg.DENSENET_SIZE),dtype=float,)

        if grid.ndim != 2:
            raise ValueError(f"La mappa di {name} deve essere 2D, "
                            f"ricevuta shape {grid.shape}"
                            )

        if not np.isfinite(grid).all():
            raise ValueError(f"La mappa di {name} contiene NaN o valori infiniti")

        ranks.append(rankdata(grid.ravel(), method="average"))

    # Diagonal set to 1 to indicate the agreement of a technique with itself.
    matrix = np.full((n_techniques, n_techniques), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)

    for i in range(n_techniques):
        for j in range(i + 1, n_techniques):
            rank_i = ranks[i]
            rank_j = ranks[j]

            # Correlazione non definita se una delle mappe è costante.
            if np.ptp(rank_i) == 0 or np.ptp(rank_j) == 0:
                continue

            corr = float(np.corrcoef(rank_i, rank_j)[0, 1])
            matrix[i, j] = corr
            matrix[j, i] = corr

    upper_values = matrix[np.triu_indices(n_techniques, k=1)]
    finite_values = upper_values[np.isfinite(upper_values)]

    mean_off_diagonal = ( float(finite_values.mean()) if finite_values.size else None )

    # None is better for JSON and remain readable
    # as not defined correlation
    matrix_json = [ [float(value) if np.isfinite(value) else None for value in row]
                    for row in matrix
                  ]

    return { "techniques": techniques,
             "matrix": matrix_json,
             "mean_off_diagonal": mean_off_diagonal,
            }

def spearman_agreement_mean(maps,cfg=CONFIG):
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



        Here, each map is flattened into a 784-pixel vector in order to compare their ranks.

        For a 5 x 5 average matrix per group, the correct workflow is: 
        calculate a correlation matrix for each image, 
        then compute the element-wise average across the images. 
        The dimensions should be determined by the number of selected techniques—hence, 
        with five techniques, it will be 5x5. Allowing for an arbitrary, 
        user-defined dimension could result in a matrix that is inconsistent 
        with the actual techniques being compared.

        Furthermore, there is a statistical caveat: argsort(argsort(...))
        assigns distinct ranks even to pixels with the same level of importance. 
        Consequently, maps containing many identical values 
        may end up with arbitrary ranks. 
        For Spearman correlation, it is preferable to assign average ranks 
        to tied values—for instance, by using scipy.stats.rankdata 
        and to explicitly handle constant maps, for which the correlation may be undefined.
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
        x.view(1, 3, -1)[:, :, order[:k]] = baseline KO

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
            # x.view(1, 3, -1)[:, :, order[:k]] = baseline
            # each selected pixel is replaced with the corresponding pixel 
            # from the blurred baseline.
            x_flat = x.reshape(1, 3, -1)
            baseline_flat = baseline.to(input_img.device).reshape(1, 3, -1)
            x_flat[:, :, order[:k]] = baseline_flat[:, :, order[:k]]

        # Calculate the probability of the target class after deletion.
        # The result is a list of probabilities, one for each fraction.
        # probs.append(float(F.softmax(model(x), dim=1)[0, target]))
        probs.append(float(F.softmax(model(x_flat), dim=1)[0, target]))
    return probs



@torch.no_grad()
def deletion_curve_for_faithfulness_2( score_fn,
                                        input_img,
                                        importance,
                                        fractions,
                                        baseline,
                                    ):
    """
    score_fn(x) return the target prob for input x.
    input_img and baseline has shape [1, C, H, W].
    importance is a 2D map.
    """
    if input_img.ndim != 4 or input_img.shape[0] != 1:
        raise ValueError("input_img deve avere forma [1, C, H, W]")

    if baseline.shape != input_img.shape:
        raise ValueError(
            f"baseline e input_img devono avere la stessa forma: "
            f"{baseline.shape} != {input_img.shape}"
        )

    h, w = input_img.shape[-2:]

    imp = torch.as_tensor(importance, dtype=torch.float32, device=input_img.device).squeeze()

    if imp.ndim != 2:
        raise ValueError(f"importance deve essere 2D, ricevuta shape {tuple(imp.shape)}")

    imp = F.interpolate(imp[None, None],
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]

    order = imp.flatten().argsort(descending=True)
    baseline = baseline.to(device=input_img.device, dtype=input_img.dtype)

    probs = []
    for frac in fractions:
        if not 0 <= frac <= 1:
            raise ValueError(f"Frazione di deletion non valida: {frac}")

        x = input_img.clone()
        k = int(round(frac * order.numel()))

        # if k > 0:
        #    x_flat = x.reshape(1, x.shape[1], -1)
        #    baseline_flat = baseline.reshape(1, baseline.shape[1], -1)
        #    selected = order[:k]
        #    x_flat[:, :, selected] = baseline_flat[:, :, selected]

        # x that has a shape [1, C, H, W], not x_flat.
        #probs.append(float(score_fn(x)))
        x = input_img.clone()
        # from input img t0 CxHxW
        x_flat = x.reshape(1, x.shape[1], -1)
        baseline_flat = baseline.reshape(1, baseline.shape[1], -1)

        k = int(round(frac * order.numel()))
        if k > 0:
            selected = order[:k]
            x_flat[:, :, selected] = baseline_flat[:, :, selected]

         # x has to have a shape [1, C, H, W]
        x_for_model = x_flat.reshape_as(input_img)
        probs.append(float(score_fn(x_for_model)))

    return probs


def deletion_metrics_from_maps(score_fn,
                                input_img,
                                maps_for_target,
                                fractions,
                                baseline,
                                rng,
                            ):
    """
    maps_for_target:
      CNN -> {name_tecnh: res}, res has .map and .signed
      LR  -> {"lr_local_map": matrix_numpy}
    """
    fractions = np.asarray(fractions, dtype=float)

    if fractions.ndim != 1 or len(fractions) < 2:
        raise ValueError("Servono almeno due frazioni per calcolare l'AUC")
    if np.any(np.diff(fractions) < 0):
        raise ValueError("Le frazioni devono essere in ordine crescente")

    h, w = input_img.shape[-2:]
    random_map = rng.random((h, w))

    random_curve = deletion_curve_for_faithfulness_2(score_fn, 
                                                     input_img, 
                                                     random_map, 
                                                     fractions, 
                                                     baseline
                                                    )
    
    random_auc = float(np.trapz(random_curve, x=fractions))

    metrics = {}
    for name, item in maps_for_target.items():
        if hasattr(item, "map"):       # result explainer CNN
            m = np.asarray(item.map)
            signed = bool(item.signed)
        else:                          # maps LR
            m = np.asarray(item)
            signed = True

        # order select evidece for target.
        # Ig there isn't, use the maps magnitudine .
        importance = np.maximum(m, 0) if signed else m
        if not np.any(importance > 0):
            importance = np.abs(m)

        curve = deletion_curve_for_faithfulness_2(score_fn, 
                                                  input_img, 
                                                  importance, 
                                                  fractions, 
                                                  baseline
                                                )
        auc = float(np.trapz(curve, x=fractions))

        metrics[name] = {"del_auc": auc,
                        "del_gain": random_auc - auc,
                        "del_curve": curve,
                        }

    meta = { "del_auc_random": random_auc,
                "del_curve_random": random_curve,
                "fractions": fractions.tolist(),
            }
    return metrics, meta



