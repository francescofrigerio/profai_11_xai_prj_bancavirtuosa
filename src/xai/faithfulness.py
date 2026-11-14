"""
     Utility Xai 
"""
import csv
import numpy as np
import torch
import torch.nn.functional as F
from utility import to_grid28


def _f(x):
    """
        float JSON-safe (nan/inf -> None)
    """
    x = float(x)
    return x if np.isfinite(x) else None

    
def mean_spearman(maps):
    """
        mean pairwise rank correlation between techniques (on |map| at 28x28)
    """
    ranks = [np.argsort(np.argsort(map_to_28(np.abs(r.map)).ravel())).astype(float)
             for r in maps.values()]
    if len(ranks) < 2:
        return None
    c = np.corrcoef(ranks)
    return _f(np.nanmean(c[np.triu_indices(len(ranks), 1)]))





@torch.no_grad()
def deletion_curve(model, input_img, target, importance, fractions, baseline):
    """
        Probability of the target class while the most important pixels
        are replaced with black. Faithful map -> probability drops fast.
    """
    h, w = input_img.shape[-2:]
    imp = F.interpolate(torch.as_tensor(np.asarray(importance), dtype=torch.float32)[None, None],
                        size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    order = imp.flatten().argsort(descending=True).to(input_img.device)
    probs = []
    for frac in fractions:
        x = input_img.clone()
        k = int(round(frac * order.numel()))
        if k > 0:
            x.view(1, 3, -1)[:, :, order[:k]] = baseline
        probs.append(float(F.softmax(model(x), dim=1)[0, target]))
    return probs


def explain_target(model, techniques, input_img, target, mask,
                   fractions, baseline, rng, cfg):
    """
        maps + quantitative metrics for ONE target class
    """
    maps = {name: tech.explain(input_img, target) for name, tech in techniques.items()}

    # random order: reference for the deletion test
    rand_curve = deletion_curve(model, input_img, target,
                                rng.random((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                                fractions, baseline)
    rand_auc = float(np.mean(rand_curve))
    frac_mask = float(mask.mean())

    metrics = {}
    for name, res in maps.items():
        m = np.asarray(res.map)
        g = map_to_28(np.abs(m), cfg.IMAGE_SIZE)
        share = float(g[mask].sum() / max(g.sum(), 1e-12))

        imp = np.maximum(m, 0) if res.signed else m   # evidence FOR the class
        if imp.max() <= 0:
            imp = np.abs(m)
        curve = deletion_curve(model, input_img, target, imp, fractions, baseline)
        auc = float(np.mean(curve))

        metrics[name] = {"mask_share": share,
                         "mask_frac": frac_mask,
                         "focus_ratio": share / max(frac_mask, 1e-12),
                         "del_auc": auc,
                         "del_gain": rand_auc - auc,      # > 0: better than random
                         "del_curve": curve}
    meta = {"agreement_spearman": mean_spearman(maps), "del_auc_random": rand_auc}
    return maps, metrics, meta