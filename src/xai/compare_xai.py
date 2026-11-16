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
from IPython.display import Image, display

from config import CONFIG
from constants import IMAGENET_MEAN_TRANSFORM , IMAGENET_STD_TRANSFORM
from constants import TITLES, GROUPS_RESULT_OK_KO

from utility import digit_mask , img28
from utility import black_normalized

from xai import GradCamCaptum , build_techniques
from xai.shap_explainer import make_background

from xai.faithfulness import explain_target , _float_cast

def plot_maps_grid(rows, target_key, names, cfg, X_data, title, path, logger, max_rows=10):
    sel = [r for r in rows if target_key in r["maps"]][:max_rows]
    if not sel:
        return
    fig = Figure(figsize=(2.1 * (len(names) + 2), 2.1 * len(sel)))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    axes = np.atleast_2d(fig.subplots(len(sel), len(names) + 2))

    for r, row in enumerate(sel):
        axes[r, 0].imshow(img28(row["pos"], X_data, size=cfg.IMAGE_SIZE), cmap="gray")
        axes[r, 0].set_ylabel(f"true {row['true']} | CNN {row['pred']} | LR {row['pred_lr']}\n"
                              f"explained: {row['targets'][target_key]}", fontsize=7)
        cells = [(row["maps"][target_key][n].map, row["maps"][target_key][n].signed)
                 for n in names] + [(row["lr_map"][target_key], True)]
        for c, (m, signed) in enumerate(cells, start=1):
            a = float(np.abs(m).max()) or 1.0
            axes[r, c].imshow(m, cmap="seismic" if signed else "viridis",
                              vmin=-a if signed else 0, vmax=a,
                              interpolation="nearest" if m.shape[0] <= 8 else "bilinear")
        for c in range(len(names) + 2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, t in enumerate(["image"] + [TITLES[n] for n in names] + ["LR local\n(w * x)"]):
        axes[0, c].set_title(t, fontsize=9)

    try:
        fig.savefig(path, dpi=90, bbox_inches="tight")
        try:
            display(Image(str(path)))
        except Exception:
            logger.exception("Impossibile mostrare %s", path)
    finally:
        fig.clear()


def run_compare_group_xai(cfg, logger, model, device, techniques, lr, ctx, group, list_images):
    """
        XAI on one group (quadrant): maps for PRED class (and TRUE class if CNN is wrong).
        This is a function designed for individual groups, and replaces phase_2_xai 
        with a loop that iterates through the four quadrants. 
        For each image, it provides an explanation 
        for the predicted class—including the true class 
        if the CNN’s prediction is incorrect—and then calculates quantitative metrics, 
        saving them in both JSON and CSV formats.

    """
    t0 = time.time()
    out_path = ctx["out_path"]
    dataset = ctx["dataloader_cnn"].dataset
    idx_to_pos = dataset.idx_dataset_position()
    X_data, y_data, y_pred_lr = ctx["X_data"], ctx["y_data"], ctx["y_pred_lr"]

    names = list(techniques)
    fractions = cfg.DELETION_FRACTIONS
    ig_max = cfg.IG_MAX_COMPLETENESS_ERROR
    # baseline = black_baseline(device)
    baseline = black_normalized(cfg.DENSENET_SIZE,device)
    
    rng = np.random.default_rng(cfg.SEED)
    model.eval()

    rows, first_input = [], None
    for id_class in sorted(list_images):
        for item in list_images[id_class]:
            original_idx = item["id_dataset"]
            if original_idx not in idx_to_pos:
                raise KeyError(f"id_dataset {original_idx} not in this split")
            position = idx_to_pos[original_idx]

            input_image, input_label, idx = dataset[position]
            assert idx == original_idx

            # check the target class on record selected
            assert int(input_label) == int(y_data[position])

            input_img = input_image.unsqueeze(0).to(device)
            
            # get the probs with softwmax (multiclass classification)
            with torch.no_grad():
                probs = F.softmax(model(input_img), dim=1)[0]

            # get predicted and true class based on the max prob
            pred, true = int(probs.argmax()), int(input_label)
            if first_input is None:
                first_input = input_img

            targets = { "pred": pred }
            if pred != true:
                targets["true"] = true          # what was missing for the right answer

            mask = digit_mask(img28(position, X_data, size=cfg.IMAGE_SIZE),
                              cfg.MASK_THRESHOLD, cfg.MASK_DILATION)

            row = { "index": original_idx, 
                    "pos": position, 
                    "true": true, 
                    "pred": pred,
                    "pred_lr": int(y_pred_lr[position]), 
                    "prob_pred": float(probs[pred]),
                    "targets": targets, 
                    "maps": {}, 
                    "metrics": {}, 
                    "meta": {}, 
                    "lr_map": {}
                }
            
            for name, tgt in targets.items():
                maps, metrics, meta = explain_target(model, 
                                                     techniques, 
                                                     input_img, 
                                                     tgt,
                                                     mask, 
                                                     fractions, 
                                                     baseline, 
                                                     rng, 
                                                     cfg)
                row["maps"][name] = maps
                row["metrics"][name] = metrics
                row["meta"][name] = meta
                row["lr_map"][name] = lr.local_map(X_data[position], tgt)

            ig = row["maps"]["pred"].get("integrated_gradients")
            if ig is not None:
                row["ig_error"] = _float_cast(ig.meta["completeness_error"])
                row["ig_unreliable"] = bool(ig.meta["completeness_error"] > ig_max)
            rows.append(row)
            logger.info("[%s] idx %d pos %d true %d CNN %d LR %d",
                        group, original_idx, position, true, pred, row["pred_lr"])

    # stochastic techniques: same image explained twice
    stability = {}
    for name in ("lime", "shap"):
        if name in techniques:
            again = techniques[name].explain(first_input, rows[0]["pred"]).map
            diff = float(np.abs(again - rows[0]["maps"]["pred"][name].map).max())
            stability[name] = {"max_abs_diff": diff, "ok": bool(diff <= cfg.RELU_MAX_REF_DIFF)}

    label = GROUPS_RESULT_OK_KO[group]
    plot_maps_grid(rows, "pred", names, cfg, X_data,
                   f"XAI validation [{label}] - explained: CNN PREDICTED class",
                   out_path / f"maps_grid_{group}_pred.png", logger)
    plot_maps_grid(rows, "true", names, cfg, X_data,
                   f"XAI validation [{label}] - explained: TRUE class (CNN wrong)",
                   out_path / f"maps_grid_{group}_true.png", logger)

    # summary and records (csv)
    summary, records = {}, []
    for n in names:
        vals = [r["metrics"]["pred"][n] for r in rows]
        summary[n] = {k: _float_cast(np.mean([v[k] for v in vals]))
                      for k in ("mask_share", "focus_ratio", "del_gain")}
    for r in rows:
        for name, tgt in r["targets"].items():
            for n in names:
                m = r["metrics"][name][n]
                records.append({"group": group, "index": r["index"], "true": r["true"],
                                "pred": r["pred"], "pred_lr": r["pred_lr"],
                                "target": name, "target_class": tgt, "technique": n,
                                "mask_share": m["mask_share"], "mask_frac": m["mask_frac"],
                                "focus_ratio": m["focus_ratio"], "del_auc": m["del_auc"],
                                "del_gain": m["del_gain"],
                                "agreement": r["meta"][name]["agreement_spearman"]})

    results = { "group": group, 
               "label": label,
                "model": {k: ctx["info"].get(k) for k in ("action", "reason", "device")},
                "n_images": len(rows),
                "n_cnn_wrong": sum(r["pred"] != r["true"] for r in rows),
                "stability": stability,
                "summary_pred_target": summary,
                "images": [{"index": r["index"], "true": r["true"], "pred": r["pred"],
                            "pred_lr": r["pred_lr"], "prob_pred": r["prob_pred"],
                            "ig_completeness_error": r.get("ig_error"),
                            "ig_unreliable": r.get("ig_unreliable"),
                            "metrics": r["metrics"], "meta": r["meta"]} for r in rows],
                "total_seconds": time.time() - t0,
            }
    
    (out_path / f"stage2_results_{group}.json").write_text(json.dumps(results, 
                                                                      indent=2, default=str))
    return results, records

def run_compare_group_xai_single_img(cfg, logger, model, device, techniques, lr, ctx, group, list_images):
    """
        XAI on one group (quadrant): maps for PRED class (and TRUE class if CNN is wrong).
        This is a function designed for individual groups, and runs only for
        1 images for group
        We can delete this check to get more simple code.
    """
    t0 = time.time()
    out_path = ctx["out_path"]
    dataset = ctx["dataloader_cnn"].dataset
    idx_to_pos = dataset.idx_dataset_position()
    X_data, y_data, y_pred_lr = ctx["X_data"], ctx["y_data"], ctx["y_pred_lr"]

    names = list(techniques)
    ig_max = cfg.IG_MAX_COMPLETENESS_ERROR
    fractions = cfg.DELETION_FRACTIONS
    
    # baseline = black_baseline(device)
    baseline = black_normalized(cfg.DENSENET_SIZE , device)
    
    rng = np.random.default_rng(cfg.SEED)
    model.eval()

    rows, first_input = [], None
    for id_class in sorted(list_images):
        for item in list_images[id_class]:
            original_idx = item["id_dataset"]
            if original_idx not in idx_to_pos:
                raise KeyError(f"id_dataset {original_idx} not in this split")
            position = idx_to_pos[original_idx]

            input_image, input_label, idx = dataset[position]
            assert idx == original_idx

            # check the target class on record selected
            assert int(input_label) == int(y_data[position])

            input_img = input_image.unsqueeze(0).to(device)
            
            # get the probs with softwmax (multiclass classification)
            with torch.no_grad():
                probs = F.softmax(model(input_img), dim=1)[0]

            # get predicted and true class based on the max prob
            pred, true = int(probs.argmax()), int(input_label)
            if first_input is None:
                first_input = input_img

            targets = { "pred": pred }
            if pred != true:
                targets["true"] = true          # what was missing for the right answer

            mask = digit_mask(img28(position, X_data, size=cfg.IMAGE_SIZE),
                              cfg.MASK_THRESHOLD, cfg.MASK_DILATION)

            row = { "index": original_idx, 
                    "pos": position, 
                    "true": true, 
                    "pred": pred,
                    "pred_lr": int(y_pred_lr[position]), 
                    "prob_pred": float(probs[pred]),
                    "targets": targets, 
                    "maps": {}, 
                    "metrics": {}, 
                    "meta": {}, 
                    "lr_map": {}
                }
            
            for name, tgt in targets.items():
                maps, metrics, meta = explain_target(model, 
                                                     techniques, 
                                                     input_img, 
                                                     tgt,
                                                     mask, 
                                                     fractions, 
                                                     baseline, 
                                                     rng, 
                                                     cfg)
                row["maps"][name] = maps
                row["metrics"][name] = metrics
                row["meta"][name] = meta
                row["lr_map"][name] = lr.local_map(X_data[position], tgt)

            ig = row["maps"]["pred"].get("integrated_gradients")
            if ig is not None:
                row["ig_error"] = _float_cast(ig.meta["completeness_error"])
                row["ig_unreliable"] = bool(ig.meta["completeness_error"] > ig_max)
            rows.append(row)
            logger.info("[%s] idx %d pos %d true %d CNN %d LR %d",
                        group, original_idx, position, true, pred, row["pred_lr"])

    
    # These select and transform the first image
    # already 3x128x128 and normlaized because we begin from transform
    first_idx = rows[0]["pos"]
    first, _ , _ = dataset[first_idx]
    # convert shape from [3,28,28] to [28,28]
    first = first.unsqueeze(0).to(device)
    # Just verify that the recalculation using the copy 
    # of the model produces results sufficiently close 
    # to the original ones.
    # it is a consistency check
    patched = copy.deepcopy(model).eval()

    check_names = ["gradcam", "integrated_gradients"]
    if "saliency" in techniques:
        check_names.append("saliency")

    label = GROUPS_RESULT_OK_KO[group]
    diff_list=[]
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
        diff_list.append((name,diff,diff <= cfg.RELU_MAX_REF_DIFF))

    # save result 
    results = { "group": group, 
                "label": label,
                "model": {k: ctx["info"].get(k) for k in ("action", "reason", "device")},
                "n_images": len(rows),
                "n_cnn_wrong": sum(r["pred"] != r["true"] for r in rows),
                "diff_list": [[name, diff, passed] for name, diff, passed in diff_list], 
                "images": [{"index": r["index"], "true": r["true"], "pred": r["pred"],
                            "pred_lr": r["pred_lr"], "prob_pred": r["prob_pred"],
                            "ig_completeness_error": r.get("ig_error"),
                            "ig_unreliable": r.get("ig_unreliable"),
                            "metrics": r["metrics"], "meta": r["meta"]} for r in rows],
                "total_seconds": time.time() - t0,
            }
    
    
    return results 

