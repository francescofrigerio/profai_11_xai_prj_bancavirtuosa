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
from xai.outputs_xai import plot_maps_grid , plot_deletion_curves

from config import CONFIG
# from constants import IMAGENET_MEAN_TRANSFORM , IMAGENET_STD_TRANSFORM
from constants import TITLES, GROUPS_RESULT_OK_KO

from utility import digit_mask , img28
# from utility import black_normalized , black_baseline
from utility import black_baseline

from xai import GradCamCaptum , build_techniques
from xai.shap_explainer import make_background

from xai.explain import explain_target 
from xai.metrics_xai import to_valid_float



def run_compare_group_xai(cfg, logger, model, device, techniques, lr, ctx, group, list_images):
    """
        The general objective of the code is to compare XAI explanations produced 
        by different techniques, by examining two aspects:
        1. Where are the explanations focused? Do the maps actually highlight 
        the region occupied by the digit?

        2. How faithful(fedele) are they to the model? 
        If I remove the pixels deemed important, does the probability 
        of the explained class decrease?

        The results are maps, metrics, and quantitative results for each image.

        XAI on one group (quadrant): maps for PRED class (and TRUE class if CNN is wrong).
        This is a function designed for individual groups,
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

    names_tech = list(techniques)
    # fractions: percentages of pixels to be removed during the deletion test.
    fractions = cfg.DELETION_FRACTIONS
    
    ig_max = cfg.IG_MAX_COMPLETENESS_ERROR
    
    # baseline: the value that will replace the removed pixels.
    baseline = black_baseline(device)

    # a random number generator with a fixed seed.
    rng = np.random.default_rng(cfg.SEED)
    model.eval()

    rows, first_input = [], None
    for id_class in sorted(list_images):
        for item in list_images[id_class]:
            original_idx = item["id_dataset"]
            if original_idx not in idx_to_pos:
                raise KeyError(f"id_dataset {original_idx} not in this split")
            position = idx_to_pos[original_idx]

            # get X and y for the current image
            input_image, input_label, idx = dataset[position]
            assert idx == original_idx

            # check the target class on record selected
            assert int(input_label) == int(y_data[position])

            # add the batch dimension to the image
            # from (3,H,W) to (1,3,H,W)
            input_img = input_image.unsqueeze(0).to(device)
            
            # The CNN produces logits, 
            # which are transformed into probabilities via Softmax.
            with torch.no_grad():
                # (multiclass classification)
                probs = F.softmax(model(input_img), dim=1)[0]

            # get predicted and real class based on the max prob
            pred, true = int(probs.argmax()), int(input_label)
            if first_input is None:
                first_input = input_img

            # If the CNN is correct, we run an explanation only for the predicted class.         
            # If the CNN is incorrect, we run 2 explanations:
            # - pred: why did the CNN choose the wrong class?
            # - true: which parts of the image support the correct class?
            targets = { "pred": pred }
            if pred != true:
                targets["true"] = true          # what was missing for the right answer

            # The next code builds a binary mask that approximately identifies 
            # the region occupied by the digit.
            # The mask contains:
            # - True: pixels belonging to the digit;
            # - False: pixels belonging to the background.
            # Dilation allows for the inclusion of a small margin around the strokes.
            # This mask will be used to verify whether the XAI maps concentrate 
            # their importance within the region actually occupied by the digit.
            mask = digit_mask(img28(position, X_data, size=cfg.IMAGE_SIZE),
                              cfg.MASK_THRESHOLD, 
                              cfg.MASK_DILATION)

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
                # Run Explanations for all techniques on current image
                # alwais passing pred on targets
                # only for error passing true on targets
                # We get as results:
                # maps	Maps generated by XAI techniques
                # metrics	Quantitative metrics for faithfulness and localization
                # meta	Additional information, such as agreement and random baseline
                maps, metrics, meta = explain_target(model, 
                                                     techniques, 
                                                     input_img, 
                                                     tgt,
                                                     mask, 
                                                     fractions, 
                                                     baseline, 
                                                     rng, 
                                                     cfg)

                plot_deletion_curves(metrics, 
                                     meta, 
                                     fractions, 
                                     out_path / f"deletion_curves_{name}.png",
                                     logger  )

                row["maps"][name] = maps
                row["metrics"][name] = metrics
                row["meta"][name] = meta
                # Also generate the map of the Logistic Regression coefficients 
                # for the same target.
                # This allows us to compare the explanations provided by the CNN 
                # with the linear representation of the LR.
                row["lr_map"][name] = lr.local_map(X_data[position], tgt)

            ig = row["maps"]["pred"].get("integrated_gradients")
            if ig is not None:
                row["ig_error"] = to_valid_float(ig.meta["completeness_error"])
                row["ig_unreliable"] = bool(ig.meta["completeness_error"] > ig_max)
            rows.append(row)
            logger.info("[%s] idx %d pos %d true %d CNN %d LR %d",
                        group, original_idx, position, true, pred, row["pred_lr"])

    # Lime Shap stability of Explanation
    # stochastic techniques: same image explained twice
    # Check whether LIME and SHAP produce the same explanation 
    # when run twice on the same image.
    stability = {}
    for name in ("lime", "shap"):
        if name in techniques:
            again = techniques[name].explain(first_input, rows[0]["pred"]).map
            diff = float(np.abs(again - rows[0]["maps"]["pred"][name].map).max())
            # Of the difference is less than RELU_MAX_REF_DIFF, 
            # the check is considered passed.
            stability[name] = {"max_abs_diff": diff, "ok": bool(diff <= cfg.RELU_MAX_REF_DIFF)}

    #  generates map visualizations;
    label = GROUPS_RESULT_OK_KO[group]
    plot_maps_grid(rows, "pred", names_tech, cfg, X_data,
                   f"XAI validation [{label}] - explained: CNN PREDICTED class",
                   out_path / f"maps_grid_{group}_pred.png", logger)
    
    plot_maps_grid(rows, "true", names_tech, cfg, X_data,
                   f"XAI validation [{label}] - explained: CNN wrong TRUE class ",
                   out_path / f"maps_grid_{group}_true.png", logger)

    summary, records = {}, []
    for name in names_tech:
        # calculates metric averages;
        vals = [r["metrics"]["pred"][name] for r in rows]
        # summary and records (csv)
        summary[name] = {k: to_valid_float(np.mean([v[k] for v in vals]))
                      for k in ("mask_share", "focus_ratio", "del_gain")}
    
    # creates records for the CSV;
    for r in rows:
        for name_pred_true, tgt in r["targets"].items():
            for name in names_tech:
                m = r["metrics"][name_pred_true][name]
                records.append({"group": group, 
                                "index": r["index"], 
                                "true": r["true"],
                                "pred": r["pred"], 
                                "pred_lr": r["pred_lr"],
                                "target": name_pred_true, 
                                "target_class": tgt, 
                                "technique": name,
                                "mask_share": m["mask_share"], 
                                "mask_frac": m["mask_frac"],
                                "focus_ratio": m["focus_ratio"], 
                                "del_auc": m["del_auc"],
                                "del_gain": m["del_gain"],
                                "agreement": r["meta"][name_pred_true]["agreement_spearman"]})

    
    # saves the results in JSON format.
    # The final structure contains information 
    # for each image, target, and XAI technique.
    results = { "group": group, 
               "label": label,
                "model": {k: ctx["info"].get(k) for k in ("action", "reason", "device")},
                "n_images": len(rows),
                "n_cnn_wrong": sum(r["pred"] != r["true"] for r in rows),
                "stability": stability,
                "summary_pred_target": summary,
                "images": [{"index": r["index"], 
                            "true": r["true"], 
                            "pred": r["pred"],
                            "pred_lr": r["pred_lr"], 
                            "prob_pred": r["prob_pred"],
                            "ig_completeness_error": r.get("ig_error"),
                            "ig_unreliable": r.get("ig_unreliable"),
                            "metrics": r["metrics"], "meta": r["meta"]} for r in rows],
                "total_seconds": time.time() - t0,
            }
    
    (out_path / f"stage2_results_{group}.json").write_text(json.dumps(results, 
                                                                      indent=2, default=str))
    return results, records
