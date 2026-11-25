"""
     This module implement run function with Xai code
"""
# import os
import json
import time
import copy
import pandas as pd
import torch
import torch.nn.functional as F

import numpy as np
from matplotlib.figure import Figure
from IPython.display import Image, display
from xai.outputs_xai import plot_maps_grid , plot_deletion_curves ,plot_maps_grid_dinamic
from xai.outputs_xai import format_json_xai_rows , inspect_digit_masks

from config import CONFIG
# from constants import IMAGENET_MEAN_TRANSFORM , IMAGENET_STD_TRANSFORM
from constants import TITLES, GROUPS_RESULT_OK_KO

from utility import digit_mask , img28
# from utility import black_normalized , black_baseline
from utility import make_blurred_baseline , black_baseline

from xai import GradCamCaptum , build_techniques
from xai.shap_explainer import make_background

from xai.explain import explain_target , explain_target_1 , explain_target_2 
from xai.metrics_xai import to_valid_float , compute_lr_metrics , signed_diags
from xai.metrics_xai import spearman_agreement_matrix , compute_topk_iou

from scipy.ndimage import distance_transform_edt
# install with pip
from skimage.morphology import skeletonize

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
            pred_cnn, true = int(probs.argmax()), int(input_label)
            pred_lr = int(y_pred_lr[position])

            if first_input is None:
                first_input = input_img

            # If the CNN is correct, we run an explanation only for the predicted class.         
            # If the CNN is incorrect, we run 2 explanations:
            # - pred_cnn: why did the CNN choose the wrong class?
            # - pred_lr: why did the LR choose the wrong class?
            # - true: which parts of the image support the correct class?
            targets_cnn = { "pred_cnn": pred_cnn }
            targets_lr = {"pred_lr": pred_lr}

            if pred_cnn != true or pred_lr != true:
                # what was missing for the right answer
                targets_cnn["true"] = true  
                targets_lr["true"] = true        

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
                    "pred_cnn": pred_cnn,
                    "pred_lr": pred_lr, 
                    "prob_pred": float(probs[pred_cnn]),
                    "targets_cnn": targets_cnn, 
                    "targets_lr": targets_lr, 
                    "mask_mean": mask.mean(), 
                    "maps_cnn": {}, 
                    "metrics_cnn": {}, 
                    "meta_cnn": {}, 
                    "maps_lr": {}, 
                    "metrics_lr": {},               
                    "meta_lr": {}, 
                    "lr_map": {}
                }

            # for each target in cnn_targets, 
            # it generate maps using XAI techniques applied to the CNN;
            for name, tgt in targets_cnn.items():
                # Run Explanations for all techniques on current image
                # alwais passing pred on targets
                # only for error passing true on targets
                # We get as results:
                # maps	Maps generated by XAI techniques
                # metrics	Quantitative metrics for faithfulness and localization
                # meta	Additional information, such as agreement and random baseline
                maps_cnn, metrics_cnn, meta_cnn = explain_target(model, 
                                                     techniques, 
                                                     input_img, 
                                                     tgt,
                                                     mask, 
                                                     fractions, 
                                                     baseline, 
                                                     rng, 
                                                     cfg)

            # for each target in lr_targets, 
            # it generate an LR map using lr.local_map(...).
            for name, tgt in targets_lr.items():
                # Run Explanations for all techniques on current image
                # alwais passing pred on targets
                # only for error passing true on targets
                # We get as results:
                # maps	Maps generated by XAI techniques
                # metrics	Quantitative metrics for faithfulness and localization
                # meta	Additional information, such as agreement and random baseline
                maps_lr, metrics_lr, meta_lr = explain_target(lr, 
                                                        techniques, 
                                                        input_img, 
                                                        tgt,
                                                        mask, 
                                                        fractions, 
                                                        baseline, 
                                                        rng, 
                                                        cfg)
                                
            # For instance, if the CNN predicts 5 and the LR predicts 8, 
            # you would obtain CNN maps for 5 and 2, and LR maps for 8 and 2. 
            # You do not calculate a CNN map for 8—that would constitute 
            # an additional analysis of the CNN for a target it did not predict, 
            # rather than an explanation of its actual decision.

            plot_deletion_curves(metrics_cnn, 
                                     meta_cnn, 
                                     fractions, 
                                     out_path / f"deletion_curves_{name}_cnn.png",
                                     logger  )

            row["maps_cnn"][name] = maps_cnn
            row["metrics_cnn"][name] = metrics_cnn
            row["meta_cnn"][name] = meta_cnn

            row["maps_lr"][name] = maps_lr
            row["metrics_lr"][name] = metrics_lr
            row["meta_lr"][name] = meta_lr
            # Also generate the map of the Logistic Regression coefficients 
            # for the same target.
            # This allows us to compare the explanations provided by the CNN 
            # with the linear representation of the LR.
            row["lr_map"][name] = lr.local_map(X_data[position], tgt)

            # completeness_error ony for cnn: This is a specific check for Integrated Gradients, 
            # not a general metric applicable to all maps. 
            # It verifies how closely the sum of the attributions approaches the difference 
            # between the model’s output for the image and its output for the chosen baseline. 
            # It does not measure 
            # how “correct” the map is, nor the extent to which it aligns with LR.
            ig = row["maps_cnn"]["pred_cnn"].get("integrated_gradients")
            if ig is not None:
                row["ig_error"] = to_valid_float(ig.meta["completeness_error"])
                row["ig_unreliable"] = bool(ig.meta["completeness_error"] > ig_max)
            rows.append(row)
            logger.info("[%s] idx %d pos %d true %d CNN %d LR %d",
                        group, original_idx, position, true, pred_cnn,pred_lr)

    # Lime Shap stability of Explanation
    # stochastic techniques: same image explained twice
    # Check whether LIME and SHAP produce the same explanation 
    # when run twice on the same image.
    stability_cnn = {} 
    for name in ("lime", "shap"):
        if name in techniques:
            again = techniques[name].explain(first_input, rows[0]["pred_cnn"]).map
            diff = float(np.abs(again - rows[0]["maps_cnn"]["pred_cnn"][name].map).max())
            # Of the difference is less than RELU_MAX_REF_DIFF, 
            # the check is considered passed.
            stability_cnn[name] = {"max_abs_diff": diff, "ok": bool(diff <= cfg.RELU_MAX_REF_DIFF)}

    #  generates map visualizations;
    label = GROUPS_RESULT_OK_KO[group]
    plot_maps_grid(rows, "pred_cnn", names_tech, cfg, X_data,
                   f"XAI validation [{label}] - explained: CNN PREDICTED class",
                   out_path / f"maps_grid_{group}_pred_cnn.png", logger)

    plot_maps_grid(rows, "pred_lr", names_tech, cfg, X_data,
                       f"XAI validation [{label}] - explained: LR PREDICTED class",
                       out_path / f"maps_grid_{group}_pred_lr.png", logger)
    
    plot_maps_grid(rows, "true", names_tech, cfg, X_data,
                   f"XAI validation [{label}] - explained: CNN wrong TRUE class ",
                   out_path / f"maps_grid_{group}_true.png", logger)

    summary_cnn,summary_lr, records = {},{}, []
    for name in names_tech:
        # calculates metric averages;
        vals_cnn = [r["metrics_cnn"]["pred_cnn"][name] for r in rows]
        vals_lr = [r["metrics_lr"]["pred_lr"][name] for r in rows]
        # summary and records (csv)
        summary_cnn[name] = {k: to_valid_float(np.mean([v[k] for v in vals_cnn]))
                      for k in ("mask_share", "focus_ratio", "del_gain")}
        summary_lr[name] = {k: to_valid_float(np.mean([v[k] for v in vals_lr]))
                              for k in ("mask_share", "focus_ratio", "del_gain")}
    
    # creates records for the CSV;
    for r in rows:
        for name_pred_true, tgt in r["targets_cnn"].items():
            for name in names_tech:
                m = r["metrics_cnn"][name_pred_true][name]
                records.append({"group": group, 
                                "index": r["index"], 
                                "true": r["true"],
                                "pred": r["pred_cnn"], 
                                "pred_lr": r["pred_lr"],
                                "mask_mean": r["mask_mean"],
                                "target": name_pred_true, 
                                "target_class": tgt, 
                                "technique": name,
                                "mask_share": m["mask_share"], 
                                "mask_frac": m["mask_frac"],
                                "focus_ratio": m["focus_ratio"], 
                                "del_auc": m["del_auc"],
                                "del_gain": m["del_gain"],
                                "agreement": r["meta"][name_pred_true]["agreement_spearman"]})
        
        for name_pred_true, tgt in r["targets_lr"].items():
            for name in names_tech:
                m = r["metrics_lr"][name_pred_true][name]
                records.append({"group": group, 
                                "index": r["index"], 
                                "true": r["true"],
                                "pred": r["pred_cnn"], 
                                "pred_lr": r["pred_lr"],
                                "mask_mean": r["mask_mean"],
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
                "n_cnn_wrong": sum(r["pred_cnn"] != r["true"] for r in rows),
                "stability_cnn": stability_cnn,
                "summary_pred_target_cnn": summary_cnn,
                "n_cnn_wrong": sum(r["pred_cnn"] != r["true"] for r in rows),
                "n_lr_wrong": sum(r["pred_lr"] != r["true"] for r in rows),
                "summary_pred_target_lr": summary_lr,
                "images": [{"index": r["index"], 
                            "true": r["true"], 
                            "pred_cnn": r["pred_cnn"],
                            "pred_lr": r["pred_lr"], 
                            "prob_pred": r["prob_pred"],
                            "ig_completeness_error": r.get("ig_error"),
                            "ig_unreliable": r.get("ig_unreliable"),
                            "metrics_cnn": r["metrics_cnn"], 
                            "meta_cnn": r["meta_cnn"],
                            "metrics_lr": r["metrics_lr"], 
                            "meta_lr": r["meta_lr"] 
                            } for r in rows],
                            "total_seconds": time.time() - t0
                        }
    
    (out_path / f"stage2_results_{group}.json").write_text(json.dumps(results, 
                                                                      indent=2, default=str))
    return results, records


def pre_compare_group_xai_1 (cfg, ctx ):
    """
        analysis before run_compare_xai
    """
    # out_path = ctx["out_path"]
    # dataset = ctx["dataloader_cnn"].dataset
    # idx_to_pos = dataset.idx_dataset_position()
    X_data, y_data, = ctx["X_data"], ctx["y_data"]
    # names_tech = list(techniques)
     
    rng = np.random.default_rng(42)
    positions = []
    
    for digit in range(10):
        candidates = np.flatnonzero(np.asarray(y_data) == digit)
        chosen = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
        positions.extend(chosen.tolist())
    
    inspect_digit_masks(positions, X_data, cfg)

def stroke_thickness_score(position, X_data, cfg):
    """
         detect stroke_thickness_score
    """
    image = img28(position, X_data, size=cfg.IMAGE_SIZE)
    mask = digit_mask(image, cfg.MASK_THRESHOLD, dilation=0).astype(bool)

    skeleton = skeletonize(mask)
    if not mask.any() or not skeleton.any():
        return None

    distances = distance_transform_edt(mask)
    # thickness estimate: twice mean distance
    # of skeleton pixels from border of mask 
    return float(2 * distances[skeleton].mean())


def select_thin_thick_examples(X_data, y_data, cfg, quantiles=(0.2, 0.8)):
    """
        select_thin_thick_examples
        select images in which the lines are particularly thin or thick.
        
    """
    labels = np.asarray(y_data).reshape(-1)
    selected = {}
    
    for digit in np.unique(labels):
        candidates = np.flatnonzero(labels == digit)
        scored = []

        for position in candidates:
            score = stroke_thickness_score(position, X_data, cfg)
            if score is not None:
                scored.append((int(position), score))

        if len(scored) < 2:
            continue

        positions = np.array([item[0] for item in scored])
        scores = np.array([item[1] for item in scored])

        thin_target, thick_target = np.quantile(scores, quantiles)

        thin_idx = np.argmin(np.abs(scores - thin_target))
        thick_idx = np.argmin(np.abs(scores - thick_target))

        selected[int(digit)] = { "thin": int(positions[thin_idx]),
                                "thick": int(positions[thick_idx]),
                                "thin_score": float(scores[thin_idx]),
                                "thick_score": float(scores[thick_idx]),
                                }

    return selected


def pre_compare_group_xai_2(cfg,ctx, quantiles=(0.2, 0.8)):
    
    """
        analysis pre compare group to choice Treshold mask
        The final print score is a useful estimate for selecting examples, 
        not a perfect measure of significance: it depends on the threshold.
    """
    
    X_data, y_data, = ctx["X_data"], ctx["y_data"]

    selected = select_thin_thick_examples(X_data, y_data, cfg)

    # positions to pass at view function 
    positions = [ position
                    for examples in selected.values()
                        for position in (examples["thin"], examples["thick"])
                ]
    inspect_digit_masks(positions, X_data, cfg)
    print(selected)

def run_compare_group_xai_1(cfg, 
                            logger, 
                            model, 
                            device, 
                            techniques, 
                            lr, 
                            ctx, 
                            group, 
                            list_images ):
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
    ig_max = cfg.IG_MAX_COMPLETENESS_ERROR
    
    model.eval()

    rows, first_input = [], None
    
    for id_class in sorted(list_images):
        for item in list_images[id_class]:
            org_idx = item["id_dataset"]
            if org_idx not in idx_to_pos:
                raise KeyError(f"id_dataset {org_idx} not in this split")
            position = idx_to_pos[org_idx]

            # get X and y for the current image
            input_image, input_label, idx = dataset[position]
            assert idx == org_idx

            # check the target class on record selected
            assert int(input_label) == int(y_data[position])

            # add the batch dimension to the image
            # from (3,H,W) to (1,3,H,W)
            input_img = input_image.unsqueeze(0).to(device)
            
            # The CNN produces logits, 
            # which are transformed into probabilities via Softmax.
            # (multiclass classification)
            with torch.no_grad():
                probs = F.softmax(model(input_img), dim=1)[0]

            # get predicted and real class based on the max prob
            pred_cnn, true = int(probs.argmax()), int(input_label)
            pred_lr = int(y_pred_lr[position])

            if first_input is None:
                first_input = input_img

            # If the CNN is correct, we run an explanation only for the predicted class.         
            # If the CNN is incorrect, we run 2 explanations:
            # - pred_cnn: why did the CNN choose the wrong class?
            # - pred_lr: why did the LR choose the wrong class?
            # - true: which parts of the image support the correct class?
            targets_cnn = { "pred_cnn": pred_cnn }
            targets_lr = {"pred_lr": pred_lr}

            if pred_cnn != true:
                # what was missing for the right answer
                targets_cnn["true"] = true    
            if pred_lr != true:
                # what was missing for the right answer
                targets_lr["true"] = true               

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

            row = { "index": org_idx, 
                    "pos": position, 
                    "true": true, 
                    "pred_cnn": pred_cnn,
                    "pred_lr": pred_lr, 
                    "prob_pred": float(probs[pred_cnn]),
                    "targets_cnn": targets_cnn, 
                    "targets_lr": targets_lr, 
                    "mask_mean": mask.mean(), 
                    "mask_dilation": cfg.MASK_DILATION, 
                    "mask_treshold": cfg.MASK_THRESHOLD, 
                    "maps_cnn": {}, 
                    "metrics_cnn": {}, 
                    "meta_cnn": {}, 
                    "maps_lr": {}, 
                    "metrics_lr": {},  
                    "meta_lr": {},             
                    "signed_diags": {"cnn": {}, "lr": {} }
                }

            # for each target in cnn_targets, 
            # it generate maps using XAI techniques applied to the CNN;
            for target_name, target_class in targets_cnn.items():
                # Run Explanations for all techniques on current image
                # alwais passing pred on targets
                # only for error passing true on targets
                # We get as results:
                # maps	Maps generated by XAI techniques
                # metrics	Quantitative metrics for faithfulness and localization
                # meta	Additional information, such as agreement and random baseline
                # generate maps explainer apply to CNN,
                maps_cnn, metrics_cnn, meta_cnn = explain_target_1(techniques, 
                                                                   input_img, 
                                                                   target_class, 
                                                                   mask, 
                                                                   cfg)
                
                # For instance, if the CNN predicts 5 and the LR predicts 8, 
                # you would obtain CNN maps for 5 and 2, and LR maps for 8 and 2. 
                # You do not calculate a CNN map for 8—that would constitute 
                # an additional analysis of the CNN for a target it did not predict, 
                # rather than an explanation of its actual decision.
                row["maps_cnn"][target_name] = maps_cnn
                row["metrics_cnn"][target_name] = metrics_cnn
                row["meta_cnn"][target_name] = meta_cnn
                row["signed_diags"]["cnn"][target_name] = {technique_name: signed_diags(res.map)
                                                           for technique_name, res in maps_cnn.items()
                                                           }

                # START SPEARMAN
                agreement = spearman_agreement_matrix(maps_cnn, cfg)
                
                # setup row["meta_cnn"][target_name] and return reference.
                # dicts are mutablle object passed by refernce
                target_meta = row["meta_cnn"].setdefault(target_name, {})
                # add the next key to row["meta_cnn"][target_name]
                target_meta["agreement_spearman_matrix"] = { "techniques": agreement["techniques"],
                                                            "matrix": agreement["matrix"],
                                                            }
                target_meta["agreement_spearman"] = agreement["mean_off_diagonal"]
                # END SPEARMAN
                
                # START TOPK IOU
                for tech_name, res in maps_cnn.items():
                    metrics_cnn[tech_name].update(compute_topk_iou(res.map, mask, cfg))
                # END TOPK IOU
                
            # for each target in lr_targets, 
            # it generate an LR map using lr.local_map(...).
            for target_name, target_class in targets_lr.items():
                maps_lr = lr.local_map(X_data[position], target_class)
                # Also generate the map of the Logistic Regression coefficients 
                # for the same target.
                # This allows us to compare the explanations provided by the CNN 
                # with the linear representation of the LR.
                row["maps_lr"][target_name] = maps_lr
                row["signed_diags"]["lr"][target_name] = {"lr_local_map": signed_diags(maps_lr)}
                
                # row["metrics_lr"][target_name] = compute_lr_metrics(maps_lr, mask)
                lr_metrics = compute_lr_metrics(maps_lr, mask)
                lr_metrics.update(compute_topk_iou(maps_lr, mask, cfg))
                row["metrics_lr"][target_name] = lr_metrics


            # completeness_error only for cnn and Integrated Gradients: 
            # not a general metric applicable to all maps. 
            # It verifies how closely the sum of the attributions approaches the difference 
            # between the model’s output for the image and its output for the chosen baseline. 
            # It does not measure 
            # how “correct” the map is, nor the extent 
            # to which it aligns with LR.
            ig_pred = row["maps_cnn"]["pred_cnn"].get("integrated_gradients")
            if ig_pred is not None:
                err = ig_pred.meta.get("completeness_error")
                if err is not None:
                    row["ig_error_pred"] = to_valid_float(err)
                    row["ig_unreliable_pred"] = bool(err > ig_max)

            # exist only if cnn has wrong
            if "true" in row["maps_cnn"]:
                ig_true = row["maps_cnn"]["true"].get("integrated_gradients")
                if ig_true is not None:
                    err = ig_true.meta.get("completeness_error")
                    if err is not None:
                        row["ig_error_true"] = to_valid_float(err)
                        row["ig_unreliable_true"] = bool(err > ig_max)

            rows.append(row)
            logger.info("[%s] idx %d pos %d true %d CNN %d LR %d",
                        group, org_idx, position, true, pred_cnn,pred_lr)

    # Lime Shap stability of Explanation
    # stochastic techniques: same image explained twice
    # Check whether LIME and SHAP produce the same explanation 
    # when run twice on the same image.
    stability_cnn = {} 
    if rows:
        for name in ("lime", "shap"):
            if name in techniques:
                again = techniques[name].explain(first_input, rows[0]["pred_cnn"]).map
                diff = float(np.abs(again - rows[0]["maps_cnn"]["pred_cnn"][name].map).max())
                # Of the difference is less than RELU_MAX_REF_DIFF, 
                # the check is considered passed.
                stability_cnn[name] = {"max_abs_diff": diff, "ok": bool(diff <= cfg.RELU_MAX_REF_DIFF)}

    #  generates map visualizations;
    label = GROUPS_RESULT_OK_KO[group]
    plot_maps_grid_dinamic(rows, "pred_cnn","maps_cnn", names_tech, cfg, X_data,
                   f"XAI validation [{label}] - explained: CNN PREDICTED class",
                   out_path / f"maps_grid_{group}_pred_cnn.png", logger)

    plot_maps_grid_dinamic(rows, "pred_lr","maps_lr", names_tech, cfg, X_data,
                       f"XAI validation [{label}] - explained: LR PREDICTED class",
                       out_path / f"maps_grid_{group}_pred_lr.png", logger)
    
    if any("true" in r["maps_cnn"] for r in rows):
        plot_maps_grid_dinamic(rows, "true" , "maps_cnn", names_tech, cfg, X_data,
                   f"XAI validation [{label}] - explained: TARGET TRUE FROM CNN MAP ",
                   out_path / f"maps_grid_{group}_cnn_true.png", logger)

    if any("true" in r["maps_lr"] for r in rows):
            plot_maps_grid_dinamic(rows, "true","maps_lr", names_tech, cfg, X_data,
                       f"XAI validation [{label}] - explained: TARGET TRUE FROM LR MAP ",
                       out_path / f"maps_grid_{group}_lr_true.png", logger)
            
    
    summary_cnn = {"pred_cnn": {}, "true": {}}
    summary_lr = {"pred_lr": {}, "true": {}}
    records = []

    if rows:
        # Summary LR on pred_lr 
        # (no ["lr_local_map"], compute_lr_metrics retunr  dict flat)
        vals_lr_pred = [r["metrics_lr"]["pred_lr"] for r in rows]
        summary_lr["pred_lr"]["lr_local_map"] = { key: to_valid_float(np.mean([v[key] 
                                                    for v in vals_lr_pred]))
                                                        for key in ("mask_share", "focus_ratio")
                                                }

        # Summary LR on true (only if exist)
        rows_lr_true = [r for r in rows if "true" in r["metrics_lr"]]
        if rows_lr_true:
            vals_lr_true = [r["metrics_lr"]["true"] for r in rows_lr_true]
            summary_lr["true"]["lr_local_map"] = { key: to_valid_float(np.mean([v[key] 
                                                    for v in vals_lr_true]))
                                                       for key in ("mask_share", "focus_ratio")
                                                 }

        # Summary CNN on pred_cnn and true
        rows_cnn_true = [r for r in rows if "true" in r["metrics_cnn"]]
        for tech_name in names_tech:
            vals_cnn_pred = [r["metrics_cnn"]["pred_cnn"][tech_name] for r in rows]
            summary_cnn["pred_cnn"][tech_name] = { key: to_valid_float(np.mean([v[key] 
                                                for v in vals_cnn_pred]))
                                                    for key in ("mask_share", "focus_ratio")
                                            }
            if rows_cnn_true:
                vals_cnn_true = [r["metrics_cnn"]["true"][tech_name] for r in rows_cnn_true]
                summary_cnn["true"][tech_name] = { key: to_valid_float(np.mean([v[key] 
                                              for v in vals_cnn_true]))
                                                for key in ("mask_share", "focus_ratio")
                }
        
    
    # creates records for the CSV;
    for r in rows:
        for target_name, target_class in r["targets_cnn"].items():
            for tech_name in names_tech:
                m = r["metrics_cnn"][target_name][tech_name]
                records.append({"group": group, 
                                "index": r["index"], 
                                "true": r["true"],
                                "pred": r["pred_cnn"], 
                                "pred_lr": r["pred_lr"],
                                "mask_mean": r["mask_mean"],
                                "mask_dilation": r["mask_dilation"],
                                "mask_treshold": r["mask_treshold"],
                                "model": "cnn",
                                "target_name": target_name, 
                                "target_class": target_class , 
                                "technique": name,
                                "mask_share": m["mask_share"], 
                                "mask_frac": m["mask_frac"],
                                "focus_ratio": m["focus_ratio"], 
                                # meta_cnn not present if only lr is wrong
                                "agreement": r["meta_cnn"].get(target_name, {}).get("agreement_spearman"),
                                "iou_top_05": m["iou_top_05"],
                                "iou_top_10": m["iou_top_10"],
                                "iou_top_20": m["iou_top_20"]
                                })
        
        for target_name, target_class in r["targets_lr"].items():
            m = r["metrics_lr"][target_name]
            records.append({ "group": group, 
                            "index": r["index"], 
                            "true": r["true"],
                            "pred": r["pred_cnn"], 
                            "pred_lr": r["pred_lr"],
                            "mask_mean": r["mask_mean"],
                            "mask_dilation": r["mask_dilation"],
                            "mask_treshold": r["mask_treshold"],
                            "model": "lr",
                            "target_name": target_name, 
                            "target_class": target_class, 
                            "technique": "lr_local_map",
                            "mask_share": m["mask_share"],
                            "mask_frac": m["mask_frac"],
                            "focus_ratio": m["focus_ratio"],
                             "iou_top_05": m["iou_top_05"],
                            "iou_top_10": m["iou_top_10"],                               "iou_top_20": m["iou_top_20"]
                        })
        
    # saves the results in JSON format.
    # The final structure contains information 
    # for each image, target, and XAI technique.
    results = { "group": group, 
                "label": label,
                "mask_mean": mask.mean(),
                "mask_dilation": cfg.MASK_DILATION,
                "mask_treshold": cfg.MASK_THRESHOLD,
                "model": {k: ctx["info"].get(k) for k in ("action", "reason", "device")},
                "n_images": len(rows),
                "n_cnn_wrong": sum(r["pred_cnn"] != r["true"] for r in rows),
                "n_lr_wrong": sum(r["pred_lr"] != r["true"] for r in rows),
                "stability_cnn": stability_cnn,
                "summary_cnn": summary_cnn,
                "summary_lr": summary_lr,
                "images": format_json_xai_rows(rows),
                "total_seconds": time.time() - t0
            }
    
    (out_path / f"stage2_compare1_results_{group}.json").write_text(json.dumps(results,
                                                                                   indent=2, 
                                                                                   default=str),
                                                                        encoding="utf-8"
                                                                        )
    
    logger.info("END run_compare_group_xai_1 return 3 params ") 
    logger.info(f"sizes -> results_keys={len(results)} records={len(records)} rows={len(rows)}") 
    logger.info(f"types -> results_keys={type(results)} records={type(records)} rows={type(rows)}")                                                         
    return results, records , rows

"""
        The general objective of the code is to compare XAI explanations produced 
        by different techniques, by examining :
        How faithful(fedele) are they to the model? 
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
def run_compare_group_xai_2(cfg, 
                            logger,
                            source_results,
                            source_records,
                            source_rows,
                            model,
                            device,
                            techniques, 
                            ctx,
                            group, 
                            list_images ):
    
    
    t0 = time.time()
    out_path = ctx["out_path"]
    dataset = ctx["dataloader_cnn"].dataset
    idx_to_pos = dataset.idx_dataset_position()
    # X_data, y_data, y_pred_lr = ctx["X_data"], ctx["y_data"], ctx["y_pred_lr"]
    names_tech = list(techniques)
    
    # fractions: percentages of pixels to be removed during the deletion test.
    fractions = cfg.DELETION_FRACTIONS
            
    # a random number generator with a fixed seed.
    rng = np.random.default_rng(cfg.SEED)
    
    model.eval()

    rows = []
    for source_row in source_rows:
        org_idx = source_row["index"]

        if org_idx not in idx_to_pos:
            raise KeyError(f"id_dataset {org_idx} not in this split")

        position = idx_to_pos[org_idx]

        # get X and y for the current image
        input_image, input_label, idx = dataset[position]

        if int(idx) != int(org_idx):
            raise ValueError(f"Indice non corrispondente: record={org_idx}, dataset={idx}")

        if int(input_label) != int(source_row["true"]):
             raise ValueError(f"Etichetta non corrispondente per index={org_idx}: "
                              f"record={source_row['true']}, dataset={input_label}")
        # add the batch dimension to the image
        # from (3,H,W) to (1,3,H,W)
        input_img = input_image.unsqueeze(0).to(device)
            
        # baseline: the value that will replace the removed pixels.
        baseline = make_blurred_baseline(input_img, sigma=cfg.DELETION_SIGMA)

        # not modify source record 
        result_row = dict(source_row)
        result_row["metrics_cnn_del"] = {}
        result_row["meta_cnn_del"] = {}

        # for each target in cnn_targets, 
        # it recovery maps from XAI techniques 
        # applied to the CNN;
        for target_name, target_class in source_row["targets_cnn"].items():
            maps_for_target = source_row["maps_cnn"].get(target_name)

            if not maps_for_target:
                raise KeyError( f"Nessuna mappa CNN per target={target_name}, "
                                f"index={org_idx}"
                                )

            # prob explain class on CNN.
            # target_class into lambda function.
            score_fn = lambda x, cls=int(target_class): float(F.softmax(model(x), dim=1)[0, cls].item())

            metrics_del, meta_del = explain_target_2(score_fn=score_fn,
                                                    input_img=input_img,
                                                    maps_for_target=maps_for_target,
                                                    fractions=fractions,
                                                    baseline=baseline,
                                                    rng=rng,
                                                )

            result_row["metrics_cnn_del"][target_name] = metrics_del
            result_row["meta_cnn_del"][target_name] = meta_del
                  
            # For instance, if the CNN predicts 5 and the LR predicts 8, 
            # you would obtain CNN maps for 5 and 2, and LR maps for 8 and 2. 
            # You do not calculate a CNN map for 8—that would constitute 
            # an additional analysis of the CNN for a target it did not predict, 
            # rather than an explanation of its actual decision.
            plot_deletion_curves(metrics_del, 
                                 meta_del, 
                                 fractions, 
                                 out_path / f"deletion_curves_{group}_{org_idx}_{target_name}_cnn.png",
                                 logger
                                 )

        rows.append(result_row)
        logger.info("[%s] idx %d pos %d: deletion CNN completata",group,org_idx,position)
        
    # creates records for the CSV;
    records_dest = []
    for r in rows:
        for target_name, target_class in r["targets_cnn"].items():
            for tech_name in names_tech:
                 metric = r["metrics_cnn"][target_name][tech_name]
                 deletion_metric = r["metrics_cnn_del"][target_name][tech_name]
                 deletion_meta = r["meta_cnn_del"][target_name]

                 records_dest.append(
                 {
                        "group": group,
                        "index": result_row["index"],
                        "true": result_row["true"],
                        "pred_cnn": result_row["pred_cnn"],
                        "pred_lr": result_row["pred_lr"],
                        "model": "cnn",
                        "target_name": target_name,
                        "target_class": target_class,
                        "technique": tech_name,
                        "mask_share": metric["mask_share"],
                        "mask_frac": metric["mask_frac"],
                        "focus_ratio": metric["focus_ratio"],
                        "agreement_spearman_mean": (
                            result_row.get("meta_cnn", {})
                            .get(target_name, {})
                            .get("agreement_spearman")
                        ),
                        "agreement_spearman_matrix": (result_row.get("meta_cnn", {})
                                                    .get(target_name, {})
                                                    .get("agreement_spearman_matrix")
                                                    ),
                        "agreement_spearman": (result_row.get("meta_cnn", {})
                                                            .get(target_name, {})
                                                            .get("agreement_spearman")
                                                            ),
                        "del_auc": deletion_metric["del_auc"],
                        "del_gain": deletion_metric["del_gain"],
                        "del_auc_random": deletion_meta["del_auc_random"],
                    }
                 ) 
                 
        for target_name, target_class in r["targets_lr"].items():
            metric = r["metrics_lr"][target_name]
            records_dest.append({ "group": group, 
                            "index": r["index"], 
                            "true": r["true"],
                            "pred_cnn": r["pred_cnn"], 
                            "pred_lr": r["pred_lr"],
                            "model": "lr",
                            "target_name": target_name, 
                            "target_class": target_class, 
                            "technique": "lr_local_map",
                            "mask_share": metric["mask_share"],
                            "mask_frac": metric["mask_frac"],
                            "focus_ratio": metric["focus_ratio"],
                            "agreement_spearman_matrix": None,
                            "agreement_spearman": None,
                            "del_auc": None,
                            "del_gain": None,
                            "del_auc_random": None,
                        })
            
    # save summary_cnn with mean of del_auc and del_gain 
    summary_cnn_dest = copy.deepcopy(source_results["summary_cnn"])
    if rows:
        rows_cnn_true = [r for r in rows if "true" in r["metrics_cnn_del"]]
        for tech_name in names_tech:
            vals_del_pred = [r["metrics_cnn_del"]["pred_cnn"][tech_name] for r in rows]
            summary_cnn_dest["pred_cnn"][tech_name].update({
                key: to_valid_float(np.mean([v[key] for v in vals_del_pred]))
                for key in ("del_auc", "del_gain")
            })
            if rows_cnn_true:
                vals_del_true = [r["metrics_cnn_del"]["true"][tech_name] for r in rows_cnn_true]
                summary_cnn_dest["true"][tech_name].update({key: to_valid_float(np.mean([v[key] 
                                                           for v in vals_del_true]))
                                                               for key in ("del_auc", "del_gain")
                })

    # for each image, target, and XAI technique.
    label = GROUPS_RESULT_OK_KO[group]
    results_dest = { "group": group, 
                "label": label,
                "model": {key: ctx["info"].get(key) for key in ("action", "reason", "device")},
                "n_images": len(rows),
                "n_cnn_wrong": sum(r["pred_cnn"] != r["true"] for r in rows),
                "n_lr_wrong": sum(r["pred_lr"] != r["true"] for r in rows),
                "stability_cnn": source_results["stability_cnn"],
                "summary_cnn": source_results["summary_cnn"],
                "summary_lr": source_results["summary_lr"],
                "summary_cnn_dest": summary_cnn_dest,
                "images": format_json_xai_rows(rows),
                "total_seconds": time.time() - t0
            }
                                                                                     
    (out_path / f"stage2_compare2_results_{group}.json").write_text(json.dumps(results_dest,
                                                                                       indent=2, 
                                                                                       default=str),
                                                                            encoding="utf-8"
                                                                            )
    
    return results_dest , records_dest 



def run_compare_group_xai_2_profai( cfg,
                            logger,
                            source_results,
                            source_records,
                            model,
                            techniques,
                            ctx,
                            group,
                        ):
    t0 = time.time()
    out_path = ctx["out_path"]
    dataset = ctx["dataloader_cnn"].dataset
    idx_to_pos = dataset.idx_dataset_position()
    names_tech = list(techniques)

    fractions = cfg.DELETION_FRACTIONS
    rng = np.random.default_rng(cfg.SEED)

    model.eval()
    rows = []

    for source_record in source_records:
        org_idx = source_record["index"]

        if org_idx not in idx_to_pos:
            raise KeyError(f"id_dataset {org_idx} non presente in questo split")

        position = idx_to_pos[org_idx]
        input_image, input_label, dataset_idx = dataset[position]

        if int(dataset_idx) != int(org_idx):
            raise ValueError(f"Indice non corrispondente: "
                            f"record={org_idx}, dataset={dataset_idx}"
                            )               

        if int(input_label) != int(source_record["true"]):
            raise ValueError(f"Etichetta non corrispondente per index={org_idx}: "
                            f"record={source_record['true']}, dataset={input_label}"
                            )

        input_img = input_image.unsqueeze(0).to(ctx["device"])
        baseline = make_blurred_baseline(input_img,sigma=cfg.DELETION_SIGMA,)

        # Copia la riga di fase 1 e aggiunge i campi di fase 2.
        result_row = dict(source_record)
        result_row["metrics_cnn_del"] = {}
        result_row["meta_cnn_del"] = {}

        for target_name, target_class in source_record["targets_cnn"].items():
            maps_for_target = source_record["maps_cnn"].get(target_name)

            if not maps_for_target:
                raise KeyError(
                    f"Nessuna mappa CNN per target={target_name}, "
                    f"index={org_idx}"
                )

            # Probabilità della classe spiegata, calcolata con la CNN.
            score_fn = lambda x, cls=int(target_class): float(F.softmax(model(x), 
                                                                        dim=1)[0, cls].item()
                                                            )

            metrics_del, meta_del = explain_target_2(score_fn=score_fn,
                                                    input_img=input_img,
                                                    maps_for_target=maps_for_target,
                                                    fractions=fractions,
                                                    baseline=baseline,
                                                    rng=rng,
                                                )

            result_row["metrics_cnn_del"][target_name] = metrics_del
            result_row["meta_cnn_del"][target_name] = meta_del

            plot_deletion_curves( metrics_del,
                                    meta_del,
                                    fractions,
                                    out_path
                                    / f"deletion_curves_{group}_{org_idx}_{target_name}_cnn.png",
                                    logger,
                                )

        rows.append(result_row)

        logger.info("[%s] idx %d pos %d: deletion CNN completata",group,org_idx,position)

    # Tabella long: una riga per immagine, target e tecnica.
    records_dest = []

    for result_row in rows:
        for target_name, target_class in result_row["targets_cnn"].items():
            for technique_name in names_tech:
                metric = result_row["metrics_cnn"][target_name][technique_name]
                deletion_metric = result_row["metrics_cnn_del"][target_name][technique_name]
                deletion_meta = result_row["meta_cnn_del"][target_name]

                records_dest.append({"group": group,
                                    "index": result_row["index"],
                                    "true": result_row["true"],
                                    "pred_cnn": result_row["pred_cnn"],
                                    "pred_lr": result_row["pred_lr"],
                                    "model": "cnn",
                                    "target_name": target_name,
                                    "target_class": target_class,
                                    "technique": technique_name,
                                    "mask_share": metric["mask_share"],
                                    "mask_frac": metric["mask_frac"],
                                    "focus_ratio": metric["focus_ratio"],
                                    "agreement_spearman_mean": (
                                        result_row.get("meta_cnn", {})
                                        .get(target_name, {})
                                        .get("agreement_spearman")
                                    ),
                                    "del_auc": deletion_metric["del_auc"],
                                    "del_gain": deletion_metric["del_gain"],
                                    "del_auc_random": deletion_meta["del_auc_random"],
                                }
                                )

        # Le righe LR contengono le metriche di localizzazione della fase 1.
        # La deletion LR non viene calcolata da questa funzione.
        for target_name, target_class in result_row["targets_lr"].items():
            metric = result_row["metrics_lr"][target_name]

            records_dest.append({"group": group,
                                "index": result_row["index"],
                                "true": result_row["true"],
                                "pred_cnn": result_row["pred_cnn"],
                                "pred_lr": result_row["pred_lr"],
                                "model": "lr",
                                "target_name": target_name,
                                "target_class": target_class,
                                "technique": "lr_local_map",
                                "mask_share": metric["mask_share"],
                                "mask_frac": metric["mask_frac"],
                                "focus_ratio": metric["focus_ratio"],
                                "agreement_spearman_mean": None,
                                "del_auc": None,
                                "del_gain": None,
                                "del_auc_random": None,
                            }
                        )

    # Copia la sintesi di fase 1 e aggiunge le medie della deletion CNN.
    summary_cnn_dest = copy.deepcopy(source_results["summary_cnn"])

    for technique_name in names_tech:
        pred_values = [ row["metrics_cnn_del"]["pred_cnn"][technique_name]
                            for row in rows
                            if technique_name
                            in row["metrics_cnn_del"].get("pred_cnn", {})
                        ]

        if pred_values:summary_cnn_dest["pred_cnn"][technique_name].update(
                                                        {key: to_valid_float(np.mean([value[key] 
                                                                                      for value in pred_values])
                                                                                    )
                                                                for key in ("del_auc", "del_gain")
                                                            }
                                                        )

        true_values = [row["metrics_cnn_del"]["true"][technique_name]
                        for row in rows
                            if technique_name in row["metrics_cnn_del"].get("true", {})
                        ]

        if true_values:
            summary_cnn_dest["true"][technique_name].update({key: 
                                                    to_valid_float(
                                                    np.mean([value[key] for value in true_values])
                                                    )
                                                for key in ("del_auc", "del_gain")
                }
            )

    label = GROUPS_RESULT_OK_KO[group]

    results_dest = {
        "group": group,
        "label": label,
        "model": {
            key: ctx["info"].get(key)
            for key in ("action", "reason", "device")
        },
        "n_images": len(rows),
        "n_cnn_wrong": sum(
            row["pred_cnn"] != row["true"] for row in rows
        ),
        "n_lr_wrong": sum(
            row["pred_lr"] != row["true"] for row in rows
        ),
        "stability_cnn": source_results["stability_cnn"],
        "summary_cnn_1": source_results["summary_cnn"],
        "summary_cnn_2": summary_cnn_dest,
        "summary_lr": source_results["summary_lr"],
        "images": format_json_xai_rows(rows),
        "total_seconds": time.time() - t0,
    }

    json_path = out_path / f"stage2_compare2_results_{group}.json"
    json_path.write_text(
        json.dumps(results_dest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    return results_dest, records_dest



