"""
     Plot Graph for Xai
"""
import math
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.figure import Figure
from IPython.display import Image, display
from utility import img28 , digit_mask
from constants import TITLES , MASK_THRESHOLD_LIST
import torch

def to_json_value(obj):
    """
        Recursive Converts type NumPy/PyTorch into serializable JSON.
    """

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, np.generic):
        return to_json_value(obj.item())

    if isinstance(obj, np.ndarray):
        return to_json_value(obj.tolist())

    if isinstance(obj, torch.Tensor):
        return to_json_value(obj.detach().cpu().tolist())

    if isinstance(obj, dict):
        return {str(key): to_json_value(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_value(value) for value in obj]

    raise TypeError(
        f"Tipo non gestito nella serializzazione JSON: {type(obj).__name__}"
    )


def format_json_xai_rows(rows,include_maps=False):
    """
        Converts XAI rows into a JSON-serializable structure.

        Includes:
        CNN maps for each target and technique, containing the map itself, 
        signed values, and all associated metadata;
        LR maps for each target;
        analysis metrics and metadata;
        sign diagnostics.

        include_maps=True to include maps
    """

    formatted_images = []

    for row in rows:
        
        img_entry = { "index": to_json_value(row["index"]),
                    "pos": to_json_value(row["pos"]),
                    "true": to_json_value(row["true"]),
                    "pred_cnn": to_json_value(row["pred_cnn"]),
                    "pred_lr": to_json_value(row["pred_lr"]),
                    "prob_pred": to_json_value(row.get("prob_pred")),

                    "targets_cnn": to_json_value(row.get("targets_cnn", {})),
                    "targets_lr": to_json_value(row.get("targets_lr", {})),

                    # "maps_cnn": maps_cnn_json,
                    # "maps_lr": maps_lr_json,

                    "metrics_cnn": to_json_value(row.get("metrics_cnn", {})),
                    "meta_cnn": to_json_value(row.get("meta_cnn", {})),
                    "metrics_lr": to_json_value(row.get("metrics_lr", {})),
                    "meta_lr": to_json_value(row.get("meta_lr", {})),

                    "signed_diags": to_json_value(row.get("signed_diags", {})),

                    "ig_completeness_error_pred": to_json_value(
                        row.get("ig_error_pred")
                    ),
                    "ig_unreliable_pred": to_json_value(
                        row.get("ig_unreliable_pred")
                    ),
                    "ig_completeness_error_true": to_json_value(
                        row.get("ig_error_true")
                    ),
                    "ig_unreliable_true": to_json_value(
                        row.get("ig_unreliable_true")
                    ),
                    "metrics_cnn_del": to_json_value(row.get("metrics_cnn_del", {})),
                    "meta_cnn_del": to_json_value(row.get("meta_cnn_del", {})),
                    # spearman is already in  in meta_cnn
                    # "agreement_matrix": row["meta_cnn"][target_name]["agreement_spearman_matrix"]["matrix"],
                    # "agreement_spearman": row["meta_cnn"][target_name]["agreement_spearman_matrix"]["agreement_spearman"],
        }

        if include_maps:
            maps_cnn_json = {}
            
            for target_name, techniques_results in row.get("maps_cnn", {}).items():
                maps_cnn_json[target_name] = {}
            
                for technique_name, res in techniques_results.items():
                            maps_cnn_json[target_name][technique_name] = { "map": to_json_value(res.map),
                                                                        "signed": bool(res.signed),
                                                                        "meta": to_json_value(getattr(res, "meta", {})),
                                                                    }
            
            maps_lr_json = { target_name: to_json_value(lr_map)
                                for target_name, lr_map in row.get("maps_lr", {}).items()
                            }

            img_entry["maps_cnn"] = maps_cnn_json
            img_entry["maps_lr"] = maps_lr_json

        formatted_images.append(img_entry)

    return formatted_images


def format_json_xai_maps_rows( rows,
                              group=None,
                              label=None,
                              mask_threshold=0.1,
                              mask_dilation=0,
                          ):
    """
         Create payload JSON light with maps and metadata 
         This is file will be used to recovery images and compare target.
    """

    formatted_images = []

    for row in rows:
        maps_cnn_json = {}

        for target_name, techniques_results in row.get("maps_cnn", {}).items():
            maps_cnn_json[target_name] = {}

            for technique_name, res in techniques_results.items():
                maps_cnn_json[target_name][technique_name] = {
                    "map": to_json_value(res.map),
                    "signed": bool(res.signed),
                }

        maps_lr_json = {
            target_name: to_json_value(lr_map)
            for target_name, lr_map in row.get("maps_lr", {}).items()
        }

        img_entry = { "index": to_json_value(row["index"]),
                      "pos": to_json_value(row["pos"]),
                      "true": to_json_value(row["true"]),
                      "pred_cnn": to_json_value(row["pred_cnn"]),
                      "pred_lr": to_json_value(row["pred_lr"]),
                      "targets_cnn": to_json_value(row.get("targets_cnn", {})),
                      "targets_lr": to_json_value(row.get("targets_lr", {})),
                      "maps_cnn": maps_cnn_json,
                      "maps_lr": maps_lr_json,
                      "input_image": to_json_value(row.get("input_image")),
                  }

        formatted_images.append(img_entry)

    return { "group": group,
              "label": label,
              "mask_threshold": mask_threshold,
              "mask_dilation": mask_dilation,
              "n_images": len(formatted_images),
              "images": formatted_images,
          }
    
def plot_maps_grid_dinamic(rows, target_key, maps_key, names, cfg, X_data, title, path, logger, max_rows=10):
    sel = [r for r in rows if target_key in r[maps_key]][:max_rows]
    if not sel:
        return

    is_cnn = (maps_key == "maps_cnn")
    targets_dict_key = "targets_cnn" if is_cnn else "targets_lr"
    n_map_cols = len(names) if is_cnn else 1
    total_cols = n_map_cols + 1  # Colonna 0 = immagine originale + colonne mappe

    fig = Figure(figsize=(2.1 * total_cols, 2.1 * len(sel)))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    axes = np.atleast_2d(fig.subplots(len(sel), total_cols))

    for r, row in enumerate(sel):
        axes[r, 0].imshow(img28(row["pos"], X_data, size=cfg.IMAGE_SIZE), cmap="gray")
        # pred_cnn and  dinamic targets_dict_key 
        axes[r, 0].set_ylabel(
            f"true {row['true']} | CNN {row['pred_cnn']} | LR {row['pred_lr']}\n"
            f"explained: {row[targets_dict_key][target_key]}",
            fontsize=7
        )

        if is_cnn:
            cells = [(row["maps_cnn"][target_key][n].map, row["maps_cnn"][target_key][n].signed)
                     for n in names]
        else:
            cells = [(row["maps_lr"][target_key], True)]

        for c, (m, signed) in enumerate(cells, start=1):
            a = float(np.abs(m).max()) or 1.0
            axes[r, c].imshow(
                m,
                cmap="seismic" if signed else "viridis",
                vmin=-a if signed else 0,
                vmax=a,
                interpolation="nearest" if m.shape[0] <= 8 else "bilinear"
            )

        for c in range(total_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    col_titles = ["image"] + ([TITLES[n] for n in names] if is_cnn else ["LR local\n(w * x)"])
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9)

    try:
        fig.savefig(path, dpi=90, bbox_inches="tight")
        try:
            display(Image(str(path)))
        except Exception:
            logger.exception("Impossibile mostrare %s", path)
    finally:
        #fig.clear()
        plt.close(fig)

def plot_maps_grid(rows, target_key, names, cfg, X_data, title, path, logger, max_rows=10):

    sel = [r for r in rows if target_key in r["maps"]][:max_rows]
    
    fig = Figure(figsize=(2.1 * (len(names) + 2), 2.1 * len(sel)))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    axes = np.atleast_2d(fig.subplots(len(sel), len(names) + 2))

    for r, row in enumerate(sel):
        axes[r, 0].imshow(img28(row["pos"], X_data, size=cfg.IMAGE_SIZE), cmap="gray")
        axes[r, 0].set_ylabel(f"true {row['true']} | CNN {row['pred']} | LR {row['pred_lr']}\n"
                              f"explained: {row['targets'][target_key]}", fontsize=7)
        cells = [(row["maps"][target_key][n].map, row["maps"][target_key][n].signed)
                 for n in names] + [(row["lr_map"][target_key], True)]
        # mind the sign is False for GradCam and True for each other
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
        # fig.clear()
        plt.close(fig)


def plot_deletion_curves(metrics, meta, fractions, path,logger):
    """
        plot deletion curves 
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        fractions,
        meta["del_curve_random"],
        linestyle="--",
        label="Random baseline"
    )

    for name, values in metrics.items():
        ax.plot(
            fractions,
            values["del_curve"],
            marker="o",
            label=name
        )

    ax.set_xlabel("Percentuale di pixel rimossi")
    ax.set_ylabel("Probabilità della classe target")
    ax.set_title("Deletion curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        try:
            display(Image(str(path)))
        except Exception:
            logger.exception("Impossibile mostrare %s", path)
    finally:
        # fig.clear()
        plt.close(fig)


def inspect_digit_masks(positions, X_data, cfg):
    """
        plot image to inspect_digit_masks and choice threshold param
    """
    fig, axes = plt.subplots(
        len(positions),
        len(MASK_THRESHOLD_LIST) + 1,
        figsize=(3 * (len(MASK_THRESHOLD_LIST) + 1), 3 * len(positions)),
        squeeze=False,
    )

    for row_idx, position in enumerate(positions):
        image = img28(position, X_data, size=cfg.IMAGE_SIZE)

        if row_idx == 0:
            axes[row_idx, 0].set_title("Immagine")
        axes[row_idx, 0].imshow(image, cmap="gray")
        axes[row_idx, 0].axis("off")

        for col_idx, threshold in enumerate(MASK_THRESHOLD_LIST, start=1):
            mask = digit_mask(image, threshold, dilation=0)
            #  print for each threshold the pixel fraction selected by the mask
            print("threshold and mask.mean ",threshold, mask.mean())
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Soglia {threshold:.2f}")
            axes[row_idx, col_idx].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col_idx].axis("off")

        print(f"pos={position}, min={np.min(image):.3f}, "
              f"max={np.max(image):.3f}, "
              f"percentili={np.percentile(image, [0, 25, 50, 75, 90, 95, 99, 100])}"
        )

    plt.tight_layout()
    plt.show()
    plt.close(fig) 


def plot_metrics_all(filename=None,
                     save_path=None):
    """
        plot metrics for final report 
    """
    # load CSV
    df = pd.read_csv(filename)

    # filter valid row on deletion (exclude lr_local_map with NaN)
    # for CNN model
    valid = df.dropna(subset=["del_auc", "del_gain", "del_auc_random"]).copy()
    cnn_df = valid[valid["model"] == "cnn"].copy()

    # Create figure with 2 panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    group_order = ["both_ok", "lr_ko_cnn_ok", "lr_ok_cnn_ko", "both_ko"]
    tech_order = ["gradcam", "integrated_gradients", "occlusion", "lime", "shap"]

    # panel left : Deletion AUC
    sns.barplot( data=cnn_df,
                x="group",
                y="del_auc",
                hue="technique",
                order=group_order,
                hue_order=tech_order,
                errorbar="sd",
                ax=axes[0],
            )
            
    axes[0].set_title("CNN Deletion AUC (del_auc) per Gruppo e Tecnica")
    axes[0].set_xlabel("Gruppo")
    axes[0].set_ylabel("del_auc (media ± dev. std)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="Tecnica", loc="upper right")

    # panel right : Deletion Gain
    sns.barplot( data=cnn_df,
                x="group",
                y="del_gain",
                hue="technique",
                order=group_order,
                hue_order=tech_order,
                errorbar="sd",
                ax=axes[1],
            )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("CNN Deletion Gain (del_gain) per Gruppo e Tecnica")
    axes[1].set_xlabel("Gruppo")
    axes[1].set_ylabel("del_gain (media ± dev. std)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="Tecnica", loc="upper right")

    plt.tight_layout()
    
    if save_path is not None:
        # fig.savefig("xai_cnn_deletion_summary.png", dpi=150, bbox_inches="tight")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    plt.show()
    plt.close(fig)

    summary = ( df.dropna(subset=["del_auc", "del_gain"])
                .groupby( ["group", "model", "target_name", "technique"])
                        [
                            ["mask_share", "focus_ratio", "del_auc", "del_gain", "agreement_spearman_mean"]
                        ].agg(["count", "mean", "std", "median"])
            )
    print(summary.to_string())
    
    
def plot_mask_sensitivity( json_path,
                            X_data,
                            cfg,
                            thresholds=(0.1, 0.2),
                            dilations=(0, 1),
                            save_path=None,
                        ):
    """
        Detect masks for all JSON images and compare mean selected pixel fractions
        across threshold and dilation values.

        Return Value:
        summary_df: summary statistics for config.
        masks: indexed masks (index, threshold, dilation)
        fig: matplotlib Figure
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images = data.get("images", [])

    if not images:
        raise ValueError(f"Nessuna immagine trovata in {json_path}")

    rows = []
    masks = {}

    for item in images:
        image_index = int(item["index"])
        position = int(item["pos"])

        # Same image recovery used by XAI phases 1 and 2
        image = img28(position, X_data, size=cfg.IMAGE_SIZE)

        for threshold in thresholds:
            for dilation in dilations:
                mask = digit_mask(image, threshold, dilation)

                # Handle both NumPy arrays and PyTorch tensors
                if hasattr(mask, "detach"):
                    mask = mask.detach().cpu().numpy()
                mask = np.asarray(mask).squeeze().astype(bool)

                masks[(image_index, threshold, dilation)] = mask

                rows.append({"index": image_index,
                                    "threshold": threshold,
                                    "dilation": dilation,
                                    "mask_fraction": float(mask.mean()),
                                })

    for_image_df = pd.DataFrame(rows)

    summary_df = ( for_image_df.groupby(["threshold", "dilation"])["mask_fraction"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                 )

    fig, ax = plt.subplots(figsize=(7, 5))

    for dilation in dilations:
        subset = summary_df[summary_df["dilation"] == dilation]
        ax.errorbar( subset["threshold"],
                    subset["mean"],
                    yerr=subset["std"].fillna(0),
                    marker="o",
                    capsize=4,
                    label=f"dilation={dilation}",
                )

    ax.set_title("Sensibilità della frazione di maschera")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Frazione media di pixel selezionati")
    ax.set_xticks(list(thresholds))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dilation")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    # return summary_df, masks, fig  
    plt.close(fig) 
    return summary_df, masks
    
    
def plot_masks_for_image( json_path,
                        image_index,
                        X_data,
                        cfg,
                        masks,
                        thresholds=(0.1, 0.2),
                        dilations=(0, 1),
                        save_path=None,
                    ):
    """
        This function views the original image with mask overlaps
        for every pair of threshold and dilation.
    """

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images = data.get("images", [])

    item = next( (row for row in images 
                            if int(row["index"]) == int(image_index)),
                                None,
                )
                
    if item is None:
        raise KeyError(f"Indice immagine {image_index} non trovato in {json_path}")

    position = int(item["pos"])
    image = img28(position, X_data, size=cfg.IMAGE_SIZE)

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.asarray(image).squeeze()

    configurations = [ (threshold, dilation)
                        for threshold in thresholds
                        for dilation in dilations
                    ]

    fig, axes = plt.subplots( 1,len(configurations),figsize=(4 * len(configurations), 4),squeeze=False,)
    axes = axes.ravel()

    for ax, (threshold, dilation) in zip(axes, configurations):
        key = (int(image_index), threshold, dilation)

        if key not in masks:
            raise KeyError(
                f"Maschera non trovata per index={image_index}, "
                f"threshold={threshold}, dilation={dilation}. "
                "Verifica che thresholds e dilations siano gli stessi "
                "usati in plot_mask_sensitivity."
            )

        mask = np.asarray(masks[key]).squeeze().astype(bool)

        if mask.shape != image.shape:
            raise ValueError(f"Shape immagine {image.shape} diversa da quella della "
                             f"maschera {mask.shape} per {key}"
                            )

        ax.imshow(image, cmap="gray")
        overlay = np.ma.masked_where(~mask, mask)
        ax.imshow(overlay, cmap="Reds", alpha=0.45, vmin=0, vmax=1)

        ax.set_title(f"threshold={threshold}, dilation={dilation}\n"
                    f"mask.mean()={mask.mean():.3f}"
                    )
        ax.axis("off")

    fig.suptitle(f"Maschere per immagine index={image_index}",
                    fontsize=14,
                )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    # return fig
    plt.close(fig) 
    
    


    









