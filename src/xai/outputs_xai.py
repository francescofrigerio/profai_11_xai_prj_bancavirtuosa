"""
     Plot Graph for Xai
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython.display import Image, display
from utility import img28
from constants import TITLES
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


def format_json_xai_rows(rows):
    """
        Converts XAI rows into a JSON-serializable structure.

        Includes:
        CNN maps for each target and technique, containing the map itself, 
        signed values, and all associated metadata;
        LR maps for each target;
        analysis metrics and metadata;
        sign diagnostics.
    """

    formatted_images = []

    for row in rows:
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

        img_entry = { "index": to_json_value(row["index"]),
                    "pos": to_json_value(row["pos"]),
                    "true": to_json_value(row["true"]),
                    "pred_cnn": to_json_value(row["pred_cnn"]),
                    "pred_lr": to_json_value(row["pred_lr"]),
                    "prob_pred": to_json_value(row.get("prob_pred")),

                    "targets_cnn": to_json_value(row.get("targets_cnn", {})),
                    "targets_lr": to_json_value(row.get("targets_lr", {})),

                    "maps_cnn": maps_cnn_json,
                    "maps_lr": maps_lr_json,

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
                    "agreement_spearman_matrix": to_json_value(row.get(row.get("meta_cnn", {})
                                                            .get(target_name, {})
                                                            .get("agreement_spearman_matrix")
                                                            .get("matrix"), {})),
                    "agreement_spearman": to_json_value(row.get(row.get("meta_cnn", {})
                                                                   .get(target_name, {})
                                                                   .get("agreement_spearman"), {})),
                    # "agreement_matrix": row["meta_cnn"][target_name]["agreement_spearman_matrix"]["matrix"],
                    # "agreement_spearman": row["meta_cnn"][target_name]["agreement_spearman_matrix"]["agreement_spearman"],
        }

        formatted_images.append(img_entry)

    return formatted_images


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
        fig.clear()

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
        fig.clear()


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
            fig.clear()
