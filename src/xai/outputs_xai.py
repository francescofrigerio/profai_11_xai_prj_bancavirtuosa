"""
     Plot Graph for Xai
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython.display import Image, display
from utility import img28
from constants import TITLES

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
