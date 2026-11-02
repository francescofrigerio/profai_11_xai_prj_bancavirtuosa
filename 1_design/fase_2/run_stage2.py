"""Stage 2 driver (TEMPORARY: the final notebook of stage 4 absorbs it).

Real run on Colab with the trained checkpoint (MODE: load): the six techniques plus the LR local
map on the 10 fixed validation images (one per class), explaining the PREDICTED class. It writes a
grid of maps, the timings and a 'Verifiche' table.

    %run run_stage2.py

If there is no compatible checkpoint it trains (and saves the Grad-CAM debug images of the first and
last epoch in outputs/stage2/debug): nothing changes silently, the MODE line says what happened.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np

from baseline import fit_baseline
from checks import Verifications
from config import CONFIG
from data import debug_indices, load_mnist, to_densenet_input, to_lr_input
from training import get_trained_model, pick_device
from xai import TECHNIQUE_NAMES, build_techniques
from xai.debug import make_gradcam_debug_callback
from xai.shap_explainer import make_background

MAX_SECONDS_PER_IMAGE = 15.0          # per technique, from phase 0
IG_COMPLETENESS_MAX = 0.05
RELU_MAX_REL_DIFF = 1e-3
TITLES = {"gradcam": "Grad-CAM", "integrated_gradients": "Integrated\nGradients", "occlusion": "Occlusion",
          "saliency": "Saliency", "lime": "LIME", "shap": "SHAP"}


def digit_mask(img28, threshold, dilation):
    from scipy.ndimage import binary_dilation
    mask = np.asarray(img28) > threshold
    return binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=dilation) if dilation else mask


def to_grid28(map2d, size, side=28):
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(np.asarray(map2d), dtype=torch.float32)[None, None]
    if t.shape[-1] < size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return F.interpolate(t, size=(side, side), mode="area")[0, 0].numpy()


def main(cfg=CONFIG):
    import torch
    from matplotlib.figure import Figure
    from model import SafeDenseNet
    out = Path(cfg.OUTPUT_DIR) / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    v = Verifications()
    t_start = time.time()

    device, warning = pick_device(cfg)
    if warning:
        print("WARNING:", warning)
        v.record("device", "WARN", warning)
    data = load_mnist(cfg)
    split = data.split
    callback = make_gradcam_debug_callback(data, cfg, device, out / "debug")
    model, info = get_trained_model(cfg, data, device, epoch_callback=callback)
    v.record("model mode", "INFO", f"{info['action']} ({info['reason']}); checkpoint sha256 {info['sha256'][:16]}...")
    background = make_background(data.X_trainval, split.idx_train, cfg, device)
    techniques = build_techniques(model, cfg, background)

    y = data.y_trainval.numpy()
    X_flat = to_lr_input(data.X_trainval)
    lr = fit_baseline(X_flat, y, split.idx_train, cfg)
    picks = debug_indices(y, split.idx_val, cfg.NUM_DEBUG_IMAGES)

    rows, seconds, ig_errors = [], {n: [] for n in techniques}, []
    for i in picks:
        x1 = to_densenet_input(data.X_trainval[i:i + 1], cfg.DENSENET_SIZE).to(device)
        with torch.no_grad():
            pred = int(model(x1).argmax(1))
        maps = {}
        for name, technique in techniques.items():
            result = technique.explain(x1, pred)
            maps[name] = result
            seconds[name].append(result.seconds)
            if name == "integrated_gradients":
                ig_errors.append(result.meta["completeness_error"])
        rows.append({"index": int(i), "true": int(y[i]), "pred": pred, "maps": maps,
                     "lr_map": lr.local_map(X_flat[i], pred)})
        print(f"image {i} (true {y[i]}, predicted {pred}): " +
              ", ".join(f"{n} {maps[n].seconds:.2f}s" for n in maps))

    # ---- verifications
    v.record("techniques run", "PASS", f"{len(techniques)} techniques on {len(rows)} images, every map "
             "finite, 2D and within the contract (the contract raises otherwise)")
    median = {n: float(np.median(s)) for n, s in seconds.items()}
    for n, m in median.items():
        v.expect(f"{n}: time per image", m <= MAX_SECONDS_PER_IMAGE, f"median {m:.2f}s (limit {MAX_SECONDS_PER_IMAGE:.0f}s)")
    v.expect("Integrated Gradients completeness", max(ig_errors) <= IG_COMPLETENESS_MAX,
             f"worst relative error {max(ig_errors):.2%} over {len(ig_errors)} images (limit {IG_COMPLETENESS_MAX:.0%})")
    first = torch.as_tensor(to_densenet_input(data.X_trainval[picks[0]:picks[0] + 1], cfg.DENSENET_SIZE)).to(device)
    for name in ("lime", "shap"):
        again = techniques[name].explain(first, rows[0]["pred"]).map
        ref = rows[0]["maps"][name].map
        v.expect(f"{name} is reproducible", bool(np.allclose(ref, again, rtol=1e-3, atol=1e-4 * np.abs(ref).max())),
                 "same seed, same map on the first image")
    if cfg.FLAG_RELU_INPLACE:
        patched = SafeDenseNet(copy.deepcopy(model)).eval()
        for name in ("gradcam", "integrated_gradients", "saliency"):
            a = rows[0]["maps"][name].map
            b = build_techniques(patched, cfg, names=(name,))[name].explain(first, rows[0]["pred"]).map
            diff = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
            v.expect(f"in-place ReLU gives the same {name} map on the trained model", diff <= RELU_MAX_REL_DIFF,
                     f"relative difference {diff:.1e}")
    else:
        v.record("in-place ReLU equivalence", "INFO", "FLAG_RELU_INPLACE=False: nothing to compare")
    shares = []
    for row in rows:
        mask = digit_mask(data.X_trainval[row["index"], 0].numpy(), cfg.MASK_THRESHOLD, cfg.MASK_DILATION)
        g = to_grid28(np.abs(row["maps"]["saliency"].map), cfg.DENSENET_SIZE)
        shares.append((float(g[mask].sum() / g.sum()), float(mask.mean())))
    v.record("saliency on the digit (information)", "INFO",
             f"{np.mean([s for s, _ in shares]):.0%} of its mass inside the digit mask, a random map would give "
             f"{np.mean([a for _, a in shares]):.0%}")

    # ---- figure
    names = list(techniques)
    fig = Figure(figsize=(2.1 * (len(names) + 2), 2.1 * len(rows)))
    axes = fig.subplots(len(rows), len(names) + 2)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(data.X_trainval[row["index"], 0], cmap="gray")
        axes[r, 0].set_ylabel(f"true {row['true']}\npred {row['pred']}", fontsize=8)
        cells = [(row["maps"][n].map, row["maps"][n].signed) for n in names] + [(row["lr_map"], True)]
        for c, (m, signed) in enumerate(cells, start=1):
            a = float(np.abs(m).max()) or 1.0
            axes[r, c].imshow(m, cmap="seismic" if signed else "viridis", vmin=-a if signed else 0, vmax=a,
                              interpolation="nearest" if m.shape[0] <= 8 else "bilinear")
        for c in range(len(names) + 2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, title in enumerate(["image"] + [TITLES[n] for n in names] + ["LR local\n(w * x)"]):
        axes[0, c].set_title(title, fontsize=9)
    fig.savefig(out / "maps_grid.png", dpi=90, bbox_inches="tight")
    try:
        from IPython.display import Image, display
        display(Image(str(out / "maps_grid.png")))
    except Exception:
        pass

    results = {
        "model": {k: info[k] for k in ("action", "reason", "sha256", "device")},
        "median_seconds_per_image": median, "ig_completeness_errors": ig_errors,
        "images": [{"index": r["index"], "true": r["true"], "pred": r["pred"],
                    "native_shapes": {n: list(r["maps"][n].map.shape) for n in names}} for r in rows],
        "per_image_all_techniques_seconds": float(sum(median.values())), "total_seconds": time.time() - t_start,
    }
    (out / "stage2_results.json").write_text(json.dumps(results, indent=2))
    (out / "verifications.md").write_text(v.to_markdown() + "\n")
    print("\n" + v.to_markdown())
    print(f"\nmedian seconds per image: " + ", ".join(f"{n} {m:.2f}" for n, m in median.items()))
    print(f"all six techniques: {sum(median.values()):.1f}s per image | worst status: {v.worst()} | total {time.time() - t_start:.0f}s")
    return results, v


if __name__ == "__main__":
    main()
