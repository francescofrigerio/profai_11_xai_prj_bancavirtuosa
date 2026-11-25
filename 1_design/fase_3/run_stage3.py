"""Stage 3 driver (TEMPORARY: the final notebook of stage 4 absorbs it).

Real run on Colab with the trained checkpoint:
  1. quantitative sample (N_QUANT test images, stratified by the predicted class): the six techniques
     plus the LR local map, on the predicted class -> Spearman / IoU matrices (7x7), random reference,
     table of the attribution outside the digit, saved maps;
  2. Occlusion diagnosis (why is it negative on some digits?);
  3. gallery of GALLERY_SIZE images (errors are explained for the predicted AND the true class);
  4. deletion curves only if COMPUTE_FAITHFULNESS is True.

    %run run_stage3.py                      # full run (~20 minutes on a T4)
    %run run_stage3.py --n-quant 20 --no-gallery     # quick smoke run (a few minutes)
"""
import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from baseline import fit_baseline
from checks import Verifications
from comparison import (agreement_matrices, digit_mask, expected_random_iou, outside_digit_table,
                        random_reference, region_mask, to_grid)
from config import CONFIG
from data import black_normalized, load_mnist, to_densenet_input, to_lr_input
from faithfulness import deletion_for_techniques
from metrics import accuracy, quadrants
from model import predict_logits, softmax_np
from sampling import select_gallery, select_quant_sample
from training import get_trained_model, pick_device
from xai import TECHNIQUE_NAMES, build_techniques
from xai.debug import make_gradcam_debug_callback
from xai.shap_explainer import make_background

TITLES = {"gradcam": "Grad-CAM", "integrated_gradients": "Integrated\nGradients", "occlusion": "Occlusion",
          "saliency": "Saliency", "lime": "LIME", "shap": "SHAP", "lr_local": "LR local\n(w * x)"}
RANDOM_SPEARMAN_TOL = 0.05
RANDOM_IOU_TOL = 0.02
TIE_RATE_WARN = 0.5


def explain_items(items, techniques, data, X_flat_test, lr, cfg, device, keep_native=False, label=""):
    """items: list of (test index, target class). Returns one dict per item."""
    out, t0 = [], time.time()
    for n, (idx, target) in enumerate(items):
        x1 = to_densenet_input(data.X_test[idx:idx + 1], cfg.DENSENET_SIZE).to(device)
        native, grid = {}, {}
        for name, technique in techniques.items():
            result = technique.explain(x1, target)
            grid[name] = to_grid(result.map, cfg.DENSENET_SIZE)
            if keep_native:
                native[name] = result.map
        lr_map = lr.local_map(X_flat_test[idx], target)
        grid["lr_local"] = lr_map
        out.append({"index": int(idx), "target": int(target), "grid": grid, "native": native, "lr_map": lr_map})
        if (n + 1) % 10 == 0 or n + 1 == len(items):
            print(f"  {label}: {n + 1}/{len(items)} explained ({time.time() - t0:.0f}s)")
    return out


def plot_matrices(agg, path):
    from matplotlib.figure import Figure
    names = [TITLES[n].replace("\n", " ") for n in agg["names"]]
    fig = Figure(figsize=(14, 6))
    for ax, key, std_key, title in zip(fig.subplots(1, 2), ("spearman_mean", "iou_mean"),
                                       ("spearman_std", "iou_std"), ("Spearman (mean +/- std)", "Top-20% IoU (mean +/- std)")):
        M = agg[key]
        im = ax.imshow(M, cmap="viridis", vmin=min(0.0, float(np.nanmin(M))), vmax=1.0)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}\n±{agg[std_key][i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=100, bbox_inches="tight")


def plot_rows(rows, names, signed, path, title):
    from matplotlib.figure import Figure
    cols = len(names) + 2
    fig = Figure(figsize=(2.0 * cols, 2.0 * len(rows) + 0.5))
    axes = fig.subplots(len(rows), cols, squeeze=False)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image28"], cmap="gray")
        axes[r, 0].set_ylabel(row["label"], fontsize=6)
        cells = [(row["maps"][n], signed[n]) for n in names] + [(row["lr_map"], True)]
        for c, (m, is_signed) in enumerate(cells, start=1):
            vmax = float(np.percentile(np.abs(m), 99)) or float(np.abs(m).max()) or 1.0     # percentile scale (E37)
            axes[r, c].imshow(m, cmap="seismic" if is_signed else "viridis", vmin=-vmax if is_signed else 0,
                              vmax=vmax, interpolation="nearest" if m.shape[0] <= 8 else "bilinear")
        for c in range(cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, t in enumerate(["image"] + [TITLES[n] for n in names] + [TITLES["lr_local"]]):
        axes[0, c].set_title(t, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=85, bbox_inches="tight")


def write_matrix_csv(path, names, M):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for n, row in zip(names, M):
            w.writerow([n] + [f"{v:.4f}" for v in row])


def occlusion_report(model, x28, target, cfg, device):
    """Logits with every occlusion window set to black: does covering a region RAISE the class logit?"""
    import torch
    size, w, s = cfg.DENSENET_SIZE, cfg.OCCLUSION_WINDOW, cfg.OCCLUSION_STRIDE
    x1 = to_densenet_input(x28, size).to(device)
    black = black_normalized(size, device)
    spots = [(r, c) for r in range(0, size - w + 1, s) for c in range(0, size - w + 1, s)]
    batch = x1.repeat(len(spots), 1, 1, 1)
    for k, (r, c) in enumerate(spots):
        batch[k, :, r:r + w, c:c + w] = black[0, :, r:r + w, c:c + w]
    with torch.no_grad():
        base = model(x1)[0].cpu().numpy()
        occluded = torch.cat([model(batch[i:i + 75]) for i in range(0, len(batch), 75)]).cpu().numpy()
        black_logits = model(black)[0].cpu().numpy()
    delta = base[target] - occluded[:, target]
    return {"logit": float(base[target]), "margin_to_runner_up": float(base[target] - np.delete(base, target).max()),
            "logit_of_the_black_image": float(black_logits[target]), "windows": len(spots),
            "share_of_windows_that_raise_the_logit": float((delta < 0).mean()),
            "min_delta": float(delta.min()), "max_delta": float(delta.max()), "mean_delta": float(delta.mean())}


def main(cfg=CONFIG, argv=None):
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-quant", type=int, default=0)
    parser.add_argument("--no-gallery", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.n_quant:
        cfg = dataclasses.replace(cfg, N_QUANT=args.n_quant)
    out = Path(cfg.OUTPUT_DIR) / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    v = Verifications()
    t_start = time.time()

    device, warning = pick_device(cfg)
    if warning:
        print("WARNING:", warning)
        v.record("device", "WARN", warning)
    data = load_mnist(cfg)
    split = data.split
    model, info = get_trained_model(cfg, data, device, epoch_callback=make_gradcam_debug_callback(
        data, cfg, device, out / "debug"))
    v.record("model mode", "INFO", f"{info['action']} ({info['reason']}); sha256 {info['sha256'][:16]}...")
    y_test = data.y_test.numpy()
    logits = predict_logits(model, data.X_test, cfg, device)
    proba = softmax_np(logits)
    pred_d, conf_d = proba.argmax(1), proba.max(1)
    X_flat = to_lr_input(data.X_trainval)
    lr = fit_baseline(X_flat, data.y_trainval.numpy(), split.idx_train, cfg)
    X_flat_test = to_lr_input(data.X_test)
    pred_l = lr.predict(X_flat_test)
    acc_d, acc_l = accuracy(y_test, pred_d), accuracy(y_test, pred_l)
    v.record("test accuracy", "INFO", f"DenseNet {acc_d:.4f}, LR {acc_l:.4f}, quadrants {quadrants(y_test, pred_d, pred_l)}")

    background = make_background(data.X_trainval, split.idx_train, cfg, device)
    techniques = build_techniques(model, cfg, background)
    signed = {n: t.signed for n, t in techniques.items()}

    # ---- 1. quantitative sample
    picks = select_quant_sample(pred_d, cfg.N_QUANT, cfg.NUM_CLASSES, cfg.SEED)
    v.expect("quantitative sample is stratified by the predicted class",
             bool((np.bincount(pred_d[picks], minlength=cfg.NUM_CLASSES) == cfg.N_QUANT // cfg.NUM_CLASSES).all()),
             f"{len(picks)} images, {cfg.N_QUANT // cfg.NUM_CLASSES} per predicted class")
    quant = explain_items([(i, int(pred_d[i])) for i in picks], techniques, data, X_flat_test, lr, cfg, device,
                          label="quantitative sample")
    names = list(TECHNIQUE_NAMES) + ["lr_local"]
    maps28 = {n: np.stack([e["grid"][n] for e in quant]) for n in names}
    images28 = np.stack([data.X_test[e["index"], 0].numpy() for e in quant])
    region = np.stack([region_mask(im, cfg) for im in images28])
    digit = np.stack([digit_mask(im, cfg.MASK_THRESHOLD, cfg.MASK_DILATION) for im in images28])
    np.savez_compressed(out / "maps28_quant.npz", indices=picks, pred=pred_d[picks], true=y_test[picks],
                        images28=images28, **maps28)

    agg = agreement_matrices(maps28, region, cfg)
    ref = random_reference(maps28, region, cfg, cfg.SEED)
    outside = outside_digit_table(maps28, digit)
    expected_iou = float(np.mean([expected_random_iou(int(m.sum()), cfg.TOPK_FRACTION) for m in region]))
    plot_matrices(agg, out / "agreement_matrices.png")
    write_matrix_csv(out / "agreement_spearman.csv", names, agg["spearman_mean"])
    write_matrix_csv(out / "agreement_iou.csv", names, agg["iou_mean"])
    with open(out / "outside_digit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique", "outside_share_mean", "outside_share_std"])
        for n in names:
            w.writerow([n, f"{outside[n][0]:.4f}", f"{outside[n][1]:.4f}"])

    # ---- verifications of the measures themselves
    v.expect("a map agrees with itself", bool(np.allclose(np.diag(agg["spearman_mean"]), 1.0, atol=1e-6)
                                              and np.allclose(np.diag(agg["iou_mean"]), 1.0)), "diagonals equal 1")
    v.expect("random reference: Spearman ~ 0", all(abs(r["spearman_mean"]) < RANDOM_SPEARMAN_TOL for r in ref.values()),
             "; ".join(f"{n} {r['spearman_mean']:+.3f}" for n, r in ref.items()))
    v.expect("random reference: IoU ~ theory", all(abs(r["iou_mean"] - expected_iou) < RANDOM_IOU_TOL for r in ref.values()),
             f"theory {expected_iou:.3f}; " + "; ".join(f"{n} {r['iou_mean']:.3f}" for n, r in ref.items()))
    v.expect("every Spearman pair is defined on every image", bool((agg["n_valid_spearman"] == agg["n_images"]).all()),
             f"minimum valid images per pair: {int(agg['n_valid_spearman'].min())} of {agg['n_images']}")
    for n, rate in zip(names, agg["tie_rate"]):
        v.expect(f"{n}: top-k threshold inside a tie group", rate <= TIE_RATE_WARN,
                 f"in {rate:.0%} of the images (the choice among tied pixels is arbitrary but deterministic)")
    lr_i = names.index("lr_local")
    v.record("agreement with the LR local map (headline)", "INFO", "; ".join(
        f"{n}: rho {agg['spearman_mean'][names.index(n), lr_i]:.2f}, IoU {agg['iou_mean'][names.index(n), lr_i]:.2f}"
        for n in TECHNIQUE_NAMES))
    v.record("attribution outside the digit (E26 cautions apply)", "INFO", "; ".join(
        f"{n} {outside[n][0]:.0%}" for n in names))

    # ---- 2. Occlusion diagnosis
    neg_share = {}
    for c in range(cfg.NUM_CLASSES):
        sel = [k for k, e in enumerate(quant) if e["target"] == c]
        m = maps28["occlusion"][sel]
        neg_share[c] = float(np.clip(-m, 0, None).sum() / np.abs(m).sum())
    v.record("Occlusion: share of NEGATIVE mass by predicted class", "INFO",
             ", ".join(f"{c}: {s:.0%}" for c, s in neg_share.items()))
    diagnostics = {}
    for c in (5, 7, 0):
        sel = [k for k, e in enumerate(quant) if e["target"] == c][:3]
        diagnostics[c] = [occlusion_report(model, data.X_test[quant[k]["index"]:quant[k]["index"] + 1], c, cfg, device)
                          for k in sel]
        v.record(f"Occlusion windows, class {c}", "INFO", "; ".join(
            f"logit {d['logit']:.1f} (margin {d['margin_to_runner_up']:.1f}, black image {d['logit_of_the_black_image']:.1f}): "
            f"{d['share_of_windows_that_raise_the_logit']:.0%} of the windows RAISE it" for d in diagnostics[c]))

    # ---- 3. deletion (only with the flag)
    predict_fn = lambda batch: softmax_np(predict_logits(model, torch.from_numpy(np.asarray(batch, dtype=np.float32)), cfg, device))
    deletion = deletion_for_techniques(cfg, predict_fn, maps28, images28, [e["target"] for e in quant], region, cfg.SEED)
    v.record("deletion curves", "INFO" if deletion is None else "PASS",
             "not computed (COMPUTE_FAITHFULNESS=False)" if deletion is None else
             "mean AUC, lower is better: " + ", ".join(f"{k} {a:.3f}" for k, a in deletion.items()))

    # ---- 4. gallery
    gallery_info = None
    if not args.no_gallery:
        gallery = select_gallery(y_test, pred_d, pred_l, conf_d, cfg.NUM_CLASSES, cfg.SEED, cfg.GALLERY_SIZE)
        roles = [g["role"] for g in gallery]
        v.expect("gallery composition", len(gallery) == cfg.GALLERY_SIZE and len({g["index"] for g in gallery}) == len(gallery),
                 f"{len(gallery)} different images: {roles.count('correct')} correct, {roles.count('confident_error')} "
                 f"confident errors, {sum(r.startswith('quadrant') for r in roles)} from the quadrants")
        items = [(g["index"], t) for g in gallery for t in g["targets"]]
        explained = explain_items(items, techniques, data, X_flat_test, lr, cfg, device, keep_native=True, label="gallery")
        by_key = {(e["index"], e["target"]): e for e in explained}
        groups = {"correct": [], "errors": [], "quadrants": []}
        for g in gallery:
            for t in g["targets"]:
                e = by_key[(g["index"], t)]
                kind = "predicted" if t == g["pred"] else "true"
                groups["correct" if g["role"] == "correct" else "errors" if g["role"] == "confident_error" else "quadrants"].append({
                    "image28": data.X_test[g["index"], 0].numpy(), "maps": e["native"], "lr_map": e["lr_map"],
                    "label": f"{g['role'].replace('quadrant:', '')}\n#{g['index']} true {g['true']} / pred {g['pred']}\n"
                             f"explained: {t} ({kind})"})
        for key, rows in groups.items():
            plot_rows(rows, list(TECHNIQUE_NAMES), signed, out / f"gallery_{key}.png", f"Gallery: {key}")
        gallery_info = [{k: g[k] for k in ("index", "role", "true", "pred", "confidence", "targets")} for g in gallery]

    results = {
        "model": {k: info[k] for k in ("action", "reason", "sha256", "device")}, "n_quant": cfg.N_QUANT,
        "agreement": {k: (agg[k].tolist() if hasattr(agg[k], "tolist") else agg[k]) for k in agg},
        "random_reference": ref, "expected_random_iou": expected_iou,
        "outside_digit": {n: {"mean": a, "std": s} for n, (a, s) in outside.items()},
        "occlusion_negative_share_by_class": neg_share, "occlusion_windows": diagnostics,
        "deletion_auc": deletion, "gallery": gallery_info, "total_seconds": time.time() - t_start,
    }
    (out / "stage3_results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "verifications.md").write_text(v.to_markdown() + "\n")
    print("\n" + v.to_markdown())
    print(f"\nworst status: {v.worst()} | total {time.time() - t_start:.0f}s")
    return results, v


if __name__ == "__main__":
    main()
