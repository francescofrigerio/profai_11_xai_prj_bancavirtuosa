"""Do the techniques look at the same pixels? (numpy + scipy only)

Pipeline for every image and technique:
  1. native map -> 28x28 grid (E03): bilinear to the DenseNet size if smaller (Grad-CAM 4x4), then
     the mean over the area; the LR local map is already 28x28 and is left untouched;
  2. sign mode (SIGN_MODE): absolute value, or the positive part only;
  3. restrict to the region (MASK_MODE): the digit mask (threshold, dilated) or the whole image;
  4. Spearman correlation of the values and IoU of the top TOPK_FRACTION pixels (k = ceil(f * n)).

Ties are not hidden: Spearman uses average ranks, and when the top-k threshold falls inside a tie
group the IoU is flagged (the choice among tied pixels is arbitrary but deterministic).
The resampling is written in numpy so that it can be tested without torch; a test checks it
against torch.nn.functional.interpolate.
"""
from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------------------
# Resampling to the 28x28 grid
# ----------------------------------------------------------------------------------------
def _axis_bilinear(n_in: int, n_out: int):
    """Indices and weights of the bilinear interpolation with align_corners=False."""
    coords = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
    lo = np.floor(coords).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    return lo, hi, coords - lo


def bilinear_resize(m, size: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    r0, r1, wr = _axis_bilinear(m.shape[0], size)
    c0, c1, wc = _axis_bilinear(m.shape[1], size)
    wr, wc = wr[:, None], wc[None, :]
    return (m[np.ix_(r0, c0)] * (1 - wr) * (1 - wc) + m[np.ix_(r0, c1)] * (1 - wr) * wc
            + m[np.ix_(r1, c0)] * wr * (1 - wc) + m[np.ix_(r1, c1)] * wr * wc)


def area_resize(m, out: int) -> np.ndarray:
    """Mean over the area of each output cell (torch's adaptive average pooling)."""
    m = np.asarray(m, dtype=np.float64)
    n = m.shape[0]
    idx = np.arange(out)
    starts, ends = (idx * n) // out, -((-(idx + 1) * n) // out)
    rows = np.stack([m[s:e].mean(axis=0) for s, e in zip(starts, ends)])
    return np.stack([rows[:, s:e].mean(axis=1) for s, e in zip(starts, ends)], axis=1)


def to_grid(map2d, densenet_size: int, side: int = 28) -> np.ndarray:
    """Native map -> (side, side). A map that already is (side, side) is returned as it is."""
    m = np.asarray(map2d, dtype=np.float64)
    if m.shape == (side, side):
        return m
    if m.shape[0] > densenet_size:
        raise ValueError(f"map of size {m.shape[0]} is larger than the DenseNet input ({densenet_size})")
    if m.shape[0] < densenet_size:
        m = bilinear_resize(m, densenet_size)
    return area_resize(m, side)


# ----------------------------------------------------------------------------------------
# Masks and sign mode
# ----------------------------------------------------------------------------------------
def digit_mask(img28, threshold: float, dilation: int) -> np.ndarray:
    """Pixels above `threshold`, dilated by `dilation` pixels (3x3 square element, E17)."""
    from scipy.ndimage import binary_dilation
    mask = np.asarray(img28) > threshold
    if dilation > 0:
        mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=dilation)
    return mask


def region_mask(img28, cfg) -> np.ndarray:
    """The region where agreement is measured: MASK_MODE 'digit' or 'full'."""
    if cfg.MASK_MODE == "full":
        return np.ones(np.asarray(img28).shape, dtype=bool)
    mask = digit_mask(img28, cfg.MASK_THRESHOLD, cfg.MASK_DILATION)
    if not mask.any():
        raise ValueError("the digit mask is empty: check MASK_THRESHOLD")
    return mask


def apply_sign_mode(m, mode: str) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if mode == "absolute":
        return np.abs(m)
    if mode == "positive":
        return np.clip(m, 0, None)
    raise ValueError(f"unknown SIGN_MODE {mode!r}")


# ----------------------------------------------------------------------------------------
# The two agreement measures
# ----------------------------------------------------------------------------------------
def spearman(a, b) -> float:
    """Spearman correlation with average ranks; NaN if either input is constant."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def top_k(values, fraction: float) -> tuple:
    """(indices of the top-k, tie flag). k = ceil(fraction * n); the flag is True when the
    threshold falls inside a group of equal values."""
    v = np.asarray(values, dtype=np.float64).ravel()
    k = max(1, math.ceil(fraction * len(v)))
    order = np.argsort(-v, kind="stable")
    tied = bool(k < len(v) and v[order[k - 1]] == v[order[k]])
    return order[:k], tied


def topk_iou(a, b, fraction: float) -> tuple:
    """(IoU of the two top-k sets, tie flag of a, tie flag of b)."""
    ia, ta = top_k(a, fraction)
    ib, tb = top_k(b, fraction)
    sa, sb = set(ia.tolist()), set(ib.tolist())
    return len(sa & sb) / len(sa | sb), ta, tb


def expected_random_iou(n_pixels: int, fraction: float) -> float:
    """Expected IoU of two independent random top-k sets (the overlap is hypergeometric)."""
    from scipy.stats import hypergeom
    k = max(1, math.ceil(fraction * n_pixels))
    x = np.arange(0, k + 1)
    return float(np.sum(hypergeom.pmf(x, n_pixels, k, k) * x / (2 * k - x)))


def compare_pair(map_a28, map_b28, mask, cfg) -> dict:
    """Both measures for one image, on the region `mask` and after the sign mode."""
    a = apply_sign_mode(map_a28, cfg.SIGN_MODE)[mask]
    b = apply_sign_mode(map_b28, cfg.SIGN_MODE)[mask]
    iou, tie_a, tie_b = topk_iou(a, b, cfg.TOPK_FRACTION)
    return {"spearman": spearman(a, b), "iou": iou, "tie_a": tie_a, "tie_b": tie_b}


# ----------------------------------------------------------------------------------------
# Matrices over a sample
# ----------------------------------------------------------------------------------------
def agreement_matrices(maps28: dict, masks, cfg) -> dict:
    """maps28: name -> (N, 28, 28); masks: (N, 28, 28) bool. Means and standard deviations over the
    images, for every pair of techniques (diagonal included: a map agrees with itself)."""
    names = list(maps28)
    n_img, k = len(masks), len(names)
    sp = np.full((k, k, n_img), np.nan)
    io = np.full((k, k, n_img), np.nan)
    ties = np.zeros((k, n_img), dtype=bool)
    for i in range(k):
        for j in range(i, k):
            for n in range(n_img):
                r = compare_pair(maps28[names[i]][n], maps28[names[j]][n], masks[n], cfg)
                sp[i, j, n] = sp[j, i, n] = r["spearman"]
                io[i, j, n] = io[j, i, n] = r["iou"]
                if i == j:
                    ties[i, n] = r["tie_a"]
    with np.errstate(all="ignore"):
        return {
            "names": names, "n_images": n_img,
            "spearman_mean": np.nanmean(sp, axis=2), "spearman_std": np.nanstd(sp, axis=2),
            "iou_mean": np.nanmean(io, axis=2), "iou_std": np.nanstd(io, axis=2),
            "n_valid_spearman": np.sum(~np.isnan(sp), axis=2), "tie_rate": ties.mean(axis=1),
        }


def random_reference(maps28: dict, masks, cfg, seed: int) -> dict:
    """Agreement of every technique with a RANDOM map (a fresh one per image): the floor that any
    real agreement has to be compared with. Expected: Spearman ~ 0, IoU ~ expected_random_iou."""
    out = {}
    for name, maps in maps28.items():
        sp, io = [], []
        for n in range(len(masks)):
            rnd = np.random.RandomState(seed + n).uniform(size=masks[n].shape)
            r = compare_pair(maps[n], rnd, masks[n], cfg)
            sp.append(r["spearman"])
            io.append(r["iou"])
        out[name] = {"spearman_mean": float(np.nanmean(sp)), "iou_mean": float(np.nanmean(io))}
    return out


def outside_digit_share(map28, mask) -> float:
    """Share of the attribution mass (absolute value) falling OUTSIDE the digit mask. NaN for an
    all-zero map. Integrated Gradients gives ~0 by construction (black baseline)."""
    a = np.abs(np.asarray(map28, dtype=np.float64))
    total = a.sum()
    return float("nan") if total == 0 else float(a[~mask].sum() / total)


def outside_digit_table(maps28: dict, masks) -> dict:
    """name -> (mean, std) of the outside share over the images. The mask is the dilated digit mask,
    whatever MASK_MODE says (E09)."""
    table = {}
    for name, maps in maps28.items():
        shares = [outside_digit_share(maps[n], masks[n]) for n in range(len(masks))]
        table[name] = (float(np.nanmean(shares)), float(np.nanstd(shares)))
    return table
