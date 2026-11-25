"""comparison.py: the measures give the right answer on cases worked out by hand, and the random
reference gives the answer theory predicts (a result that cannot be wrong would prove nothing)."""
import dataclasses
import math

import numpy as np
import pytest

from helpers import tiny_cfg
from comparison import (agreement_matrices, apply_sign_mode, area_resize, bilinear_resize, compare_pair,
                        digit_mask, expected_random_iou, outside_digit_share, outside_digit_table,
                        random_reference, region_mask, spearman, to_grid, top_k, topk_iou)

CFG = tiny_cfg()


# ---------------------------------------------------------------- resampling (rule E03)
def test_resampling_keeps_every_pixel_in_place():
    rng = np.random.RandomState(0)
    spots = [(0, 0), (0, 27), (27, 0), (27, 27)] + [tuple(rng.randint(0, 28, 2)) for _ in range(150)]
    for (i, j) in spots:                                              # 28 -> 128 -> 28: a bright pixel stays put
        delta = np.zeros((28, 28))
        delta[i, j] = 1.0
        back = area_resize(bilinear_resize(delta, 128), 28)
        assert np.unravel_index(back.argmax(), back.shape) == (i, j)
    for i in range(4):                                                # each 4x4 Grad-CAM cell peaks in its own 7x7 block
        for j in range(4):
            cell = np.zeros((4, 4))
            cell[i, j] = 1.0
            g = to_grid(cell, 128)
            pi, pj = np.unravel_index(g.argmax(), g.shape)
            assert 7 * i <= pi < 7 * (i + 1) and 7 * j <= pj < 7 * (j + 1)


def test_resampling_special_cases():
    const = np.full((4, 4), 2.5)
    assert np.allclose(to_grid(const, 128), 2.5)                      # a constant map stays constant
    lr = np.random.RandomState(1).rand(28, 28)
    assert to_grid(lr, 128) is not None and np.array_equal(to_grid(lr, 128), lr)   # the LR map is untouched
    assert to_grid(np.zeros((128, 128)), 128).shape == (28, 28)
    with pytest.raises(ValueError):
        to_grid(np.zeros((200, 200)), 128)


def test_bilinear_reproduces_a_linear_ramp_exactly():
    """Pixel centres, not corners: out[k] must equal (k + 0.5) * n_in / n_out - 0.5, clipped to the image."""
    ramp = np.tile(np.arange(4.0), (4, 1))                              # value = column index
    up = bilinear_resize(ramp, 128)
    expected = np.clip((np.arange(128) + 0.5) * 4 / 128 - 0.5, 0, 3)
    assert np.allclose(up[0], expected, atol=1e-12) and np.allclose(up[:, 5], up[0, 5])
    mean_preserving = area_resize(np.tile(np.arange(128.0), (128, 1)), 28)     # area mean of a ramp = centre of the window
    starts, ends = (np.arange(28) * 128) // 28, -((-(np.arange(28) + 1) * 128) // 28)
    assert np.allclose(mean_preserving[0], [(s + e - 1) / 2 for s, e in zip(starts, ends)])


def test_numpy_resampling_equals_torch():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    rng = np.random.RandomState(0)
    for n_in, n_out in ((4, 128), (8, 128), (28, 128)):
        m = rng.rand(n_in, n_in)
        ref = F.interpolate(torch.from_numpy(m)[None, None], size=(n_out, n_out), mode="bilinear",
                            align_corners=False)[0, 0].numpy()
        assert np.abs(bilinear_resize(m, n_out) - ref).max() < 1e-9
    m = rng.rand(128, 128)
    ref = F.interpolate(torch.from_numpy(m)[None, None], size=(28, 28), mode="area")[0, 0].numpy()
    assert np.abs(area_resize(m, 28) - ref).max() < 1e-9


# ---------------------------------------------------------------- masks and sign mode
def test_masks_and_sign_modes():
    img = np.zeros((28, 28))
    img[10, 10] = 0.5
    img[5, 5] = 0.05                                                  # below the threshold
    assert [digit_mask(img, 0.1, d).sum() for d in (0, 1, 2)] == [1, 9, 25]
    assert region_mask(img, CFG).sum() == 9                            # threshold 0.1, dilation 1
    assert region_mask(img, dataclasses.replace(CFG, MASK_MODE="full")).sum() == 784
    with pytest.raises(ValueError):
        region_mask(np.zeros((28, 28)), CFG)                           # an empty mask is an error, not a result
    m = np.array([-2.0, 0.0, 3.0])
    assert apply_sign_mode(m, "absolute").tolist() == [2.0, 0.0, 3.0]
    assert apply_sign_mode(m, "positive").tolist() == [0.0, 0.0, 3.0]


# ---------------------------------------------------------------- the two measures, by hand
def test_spearman_known_cases():
    a = np.arange(10.0)
    assert abs(spearman(a, a) - 1) < 1e-12
    assert abs(spearman(a, -a) + 1) < 1e-12                           # reversed
    assert abs(spearman(a, a ** 3) - 1) < 1e-12                       # any monotone transform
    assert math.isnan(spearman(a, np.ones(10)))                       # a constant map has no ranking
    # ties: ranks of [1,1,2,3] are [1.5,1.5,3,4]; against [1,2,3,4] -> Pearson of the ranks
    expected = np.corrcoef([1.5, 1.5, 3, 4], [1, 2, 3, 4])[0, 1]
    assert abs(spearman([1, 1, 2, 3], [1, 2, 3, 4]) - expected) < 1e-12


def test_topk_iou_known_cases_and_ties():
    v = np.arange(10.0)
    assert len(top_k(v, 0.2)[0]) == 2 and len(top_k(np.arange(11.0), 0.2)[0]) == 3      # k = ceil(f * n)
    assert topk_iou(v, v, 0.2)[0] == 1.0
    assert topk_iou(v, -v, 0.2)[0] == 0.0                              # disjoint top sets
    a = np.array([5, 4, 3, 2, 1, 0, 0, 0, 0, 0.0])
    b = np.array([5, 0, 3, 2, 1, 4, 0, 0, 0, 0.0])
    assert abs(topk_iou(a, b, 0.2)[0] - 1 / 3) < 1e-12                 # {0,1} vs {0,5}: 1 shared of 3
    assert topk_iou(np.ones(10), v, 0.2)[1] is True                    # threshold inside a tie group: flagged
    assert topk_iou(v, v, 0.2)[1] is False
    assert top_k(np.ones(10), 0.2)[0].tolist() == [0, 1]               # deterministic among ties


# ---------------------------------------------------------------- matrices
def three_techniques(n=20, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.rand(n, 28, 28)
    maps = {"a": base, "b": base * 2 + 0.01 * rng.rand(n, 28, 28), "c": rng.rand(n, 28, 28)}
    masks = np.zeros((n, 28, 28), dtype=bool)
    masks[:, 6:22, 6:22] = True
    return maps, masks


def test_agreement_matrices_have_the_expected_structure():
    maps, masks = three_techniques()
    r = agreement_matrices(maps, masks, CFG)
    assert r["names"] == ["a", "b", "c"] and r["n_images"] == 20
    for key in ("spearman_mean", "iou_mean"):
        assert np.allclose(r[key], r[key].T)                           # symmetric
        assert np.allclose(np.diag(r[key]), 1.0)                       # a map agrees with itself
    assert r["spearman_mean"][0, 1] > 0.99 and r["iou_mean"][0, 1] > 0.9      # b is a rescaled a
    assert abs(r["spearman_mean"][0, 2]) < 0.1                         # c is independent
    assert (r["n_valid_spearman"] == 20).all() and r["tie_rate"].shape == (3,)


def test_sign_mode_changes_the_comparison():
    maps, masks = three_techniques()
    maps["neg"] = -maps["a"]                                           # same magnitudes, opposite sign
    absolute = agreement_matrices(maps, masks, CFG)                    # SIGN_MODE absolute: |a| and |-a| are the same map
    i, j = absolute["names"].index("a"), absolute["names"].index("neg")
    assert absolute["spearman_mean"][i, j] > 0.999
    positive = agreement_matrices(maps, masks, dataclasses.replace(CFG, SIGN_MODE="positive"))
    assert positive["n_valid_spearman"][i, j] == 0                     # the positive part of -a is all zeros: no ranking


def test_random_reference_matches_theory():
    """Random against random: Spearman ~ 0 and IoU ~ the hypergeometric expectation (about 0.11)."""
    n, rng = 300, np.random.RandomState(7)
    maps = {"noise": rng.rand(n, 28, 28)}
    masks = np.zeros((n, 28, 28), dtype=bool)
    masks[:, 5:23, 5:23] = True                                        # 324 pixels
    ref = random_reference(maps, masks, CFG, seed=123)["noise"]
    expected = expected_random_iou(324, CFG.TOPK_FRACTION)
    assert abs(ref["spearman_mean"]) < 0.03
    assert abs(ref["iou_mean"] - expected) < 0.01 and 0.10 < expected < 0.12
    # the formula itself, checked by hand on a case small enough to enumerate: n=4, k=2
    assert abs(expected_random_iou(4, 0.5) - (4 / 6 * 1 / 3 + 1 / 6 * 1)) < 1e-12


def test_outside_digit_share():
    mask = np.zeros((28, 28), dtype=bool)
    mask[10:14, 10:14] = True
    inside, outside = np.zeros((28, 28)), np.zeros((28, 28))
    inside[11, 11], outside[2, 2] = 3.0, 3.0
    assert outside_digit_share(inside, mask) == 0.0 and outside_digit_share(outside, mask) == 1.0
    assert abs(outside_digit_share(inside + outside, mask) - 0.5) < 1e-12
    assert abs(outside_digit_share(-(inside + outside), mask) - 0.5) < 1e-12      # absolute value
    assert math.isnan(outside_digit_share(np.zeros((28, 28)), mask))
    table = outside_digit_table({"x": np.stack([inside, outside])}, np.stack([mask, mask]))
    assert abs(table["x"][0] - 0.5) < 1e-12
