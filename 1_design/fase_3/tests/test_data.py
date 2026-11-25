"""data.py: the split is stratified, disjoint and reproducible; the two views describe the same image."""
import numpy as np
import pytest

from helpers import tiny_cfg, synthetic_mnist
from data import debug_indices, stratified_split, to_lr_input


def test_split_is_stratified_disjoint_and_complete():
    _, y = synthetic_mnist(n_per_class=30)
    s = stratified_split(y, val_size=60, seed=42)
    assert len(s.idx_val) == 60 and len(s.idx_train) == len(y) - 60
    assert set(s.idx_train).isdisjoint(set(s.idx_val))
    assert set(s.idx_train) | set(s.idx_val) == set(range(len(y)))
    assert (np.bincount(y[s.idx_val], minlength=10) == 6).all()          # 6 per class


def test_split_is_deterministic_and_depends_on_the_seed():
    _, y = synthetic_mnist(n_per_class=30)
    a, b, c = (stratified_split(y, 60, seed) for seed in (42, 42, 43))
    assert np.array_equal(a.idx_val, b.idx_val)
    assert not np.array_equal(a.idx_val, c.idx_val)


def test_debug_indices_one_validation_image_per_class():
    _, y = synthetic_mnist(n_per_class=30)
    s = stratified_split(y, 60, 42)
    picks = debug_indices(y, s.idx_val, 10)
    assert list(y[picks]) == list(range(10)) and set(picks) <= set(s.idx_val)
    with pytest.raises(ValueError):
        debug_indices(y, [i for i in s.idx_val if y[i] != 3], 10)         # class 3 absent


def test_lr_view_is_not_standardized():
    X, _ = synthetic_mnist()
    flat = to_lr_input(X)
    assert flat.shape == (len(X), 784) and flat.dtype == np.float64
    assert np.array_equal(flat, X.reshape(len(X), -1).astype(np.float64))   # values untouched
    assert flat.min() == 0.0 and 0.0 < flat.max() <= 1.0
    assert (flat[:, :28] == 0).all()                                        # background is exactly 0


def test_densenet_view_normalization_and_black_baseline():
    torch = pytest.importorskip("torch")
    from checks import assert_densenet_input
    from data import IMAGENET_MEAN, IMAGENET_STD, black_normalized, denormalize, to_densenet_input
    cfg = tiny_cfg()
    s = cfg.DENSENET_SIZE
    black = to_densenet_input(torch.zeros(2, 1, 28, 28), s)
    white = to_densenet_input(torch.ones(2, 1, 28, 28), s)
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    # constant images survive any resize: independent, hand-computed expectation
    assert np.allclose(black[0, :, 3, 3].numpy(), -mean / std, atol=1e-6)
    assert np.allclose(white[0, :, 3, 3].numpy(), (1 - mean) / std, atol=1e-5)
    assert_densenet_input(black, cfg)                                       # the guards accept both
    assert_densenet_input(white, cfg)
    assert black.shape == (2, 3, s, s) and black.dtype == torch.float32
    # the baseline is the black of the ORIGINAL image, bit by bit equal to a black image's pixels
    base = black_normalized(s, "cpu")
    assert torch.equal(base[0], black[0])
    assert float(denormalize(base).abs().max()) < 1e-6
    assert np.allclose(base[0, :, 0, 0].numpy(), [-2.118, -2.036, -1.804], atol=1e-3)


def test_both_views_describe_the_same_image():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from data import denormalize, to_densenet_input
    cfg = tiny_cfg()
    X, _ = synthetic_mnist()
    x = torch.from_numpy(X[:10])
    back = F.interpolate(denormalize(to_densenet_input(x, cfg.DENSENET_SIZE))[:, 0:1],
                         size=(28, 28), mode="area")
    for i in range(10):
        corr = np.corrcoef(back[i, 0].numpy().ravel(), to_lr_input(x)[i])[0, 1]
        assert corr > 0.95                          # the DenseNet view, brought back, IS the LR view
        assert np.unravel_index(back[i, 0].argmax(), (28, 28)) in [
            (r, c) for r in range(28) for c in range(28) if X[i, 0, r, c] > 0]   # brightest pixel on the block
