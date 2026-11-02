"""Small plain helpers shared by the tests (no pytest fixtures: easy to read and to reuse)."""
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402


def tiny_cfg(**overrides):
    """Config for fast CPU tests: 64-pixel DenseNet input, tiny split, CPU, no downloads."""
    base = dict(DENSENET_SIZE=64, BATCH_SIZE=16, NUM_EPOCHS=2, VAL_SIZE=30, N_QUANT=50,
                DEVICE="cpu")
    base.update(overrides)
    return dataclasses.replace(CONFIG, **base)


def synthetic_mnist(n_per_class=12, seed=0):
    """(N, 1, 28, 28) float32 in [0, 1] and labels (N,). Class c is a bright 5x5 block at a
    class-dependent position, with a random brightness. The background is exactly 0."""
    rng = np.random.RandomState(seed)
    y = np.repeat(np.arange(10), n_per_class)
    X = np.zeros((len(y), 1, 28, 28), dtype=np.float32)
    for i, c in enumerate(y):
        r, col = 3 + 12 * (c // 5), 2 + 5 * (c % 5)
        X[i, 0, r:r + 5, col:col + 5] = rng.uniform(0.6, 1.0)
    order = rng.permutation(len(y))
    return X[order], y[order]


def synthetic_data_bundle(cfg, n_per_class=12, seed=0):
    """The same structure as data.MnistData, built from the synthetic images."""
    import torch
    from data import stratified_split
    X, y = synthetic_mnist(n_per_class, seed)
    return SimpleNamespace(
        X_trainval=torch.from_numpy(X), y_trainval=torch.from_numpy(y).long(),
        X_test=torch.from_numpy(X[:20]), y_test=torch.from_numpy(y[:20]).long(),
        split=stratified_split(y, cfg.VAL_SIZE, cfg.SEED))
