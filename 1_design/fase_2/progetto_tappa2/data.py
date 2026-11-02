"""Data: MNIST loading, the stratified split and the two views of the SAME image.

  - LR view:       (N, 784) float64 in [0, 1], NOT standardized (the local map w*x stays zero
                   on the background).
  - DenseNet view: (N, 3, DENSENET_SIZE, DENSENET_SIZE) float32, normalized with the ImageNet
                   statistics. Built on tensors (resize -> replicate channel -> normalize), so
                   it can run on the GPU (phase 0, check 6a: max difference from the torchvision
                   PIL pipeline 0.017 in normalized units).

torch is imported lazily: the split and the LR view work (and are tested) without it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    idx_train: np.ndarray
    idx_val: np.ndarray


def stratified_split(labels, val_size: int, seed: int) -> Split:
    """Stratified train/validation split, by index. One split for BOTH models."""
    from sklearn.model_selection import train_test_split
    labels = np.asarray(labels)
    idx_train, idx_val = train_test_split(
        np.arange(len(labels)), test_size=val_size, stratify=labels, random_state=seed)
    return Split(idx_train=idx_train, idx_val=idx_val)


def debug_indices(labels, idx_val, num_classes: int) -> np.ndarray:
    """First validation image of each class, in a fixed order: the images of the debug Grad-CAM."""
    labels = np.asarray(labels)
    picks = []
    for c in range(num_classes):
        hits = [int(i) for i in idx_val if labels[i] == c]
        if not hits:
            raise ValueError(f"no validation image of class {c}")
        picks.append(hits[0])
    return np.asarray(picks)


@dataclass
class MnistData:
    X_trainval: object      # torch.Tensor (60000, 1, 28, 28) float32 in [0, 1]
    y_trainval: object      # torch.Tensor (60000,) int64
    X_test: object
    y_test: object
    split: Split


def load_mnist(cfg) -> MnistData:
    from torchvision import datasets
    train = datasets.MNIST(root=cfg.DATA_ROOT, train=True, download=True)
    test = datasets.MNIST(root=cfg.DATA_ROOT, train=False, download=True)
    y_tv = train.targets.clone()
    return MnistData(
        X_trainval=train.data.unsqueeze(1).float() / 255.0, y_trainval=y_tv,
        X_test=test.data.unsqueeze(1).float() / 255.0, y_test=test.targets.clone(),
        split=stratified_split(y_tv.numpy(), cfg.VAL_SIZE, cfg.SEED))


# ----------------------------------------------------------------------------------------
# LR view
# ----------------------------------------------------------------------------------------
def to_lr_input(x28) -> np.ndarray:
    """(N, 1, 28, 28) tensor or array in [0, 1] -> (N, 784) float64, unchanged otherwise."""
    arr = x28.detach().cpu().numpy() if hasattr(x28, "detach") else np.asarray(x28)
    return arr.reshape(len(arr), -1).astype(np.float64)


# ----------------------------------------------------------------------------------------
# DenseNet view
# ----------------------------------------------------------------------------------------
def _stats(like):
    import torch
    mean = torch.tensor(IMAGENET_MEAN, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    return mean, std


def normalize(x):
    mean, std = _stats(x)
    return (x - mean) / std


def denormalize(x):
    mean, std = _stats(x)
    return x * std + mean


def denormalize_np(x: np.ndarray) -> np.ndarray:
    """numpy version of denormalize, for the contract checks (no torch needed)."""
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float64).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float64).reshape(1, 3, 1, 1)
    return np.asarray(x, dtype=np.float64) * std + mean


def to_densenet_input(x28, size: int):
    """(N, 1, 28, 28) in [0, 1] -> (N, 3, size, size) normalized.
    The ORDER matters: resize, replicate the channel, then normalize."""
    import torch.nn.functional as F
    x = F.interpolate(x28, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return normalize(x)


def black_normalized(size: int, device):
    """The black of the ORIGINAL image expressed in the normalized space (baseline of IG, SHAP,
    Occlusion and fill colour of LIME). It is NOT zero: about (-2.12, -2.04, -1.80)."""
    import torch
    return normalize(torch.zeros(1, 3, size, size, device=device))
