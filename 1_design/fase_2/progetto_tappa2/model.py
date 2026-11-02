"""DenseNet121 for MNIST: ImageNet backbone (frozen by default) + a new Linear head.

Two flags decide the behaviour (nothing switches by itself):
  - FLAG_BATCHNORM_EVAL: the BatchNorm of the backbone stay in eval mode during training (only
    meaningful when FREEZE_BACKBONE=True). Phase 0: validation accuracy 0.9700 with it, 0.9493
    without (the running statistics drift away from the ImageNet ones).
  - FLAG_RELU_INPLACE: True keeps torchvision's in-place ReLU (less VRAM), False replaces them
    with a patched model. Phase 0 (check 2): Grad-CAM, IG, Saliency and SHAP give the same maps
    in both cases.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------------------------
def build_densenet(cfg, pretrained: bool) -> nn.Module:
    """torchvision DenseNet121 with a new Linear head (NUM_CLASSES outputs).
    pretrained=False is used when a checkpoint is loaded: no ImageNet download is needed."""
    from torchvision.models import DenseNet121_Weights, densenet121
    weights = None
    if pretrained:
        if not hasattr(DenseNet121_Weights, cfg.BACKBONE_WEIGHTS):
            raise ValueError(f"unknown DenseNet121 weights {cfg.BACKBONE_WEIGHTS!r}")
        weights = getattr(DenseNet121_Weights, cfg.BACKBONE_WEIGHTS)
    base = densenet121(weights=weights)
    base.classifier = nn.Linear(base.classifier.in_features, cfg.NUM_CLASSES)
    return base


class SafeDenseNet(nn.Module):
    """DenseNet121 with no in-place ReLU (module flags AND the functional one of forward).
    Same parameter names as torchvision's model: state dicts are interchangeable."""

    def __init__(self, base: nn.Module):
        super().__init__()
        for m in base.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.features = base.features
        self.classifier = base.classifier

    def forward(self, x):
        out = F.relu(self.features(x), inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return self.classifier(torch.flatten(out, 1))


def wrap_for_flag(base: nn.Module, cfg) -> nn.Module:
    """FLAG_RELU_INPLACE=True keeps torchvision's model, False patches the ReLU."""
    return base if cfg.FLAG_RELU_INPLACE else SafeDenseNet(base)


def count_inplace_relu(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.ReLU) and m.inplace)


def validate_layers(model: nn.Module, cfg) -> None:
    """The Grad-CAM layers named in the config must exist in the model."""
    names = dict(model.named_modules())
    for param in ("GRADCAM_LAYER", "GRADCAM_ALT_LAYER"):
        if getattr(cfg, param) not in names:
            raise ValueError(f"{param}={getattr(cfg, param)!r} is not a module of the model")


# ----------------------------------------------------------------------------------------
# Training mode
# ----------------------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad_(False)


def set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def enter_train_mode(model: nn.Module, cfg) -> None:
    """Single entry point for training mode, called at the start of EVERY epoch (a validation
    pass in between would otherwise put the BatchNorm back in train mode)."""
    model.train()
    if cfg.FREEZE_BACKBONE and cfg.FLAG_BATCHNORM_EVAL:
        model.features.apply(set_batchnorm_eval)


def trainable_parameters(model: nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(model: nn.Module, cfg):
    """Head only (frozen backbone) or the whole network, each with its own learning rate."""
    if cfg.FREEZE_BACKBONE:
        return torch.optim.Adam(trainable_parameters(model), lr=cfg.LEARNING_RATE_HEAD)
    return torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE_FULL)


def bn_buffers(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.features.named_buffers()
            if k.endswith("running_mean") or k.endswith("running_var")}


def bn_drift(before: dict, after: dict) -> float:
    return max((after[k] - before[k]).abs().max().item() for k in before) if before else 0.0


# ----------------------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------------------
def predict_logits(model: nn.Module, x28, cfg, device, batch_size: int = 256) -> np.ndarray:
    """(N, 1, 28, 28) CPU tensor in [0, 1] -> (N, NUM_CLASSES) logits (numpy), in eval mode."""
    from checks import assert_densenet_input, assert_logits
    from data import to_densenet_input
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x28), batch_size):
            batch = to_densenet_input(x28[i:i + batch_size].to(device), cfg.DENSENET_SIZE)
            if i == 0:
                assert_densenet_input(batch, cfg)
            logits = model(batch)
            if i == 0:
                assert_logits(logits, cfg)
            out.append(logits.cpu().numpy())
    return np.concatenate(out)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
