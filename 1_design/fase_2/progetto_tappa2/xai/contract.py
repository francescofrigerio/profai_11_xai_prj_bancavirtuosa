"""The common contract of the six explanation techniques.

Every technique receives ONE preprocessed image and a target class and returns ONE 2D map at its
native resolution, with the metadata needed to compare it with the others fairly:

  input   x1      (1, 3, DENSENET_SIZE, DENSENET_SIZE) float32, ImageNet-normalized, on the model device
  input   target  int, the class that is being explained
  output  map     float64 (h, w), finite; >= 0 when `signed` is False
          signed  whether the sign of the map means something (Grad-CAM and Saliency are unsigned)
          output_space  "logit" or "probability" (LIME explains the probability)
          seconds       wall time of the explanation (GPU synchronized)

The resampling of the native maps to the 28x28 grid and the sign mode belong to the comparison
module (stage 3), not to the techniques.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from checks import ContractError, assert_densenet_input


@dataclass
class AttributionResult:
    technique: str
    target: int
    map: np.ndarray
    signed: bool
    output_space: str
    seconds: float
    meta: dict = field(default_factory=dict)


class Explainer:
    """Base class: validates the input, times the call and validates the output."""

    name: str = ""
    signed: bool = True
    output_space: str = "logit"

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def _attribute(self, x1, target: int) -> np.ndarray:           # implemented by each technique
        raise NotImplementedError

    def _meta(self, x1, target: int) -> dict:                      # optional diagnostics
        return {}

    def explain(self, x1, target: int) -> AttributionResult:
        assert_densenet_input(x1, self.cfg)
        if x1.shape[0] != 1:
            raise ContractError(f"{self.name}: explains ONE image at a time, got a batch of {x1.shape[0]}")
        if not (0 <= int(target) < self.cfg.NUM_CLASSES):
            raise ContractError(f"{self.name}: target {target} is not a class of the model")
        if self.model.training:
            raise ContractError(f"{self.name}: the model must be in eval mode (BatchNorm would corrupt the explanation)")
        device = x1.device
        _sync(device)
        t0 = time.perf_counter()
        raw = self._attribute(x1, int(target))
        _sync(device)
        seconds = time.perf_counter() - t0
        m = np.asarray(raw, dtype=np.float64)
        if m.ndim != 2:
            raise ContractError(f"{self.name}: the map must be 2D, got shape {m.shape}")
        if not np.isfinite(m).all():
            raise ContractError(f"{self.name}: the map contains NaN or inf")
        if not self.signed and float(m.min()) < 0:
            raise ContractError(f"{self.name}: declared unsigned but has negative values")
        return AttributionResult(self.name, int(target), m, self.signed, self.output_space, seconds,
                                 self._meta(x1, int(target)))


def _sync(device) -> None:
    if getattr(device, "type", "") == "cuda":
        import torch
        torch.cuda.synchronize()
