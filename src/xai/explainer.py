"""
    The common contract of the six explanation techniques.

    Every technique receives ONE preprocessed image and a target class and returns ONE 2D map at its
    native resolution, with the metadata needed to compare it with the others fairly:

    input   x_data  (1, 3, DENSENET_SIZE, DENSENET_SIZE) float32, ImageNet-normalized, on the model device
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
from checks_errors import CheckErrors, assert_densenet_input
from utility import sync

@dataclass
class AttributionResult:
    technique: str
    target: int
    map: np.ndarray
    signed: bool
    output_space: str
    seconds: float
    # meta: dict = field(default_factory=dict)

class Explainer:
    """
        Base class: validates the input, times the call and validates the output.
    """

    name: str = ""
    signed: bool = True
    output_space: str = "logit"

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def explain(self, x_data, target: int) -> AttributionResult:
        assert_densenet_input(x_data, self.cfg)
        if x_data.shape[0] != 1:
            raise CheckErrors(f"{self.name}: explains ONE image at a time, got a batch of {x_data.shape[0]}")
        if not (0 <= int(target) < self.cfg.NUM_CLASSES):
            raise CheckErrors(f"{self.name}: target {target} is not a class of the model")
        if self.model.training:
            raise CheckErrors(f"{self.name}: the model must be in eval mode (BatchNorm would corrupt the explanation)")
        device = x_data.device
        sync(device)
        t0 = time.perf_counter()
        
        sync(device)
        seconds = time.perf_counter() - t0
        map = x_data
        if not np.isfinite(map).all():
            raise CheckErrors(f"{self.name}: the map contains NaN or inf")
        if not self.signed and float(map.min()) < 0:
            raise CheckErrors(f"{self.name}: declared unsigned but has negative values")
        return AttributionResult(self.name, int(target), map, self.signed, self.output_space, seconds)


