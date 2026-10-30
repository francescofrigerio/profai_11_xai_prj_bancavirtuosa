"""
   Errors checks on the data entering and leaving the two models.

  kinds of checks (design principle: falsifiability, but nothing stops silently):
  - assert_*  : the CONTRACT between modules. A violation is a bug -> ContractError.
  - expect : an EXPECTATION of the design (accuracy, gap between models...). A verification Table
    failure is a WARN line in the report, it never stops the run.

No torch needed: tensors are converted to numpy.
"""
from __future__ import annotations

import numpy as np

from data import denormalize_np


class CheckErrors(AssertionError):
    """
       A module received (or produced) data that violates the contract.
    """


def detach_cpu_numpy(x_data) -> np.ndarray:
    return x_data.detach().cpu().numpy() if hasattr(x_data, "detach") else np.asarray(x_data)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckErrors(message)

# Inputs
def assert_lr_input(x_data, cfg) -> None:
    """
        (B, IMAGE_SIZE**2) in [0, 1], finite, not standardized.
    """
    a = detach_cpu_numpy(x_data)
    _require(a.ndim == 2 and a.shape[1] == cfg.IMAGE_SIZE ** 2,
             f"LR input must be (B, {cfg.IMAGE_SIZE ** 2}), got {a.shape}")
    _require(np.issubdtype(a.dtype, np.floating), f"LR input must be floating, got {a.dtype}")
    _require(bool(np.isfinite(a).all()), "LR input contains NaN or inf")
    _require(float(a.min()) >= 0.0 and float(a.max()) <= 1.0,
             f"LR input must lie in [0, 1] (no standardization), got [{a.min():.3f}, {a.max():.3f}]")


def assert_densenet_input(x_data, cfg, tol: float = 1e-4) -> None:
    """(B, 3, S, S) float32; once denormalized: values in [0, 1] and identical channels.
    It catches a missing/duplicated normalization and the wrong order of the operations."""
    a = detach_cpu_numpy(x_data)
    s = cfg.DENSENET_SIZE
    _require(a.ndim == 4 and a.shape[1:] == (3, s, s),
             f"DenseNet input must be (B, 3, {s}, {s}), got {a.shape}")
    _require(a.dtype == np.float32, f"DenseNet input must be float32, got {a.dtype}")
    _require(bool(np.isfinite(a).all()), "DenseNet input contains NaN or inf")
    pixels = denormalize_np(a)
    _require(float(pixels.min()) >= -tol and float(pixels.max()) <= 1.0 + tol,
             "DenseNet input is not an ImageNet-normalized image in [0, 1] "
             f"(denormalized range [{pixels.min():.3f}, {pixels.max():.3f}])")
    spread = max(float(np.abs(pixels[:, 0] - pixels[:, 1]).max()),
                 float(np.abs(pixels[:, 0] - pixels[:, 2]).max()))
    _require(spread <= tol, f"the three channels differ after denormalization (max {spread:.2e}): "
                            "the grey channel must be replicated BEFORE normalizing")

# Outputs
def assert_logits(logits, cfg) -> None:
    a = detach_cpu_numpy(logits)
    _require(a.ndim == 2 and a.shape[1] == cfg.NUM_CLASSES,
             f"logits must be (B, {cfg.NUM_CLASSES}), got {a.shape}")
    _require(bool(np.isfinite(a).all()), "logits contain NaN or inf")


def assert_proba(p, cfg, atol: float = 1e-4) -> None:
    a = detach_cpu_numpy(p)
    _require(a.ndim == 2 and a.shape[1] == cfg.NUM_CLASSES,
             f"probabilities must be (B, {cfg.NUM_CLASSES}), got {a.shape}")
    _require(bool(np.isfinite(a).all()), "probabilities contain NaN or inf")
    _require(float(a.min()) >= -atol and float(a.max()) <= 1.0 + atol,
             f"probabilities outside [0, 1]: [{a.min():.4f}, {a.max():.4f}]")
    _require(bool(np.allclose(a.sum(axis=1), 1.0, atol=atol)),
             f"probabilities do not sum to 1 (max deviation {np.abs(a.sum(axis=1) - 1).max():.2e})")


# The 'Verification' table
class Verifications:
    """Rows (name, status, detail) of the table shown in the report appendix."""

    ORDER = {"PASS": 0, "INFO": 0, "WARN": 1, "FAIL": 2}

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, name: str, status: str, detail: str) -> None:
        if status not in self.ORDER:
            raise ValueError(f"status must be one of {tuple(self.ORDER)}, got {status!r}")
        self.rows.append({"check": name, "status": status, "detail": detail})

    def expect(self, name: str, ok: bool, detail: str) -> bool:
        """
           An expectation of the design: PASS if met, WARN (never a stop) otherwise.
        """
        self.record(name, "PASS" if ok else "WARN", detail)
        return bool(ok)

    def worst(self) -> str:
        return max((r["status"] for r in self.rows), key=lambda s: self.ORDER[s], default="PASS")

    def to_markdown(self) -> str:
        lines = ["| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {r['check']} | {r['status']} | {r['detail']} |" for r in self.rows]
        return "\n".join(lines)