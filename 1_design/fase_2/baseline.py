"""Logistic regression baseline: an interpretable system that explains itself exactly.

  - trained on the TRAIN indices only (the validation set stays untouched: bug F8 of the
    exploratory script trained it on train+validation);
  - every hyper-parameter comes from the config;
  - local map for class a on image x:  w_a * x  (exact contribution of each pixel to the logit,
    the intercept b_a is not a pixel and is not part of the map);
  - global map for class c:            w_c - mean over classes of w  (relative coefficients).
    With the l2 penalty the coefficients already sum to ~0 over the classes (phase 0: 1e-13), so
    the relative and the absolute versions coincide; the sum is exposed as a diagnostic.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np


def lr_kwargs(cfg) -> dict:
    """Config -> LogisticRegression kwargs. The l2 penalty is the default of every sklearn
    version, so it is never passed (sklearn >= 1.8 deprecates the `penalty` argument)."""
    import sklearn
    kw = dict(C=cfg.LR_C, solver=cfg.LR_SOLVER, max_iter=cfg.LR_MAX_ITER, random_state=cfg.SEED)
    if cfg.LR_PENALTY != "l2":
        major_minor = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
        if major_minor >= (1, 8):
            kw["l1_ratio"] = 1.0
        else:
            kw["penalty"] = cfg.LR_PENALTY
    return kw


@dataclass
class LRBaseline:
    model: object
    cfg: object
    diagnostics: dict

    # --- predictions
    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def decision_function(self, X) -> np.ndarray:
        return self.model.decision_function(X)

    # --- maps
    def local_map(self, x_flat, class_idx: int) -> np.ndarray:
        """Contribution of each pixel to the logit of `class_idx`: (IMAGE_SIZE, IMAGE_SIZE)."""
        x = np.asarray(x_flat, dtype=np.float64).reshape(-1)
        side = self.cfg.IMAGE_SIZE
        return (self.model.coef_[class_idx] * x).reshape(side, side)

    def global_map(self, class_idx: int) -> np.ndarray:
        """Relative coefficients of `class_idx`: (IMAGE_SIZE, IMAGE_SIZE)."""
        coef = self.model.coef_
        side = self.cfg.IMAGE_SIZE
        return (coef[class_idx] - coef.mean(axis=0)).reshape(side, side)

    def check_exactness(self, X_sample) -> dict:
        """Falsifiable facts about the model, measured on a sample of inputs."""
        manual = X_sample @ self.model.coef_.T + self.model.intercept_
        return {
            "decision_exact_max_err": float(np.abs(manual - self.decision_function(X_sample)).max()),
            "coef_colsum_max": float(np.abs(self.model.coef_.sum(axis=0)).max()),
            "classes_ok": bool(np.array_equal(self.model.classes_, np.arange(self.cfg.NUM_CLASSES))),
        }


def fit_baseline(X, y, idx_train, cfg) -> LRBaseline:
    """Fit on X[idx_train] only. A ConvergenceWarning is re-emitted, never swallowed."""
    from sklearn.linear_model import LogisticRegression
    X, y, idx_train = np.asarray(X), np.asarray(y), np.asarray(idx_train)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model = LogisticRegression(**lr_kwargs(cfg)).fit(X[idx_train], y[idx_train])
        fit_seconds = time.time() - t0
    for w in caught:
        if w.category.__name__ == "ConvergenceWarning":
            warnings.warn(w.message, w.category, stacklevel=2)
    diagnostics = {
        "fit_seconds": fit_seconds,
        "n_iter": int(np.max(model.n_iter_)),
        "convergence_warning": any(w.category.__name__ == "ConvergenceWarning" for w in caught),
        "other_warnings": sorted({f"{w.category.__name__}: {str(w.message)[:140]}"
                                  for w in caught if w.category.__name__ != "ConvergenceWarning"}),
        "n_train": int(len(idx_train)),
    }
    return LRBaseline(model=model, cfg=cfg, diagnostics=diagnostics)
