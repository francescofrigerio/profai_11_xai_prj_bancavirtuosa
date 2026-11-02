"""baseline.py: the LR trains on the train indices only, is exact, and is not a fluke."""
import dataclasses
import warnings

import numpy as np
import pytest

from helpers import tiny_cfg, synthetic_mnist
from baseline import fit_baseline, lr_kwargs
from data import stratified_split, to_lr_input

CFG = tiny_cfg(LR_MAX_ITER=300)


def data():
    X, y = synthetic_mnist(n_per_class=30)
    return to_lr_input(X), y, stratified_split(y, 60, CFG.SEED)


def test_fit_uses_only_the_train_indices():
    """Regression of F8: the exploratory script trained on train+validation."""
    X, y, s = data()
    base = fit_baseline(X, y, s.idx_train, CFG)
    y_other = y.copy()
    y_other[s.idx_val] = (y_other[s.idx_val] + 3) % 10             # corrupt ONLY the validation labels
    other = fit_baseline(X, y_other, s.idx_train, CFG)
    assert np.array_equal(base.model.coef_, other.model.coef_)
    assert base.diagnostics["n_train"] == len(s.idx_train)


def test_model_is_exact_and_well_formed():
    X, y, s = data()
    base = fit_baseline(X, y, s.idx_train, CFG)
    facts = base.check_exactness(X[s.idx_val])
    assert facts["classes_ok"]                                       # coefficient rows aligned with the digits
    assert facts["decision_exact_max_err"] < 1e-8                    # logit == w.x + b
    assert facts["coef_colsum_max"] < 1e-6                           # l2: relative == absolute coefficients
    assert (base.predict(X[s.idx_val]) == y[s.idx_val]).mean() > 0.95
    p = base.predict_proba(X[s.idx_val])
    assert np.allclose(p.sum(1), 1.0) and p.shape == (60, 10)


def test_local_and_global_maps():
    X, y, s = data()
    base = fit_baseline(X, y, s.idx_train, CFG)
    i, c = int(s.idx_val[0]), 4
    m = base.local_map(X[i], c)
    assert m.shape == (28, 28)
    assert abs(m.sum() + base.model.intercept_[c] - base.decision_function(X[i:i + 1])[0, c]) < 1e-9
    assert (m[X[i].reshape(28, 28) == 0] == 0).all()                 # exactly zero on the background
    g = np.stack([base.global_map(k) for k in range(10)])
    assert np.allclose(g.sum(axis=0), 0, atol=1e-9)                  # relative maps sum to 0 over classes
    assert np.allclose(g[c], base.model.coef_[c].reshape(28, 28) - base.model.coef_.mean(0).reshape(28, 28))


def test_shuffled_labels_control():
    """If the pipeline leaked the labels, shuffled labels would still score high."""
    X, y, s = data()
    real = fit_baseline(X, y, s.idx_train, CFG)
    y_shuffled = y.copy()
    y_shuffled[s.idx_train] = np.random.RandomState(0).permutation(y[s.idx_train])
    fake = fit_baseline(X, y_shuffled, s.idx_train, CFG)
    acc = lambda m: float((m.predict(X[s.idx_val]) == y[s.idx_val]).mean())
    assert acc(real) > 0.95 and acc(fake) < 0.30                     # chance is 0.10


def test_config_is_mapped_without_deprecated_arguments():
    assert "penalty" not in lr_kwargs(CFG) and lr_kwargs(CFG)["C"] == CFG.LR_C
    l1 = lr_kwargs(dataclasses.replace(CFG, LR_PENALTY="l1", LR_SOLVER="saga"))
    assert ("penalty" in l1) != ("l1_ratio" in l1)                   # one or the other, by sklearn version
    X, y, s = data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_baseline(X, y, s.idx_train, CFG)
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]


def test_non_convergence_is_visible_not_swallowed():
    X, y, s = data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base = fit_baseline(X, y, s.idx_train, dataclasses.replace(CFG, LR_MAX_ITER=2))
    assert base.diagnostics["convergence_warning"]
    assert any(w.category.__name__ == "ConvergenceWarning" for w in caught)
