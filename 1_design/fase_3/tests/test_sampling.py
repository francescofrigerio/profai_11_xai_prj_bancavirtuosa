"""sampling.py: who gets explained. Stratified, reproducible, no repeats, nothing shrinks silently."""
import numpy as np
import pytest

from sampling import explanation_targets, select_gallery, select_quant_sample


def fake_predictions(n=3000, seed=0):
    """True labels and two models' predictions with a few errors each, so every quadrant is populated."""
    rng = np.random.RandomState(seed)
    y = np.arange(n) % 10
    pd, pl = y.copy(), y.copy()
    wrong_d = rng.rand(n) < 0.04
    wrong_l = rng.rand(n) < 0.12
    pd[wrong_d] = (y[wrong_d] + 1 + rng.randint(0, 9, wrong_d.sum())) % 10
    pl[wrong_l] = (y[wrong_l] + 1 + rng.randint(0, 9, wrong_l.sum())) % 10
    conf = rng.uniform(0.5, 1.0, n)
    return y, pd, pl, conf


def test_quant_sample_is_stratified_by_the_predicted_class_and_reproducible():
    _, pd, _, _ = fake_predictions()
    s = select_quant_sample(pd, 100, 10, seed=42)
    assert len(s) == 100 and len(set(s.tolist())) == 100
    assert (np.bincount(pd[s], minlength=10) == 10).all()            # 10 per PREDICTED class
    assert np.array_equal(s, select_quant_sample(pd, 100, 10, seed=42))
    assert not np.array_equal(s, select_quant_sample(pd, 100, 10, seed=43))
    with pytest.raises(ValueError):
        select_quant_sample(pd, 105, 10, seed=42)                    # not a multiple of the classes
    with pytest.raises(ValueError):
        select_quant_sample(np.where(pd == 3, 4, pd), 100, 10, seed=42)   # class 3 never predicted


def test_gallery_has_the_agreed_composition_and_never_repeats():
    y, pd, pl, conf = fake_predictions()
    g = select_gallery(y, pd, pl, conf, num_classes=10, seed=42, gallery_size=24)
    roles = [e["role"] for e in g]
    assert len(g) == 24 and len({e["index"] for e in g}) == 24       # 24 different images
    assert roles.count("correct") == 10 and roles.count("confident_error") == 6
    assert sum(r.startswith("quadrant:") for r in roles) == 8
    correct = [e for e in g if e["role"] == "correct"]
    assert sorted(e["true"] for e in correct) == list(range(10)) and all(e["pred"] == e["true"] for e in correct)
    errors = [e for e in g if e["role"] == "confident_error"]
    all_errors = np.where(pd != y)[0]
    top6 = all_errors[np.lexsort((all_errors, -conf[all_errors]))][:6]
    assert [e["index"] for e in errors] == top6.tolist()             # the 6 most confident, in order
    for name in ("both_correct", "only_densenet", "only_lr", "both_wrong"):
        assert roles.count(f"quadrant:{name}") == 2
    assert all(e["targets"] == [e["pred"]] for e in g if e["pred"] == e["true"])
    assert all(e["targets"] == [e["pred"], e["true"]] for e in g if e["pred"] != e["true"])   # both classes for errors
    again = select_gallery(y, pd, pl, conf, 10, 42, 24)
    assert [e["index"] for e in again] == [e["index"] for e in g]
    other = select_gallery(y, pd, pl, conf, 10, 43, 24)
    assert [e["index"] for e in other] != [e["index"] for e in g]


def test_gallery_refuses_instead_of_shrinking():
    y, pd, pl, conf = fake_predictions()
    with pytest.raises(ValueError):
        select_gallery(y, pd, pl, conf, 10, 42, gallery_size=20)     # composition says 24
    few_errors = np.where(np.arange(len(y)) < 3, (y + 1) % 10, y)    # only 3 DenseNet errors
    with pytest.raises(ValueError):
        select_gallery(y, few_errors, pl, conf, 10, 42, 24)
    assert explanation_targets(5, 5) == [5] and explanation_targets(5, 3) == [3, 5]
