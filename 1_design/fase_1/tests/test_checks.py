"""checks.py: the guards accept valid data and, as importantly, REFUSE corrupted data
(negative controls: a check that cannot fail proves nothing)."""
import numpy as np
import pytest

from helpers import tiny_cfg
from checks import (ContractError, Verifications, assert_densenet_input, assert_logits,
                    assert_lr_input, assert_proba)
from data import IMAGENET_MEAN, IMAGENET_STD

CFG = tiny_cfg()
MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def good_densenet_input(batch=2):
    grey = np.random.RandomState(0).uniform(0, 1, (batch, 1, 64, 64)).astype(np.float32)
    return ((np.repeat(grey, 3, axis=1) - MEAN) / STD).astype(np.float32)   # replicate, THEN normalize


def test_valid_inputs_pass():
    assert_lr_input(np.random.RandomState(0).uniform(0, 1, (5, 784)), CFG)
    assert_densenet_input(good_densenet_input(), CFG)


@pytest.mark.parametrize("name, bad", [
    ("standardized LR input (negative values)", lambda: np.random.RandomState(0).randn(5, 784)),
    ("LR input in 0..255", lambda: np.random.RandomState(0).uniform(0, 255, (5, 784))),
    ("LR input with the wrong width", lambda: np.zeros((5, 700))),
    ("LR input with NaN", lambda: np.full((5, 784), np.nan)),
])
def test_lr_input_negative_controls(name, bad):
    with pytest.raises(ContractError):
        assert_lr_input(bad(), CFG)


@pytest.mark.parametrize("name, bad", [
    ("not normalized (values in 0..1, channels replicated)",
     lambda: np.repeat(np.random.RandomState(0).uniform(0, 1, (2, 1, 64, 64)), 3, axis=1).astype(np.float32)),
    ("normalized before replicating the channel",
     lambda: np.repeat((np.random.RandomState(0).uniform(0, 1, (2, 1, 64, 64)).astype(np.float32) - 0.485) / 0.229, 3, axis=1)),
    ("pixels in 0..255", lambda: (good_densenet_input() * 0 + 200).astype(np.float32)),
    ("float64", lambda: good_densenet_input().astype(np.float64)),
    ("wrong size", lambda: good_densenet_input()[:, :, :32, :32]),
])
def test_densenet_input_negative_controls(name, bad):
    with pytest.raises(ContractError):
        assert_densenet_input(bad(), CFG)


def test_output_guards():
    logits = np.random.RandomState(0).randn(4, 10)
    assert_logits(logits, CFG)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert_proba(p, CFG)
    for bad in (np.full((4, 10), np.nan), np.zeros((4, 9))):
        with pytest.raises(ContractError):
            assert_logits(bad, CFG)
    with pytest.raises(ContractError):
        assert_proba(p * 1.1, CFG)                  # rows do not sum to 1
    with pytest.raises(ContractError):
        assert_proba(-p, CFG)                       # negative probabilities


def test_verifications_table_never_stops_on_an_expectation():
    v = Verifications()
    assert v.expect("accuracy above the threshold", True, "0.97 >= 0.95") is True
    assert v.expect("LR below DenseNet", False, "gap -0.01") is False      # WARN, no exception
    assert [r["status"] for r in v.rows] == ["PASS", "WARN"] and v.worst() == "WARN"
    assert "| accuracy above the threshold | PASS |" in v.to_markdown()
    with pytest.raises(ValueError):
        v.record("x", "MAYBE", "invalid status")
