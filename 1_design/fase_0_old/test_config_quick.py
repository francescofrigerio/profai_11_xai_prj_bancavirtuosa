"""Smoke tests for config.py: defaults are valid, invalid values are rejected loudly."""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG, ConfigConstantsDefault  # noqa: E402


def test_defaults_are_valid_and_frozen():
    assert CONFIG.MIN_VAL_ACCURACY == 0.94
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.SEED = 1  # type: ignore[misc]


def test_surrogate_option_is_gone():
    names = {f.name for f in dataclasses.fields(ConfigConstantsDefault)}
    assert "USE_SURROGATE" not in names


def test_every_parameter_is_an_overridable_field():
    # Un-annotated class attributes would not be fields (and not overridable): guard against it.
    public = {k for k in vars(ConfigConstantsDefault)
              if k.isupper() and not k.startswith("_")}
    fields = {f.name for f in dataclasses.fields(ConfigConstantsDefault)}
    assert public == fields


def test_replace_overrides_and_revalidates():
    small = dataclasses.replace(CONFIG, IMAGE_SIZE=64, N_QUANT=50)
    assert small.IMAGE_SIZE == 64
    with pytest.raises(ValueError):
        dataclasses.replace(CONFIG, IMAGE_SIZE=100)   # not a multiple of LIME_GRID_CELLS=8


@pytest.mark.parametrize("override", [
    {"LR_PENALTY": "l1"},                   # l1 with lbfgs
    {"LR_PENALTY": "elasticnet", "LR_SOLVER": "saga"},
    {"MODEL_MODE": "maybe"},
    {"SIGN_MODE": "negative"},
    {"MASK_MODE": "half"},
    {"LIME_BACKEND": "other"},
    {"SHAP_BACKEND": "other"},
    {"OPTIMIZER": "sgd"},
    {"DEVICE": "tpu"},
    {"TOPK_FRACTION": 0.0},
    {"TOPK_FRACTION": 1.5},
    {"MIN_VAL_ACCURACY": 0.0},
    {"IMAGE_SIZE": 16},
    {"OCCLUSION_WINDOW": 256},
    {"N_QUANT": 105},                       # not a multiple of 10
    {"VAL_SIZE": 60000},
    {"BATCH_SIZE": 0},
])
def test_invalid_values_raise(override):
    with pytest.raises(ValueError):
        dataclasses.replace(CONFIG, **override)


def test_l1_with_saga_is_accepted():
    cfg = dataclasses.replace(CONFIG, LR_PENALTY="l1", LR_SOLVER="saga")
    assert cfg.LR_PENALTY == "l1"
