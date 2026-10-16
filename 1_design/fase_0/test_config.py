"""Smoke tests for config.py: defaults are valid, invalid values are rejected loudly."""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG, ConfigConstantsDefault  # noqa: E402


def test_defaults_are_valid_and_frozen():
    """
         Min Accuracy Test Value
    """
    assert CONFIG.MIN_VAL_ACCURACY == 0.95
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.SEED = 1  # type: ignore[misc]


def test_surrogate_option_is_gone():
    names = {f.name for f in dataclasses.fields(ConfigConstantsDefault)}
    assert "USE_SURROGATE" not in names


def test_early_stopping_is_gone_and_flags_exist():
    names = {f.name for f in dataclasses.fields(ConfigConstantsDefault)}
    assert not any("EARLY" in n for n in names)
    assert CONFIG.FLAG_BATCHNORM_EVAL is True and CONFIG.FLAG_RELU_INPLACE is True


def test_mnist_side_and_densenet_side_are_separate():
    """
         Mnist Size and DenseNet Size Test Value
    """
    assert (CONFIG.IMAGE_SIZE, CONFIG.DENSENET_SIZE) == (28, 128)


def test_every_parameter_is_an_overridable_field():
    # Un-annotated class attributes would not be fields (and not overridable): guard against it.
    public = {k for k in vars(ConfigConstantsDefault)
              if k.isupper() and not k.startswith("_")}
    fields = {f.name for f in dataclasses.fields(ConfigConstantsDefault)}
    assert public == fields


def test_replace_overrides_and_revalidates():
    small = dataclasses.replace(CONFIG, DENSENET_SIZE=64, N_QUANT=50)
    assert small.DENSENET_SIZE == 64
    with pytest.raises(ValueError):
        dataclasses.replace(CONFIG, DENSENET_SIZE=100)   # not a multiple of LIME_GRID_CELLS=8


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
    {"DENSENET_SIZE": 16},
    {"IMAGE_SIZE": 32},                     # MNIST side is fixed
    {"FLAG_RELU_INPLACE": "yes"},
    {"FLAG_BATCHNORM_EVAL": 1},
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
