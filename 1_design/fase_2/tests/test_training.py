"""training.py: modes, checkpoint compatibility, reproducibility, debug hook."""
import hashlib
import tempfile
from pathlib import Path

import pytest

from helpers import tiny_cfg, synthetic_data_bundle
from training import (SNAPSHOT_FIELDS, debug_epochs, file_sha256, resolve_mode,
                      snapshot_from_config, snapshot_mismatches)


# ---------------------------------------------------------------- pure logic (no torch needed)
def test_snapshot_detects_every_field_that_changes_the_weights():
    cfg = tiny_cfg()
    saved = snapshot_from_config(cfg)
    assert snapshot_mismatches(saved, cfg) == []
    changes = {"FREEZE_BACKBONE": False, "DENSENET_SIZE": 96, "NUM_CLASSES": 9, "BACKBONE_WEIGHTS": "OTHER",
               "SEED": 1, "VAL_SIZE": 60, "BATCH_SIZE": 32, "NUM_EPOCHS": 7, "OPTIMIZER": "adamw",
               "LEARNING_RATE_HEAD": 0.5, "LEARNING_RATE_FULL": 0.5, "FLAG_BATCHNORM_EVAL": False}
    assert set(changes) == set(SNAPSHOT_FIELDS)                       # no field escapes this test
    for field, value in changes.items():
        other = dict(saved, **{field: value})
        found = snapshot_mismatches(other, cfg)
        assert len(found) == 1 and found[0].startswith(field)
    missing = dict(saved)
    del missing["SEED"]
    assert "SEED: missing" in snapshot_mismatches(missing, cfg)[0]


def test_model_modes_never_change_silently():
    assert resolve_mode(tiny_cfg(MODEL_MODE="train"), True, [])[0] == "train"
    with pytest.raises(FileNotFoundError):
        resolve_mode(tiny_cfg(MODEL_MODE="load"), False, [])
    with pytest.raises(ValueError):
        resolve_mode(tiny_cfg(MODEL_MODE="load"), True, ["SEED: checkpoint 1 vs config 42"])
    assert resolve_mode(tiny_cfg(MODEL_MODE="load"), True, [])[0] == "load"
    action, reason = resolve_mode(tiny_cfg(MODEL_MODE="auto"), False, [])
    assert action == "train" and "no checkpoint" in reason
    action, reason = resolve_mode(tiny_cfg(MODEL_MODE="auto"), True, ["BATCH_SIZE: checkpoint 64 vs config 32"])
    assert action == "train" and "BATCH_SIZE" in reason                 # it says WHY it retrains
    assert resolve_mode(tiny_cfg(MODEL_MODE="auto"), True, [])[0] == "load"


def test_debug_epochs_and_file_hash():
    assert debug_epochs(5) == (1, 5) and debug_epochs(1) == (1,) and debug_epochs(2) == (1, 2)
    with pytest.raises(ValueError):
        debug_epochs(5, "every_epoch")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.bin"
        f.write_bytes(b"banca virtuosa")
        assert file_sha256(f) == hashlib.sha256(b"banca virtuosa").hexdigest()


# ---------------------------------------------------------------- with torch (CPU, random weights)
def run_training(cfg, callback=None):
    torch = pytest.importorskip("torch")
    from model import bn_buffers, build_densenet, wrap_for_flag
    from training import set_seed, train_model
    data = synthetic_data_bundle(cfg)
    set_seed(cfg.SEED)                                              # as get_trained_model does
    model = wrap_for_flag(build_densenet(cfg, pretrained=False), cfg)
    bn_before = bn_buffers(model)
    history = train_model(model, data, cfg, torch.device("cpu"), epoch_callback=callback, log=lambda *_: None)
    return model, history, bn_before


def test_training_learns_keeps_batchnorm_frozen_and_is_reproducible():
    torch = pytest.importorskip("torch")
    from model import bn_buffers, bn_drift
    cfg = tiny_cfg(NUM_EPOCHS=3)
    model, history, bn_before = run_training(cfg)
    assert len(history) == 3 and history[-1]["train_loss"] < history[0]["train_loss"]
    assert bn_drift(bn_before, bn_buffers(model)) == 0.0            # FLAG_BATCHNORM_EVAL: statistics untouched
    again, history2, _ = run_training(cfg)
    assert torch.equal(model.classifier.weight, again.classifier.weight)    # same seed -> identical weights
    assert [h["val_acc"] for h in history] == [h["val_acc"] for h in history2]
    other, _, _ = run_training(tiny_cfg(NUM_EPOCHS=3, SEED=7))
    assert not torch.equal(model.classifier.weight, other.classifier.weight)  # another seed -> different weights


def test_debug_callback_runs_on_the_first_and_last_epoch_only():
    calls = []
    run_training(tiny_cfg(NUM_EPOCHS=3), callback=lambda m, e: calls.append(e))
    assert calls == [1, 3]
    calls.clear()
    run_training(tiny_cfg(NUM_EPOCHS=1), callback=lambda m, e: calls.append(e))
    assert calls == [1]


def test_checkpoint_round_trip_and_automatic_mode():
    torch = pytest.importorskip("torch")
    from training import get_trained_model
    with tempfile.TemporaryDirectory() as d:
        cfg = tiny_cfg(NUM_EPOCHS=1, CHECKPOINT_PATH=str(Path(d) / "m.pt"), MODEL_MODE="auto")
        data, device, quiet = synthetic_data_bundle(cfg), torch.device("cpu"), (lambda *_: None)
        _, first = get_trained_model(cfg, data, device, log=quiet, pretrained=False)
        assert first["action"] == "train" and "no checkpoint" in first["reason"]
        raw = torch.load(cfg.CHECKPOINT_PATH, map_location="cpu", weights_only=True)   # plain types only
        assert set(raw) == {"state_dict", "snapshot", "history", "versions"}
        assert all(type(v) is str for v in raw["versions"].values())
        _, second = get_trained_model(cfg, data, device, log=quiet, pretrained=False)
        assert second["action"] == "load" and second["sha256"] == first["sha256"]
        assert abs(second["val_acc_reloaded"] - second["val_acc_history"]) < 1e-6    # the round trip is faithful
        changed = tiny_cfg(NUM_EPOCHS=1, BATCH_SIZE=8, CHECKPOINT_PATH=cfg.CHECKPOINT_PATH)
        _, third = get_trained_model(changed, data, device, log=quiet, pretrained=False)
        assert third["action"] == "train" and "BATCH_SIZE" in third["reason"]
        with pytest.raises(ValueError):                              # MODEL_MODE=load refuses an incompatible file
            get_trained_model(tiny_cfg(NUM_EPOCHS=1, BATCH_SIZE=4, MODEL_MODE="load",
                                       CHECKPOINT_PATH=cfg.CHECKPOINT_PATH), data, device, log=quiet, pretrained=False)
