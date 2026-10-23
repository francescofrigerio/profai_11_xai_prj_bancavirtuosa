"""Training and checkpoints.

  - MODEL_MODE: "train" always trains; "load" loads and refuses a missing or incompatible
    checkpoint; "auto" loads a compatible checkpoint and otherwise trains, SAYING WHY. Nothing
    changes silently.
  - Fixed number of epochs, the FINAL weights are kept (no early stopping, E20).
  - The checkpoint stores a snapshot of the parameters that change the weights (E05): a
    checkpoint trained with other values is incompatible.
  - The Grad-CAM debug of the first and last epoch (D09) is produced by the `epoch_callback`
    hook, so that this module does not depend on the explanation techniques.

torch is imported lazily: the pure logic (modes, snapshot, hashes) works without it.
"""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import numpy as np

SNAPSHOT_FIELDS = (
    "FREEZE_BACKBONE", "DENSENET_SIZE", "NUM_CLASSES", "BACKBONE_WEIGHTS", "SEED", "VAL_SIZE",
    "BATCH_SIZE", "NUM_EPOCHS", "OPTIMIZER", "LEARNING_RATE_HEAD", "LEARNING_RATE_FULL",
    "FLAG_BATCHNORM_EVAL",
)


# ----------------------------------------------------------------------------------------
# Pure logic
# ----------------------------------------------------------------------------------------
def snapshot_from_config(cfg) -> dict:
    return {k: getattr(cfg, k) for k in SNAPSHOT_FIELDS}


def snapshot_mismatches(saved: dict, cfg) -> list:
    """Human-readable differences between a checkpoint snapshot and the current config."""
    out = []
    for k in SNAPSHOT_FIELDS:
        if k not in saved:
            out.append(f"{k}: missing in the checkpoint")
        elif saved[k] != getattr(cfg, k):
            out.append(f"{k}: checkpoint {saved[k]!r} vs config {getattr(cfg, k)!r}")
    return out


def resolve_mode(cfg, ckpt_exists: bool, mismatches: list) -> tuple:
    """(action, reason). action is "train" or "load"."""
    if cfg.MODEL_MODE == "train":
        return "train", "MODEL_MODE=train"
    if cfg.MODEL_MODE == "load":
        if not ckpt_exists:
            raise FileNotFoundError(f"MODEL_MODE=load but there is no checkpoint at {cfg.CHECKPOINT_PATH}")
        if mismatches:
            raise ValueError("MODEL_MODE=load but the checkpoint is incompatible: " + "; ".join(mismatches))
        return "load", "MODEL_MODE=load"
    if not ckpt_exists:
        return "train", f"no checkpoint at {cfg.CHECKPOINT_PATH}"
    if mismatches:
        return "train", "checkpoint incompatible with the config: " + "; ".join(mismatches)
    return "load", "compatible checkpoint found"


def debug_epochs(num_epochs: int, mode: str = "first_last") -> tuple:
    """Epochs after which the debug callback runs: the first and the last (once if equal)."""
    if mode != "first_last":
        raise ValueError(f"unknown debug mode {mode!r}")
    return tuple(sorted({1, num_epochs}))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(cfg) -> tuple:
    """(device, warning or None). A visible warning on CPU: the full run would be very slow."""
    import torch
    want_cuda = cfg.DEVICE == "cuda" or (cfg.DEVICE == "auto" and torch.cuda.is_available())
    if cfg.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but no GPU is available")
    device = torch.device("cuda" if want_cuda else "cpu")
    warning = None if want_cuda else "NO GPU: 'Esegui tutto' would be very slow. Use a GPU runtime."
    return device, warning


# ----------------------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------------------
def save_checkpoint(path, model, cfg, history) -> str:
    """Saves weights + snapshot + history. Returns the SHA-256 of the file (E14)."""
    import torch
    import torchvision
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "snapshot": snapshot_from_config(cfg),
        "history": history,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
    }, path)
    return file_sha256(path)


def read_checkpoint(path) -> dict:
    import torch
    return torch.load(path, map_location="cpu")


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
def evaluate(model, X, y, idx, cfg, device, batch_size: int = 256) -> tuple:
    """(loss, accuracy) on X[idx]; X, y, idx already on `device`."""
    import torch
    import torch.nn.functional as F
    from data import to_densenet_input
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            out = model(to_densenet_input(X[b], cfg.DENSENET_SIZE))
            total_loss += F.cross_entropy(out, y[b], reduction="sum").item()
            correct += (out.argmax(1) == y[b]).sum().item()
    return total_loss / len(idx), correct / len(idx)


def train_model(model, data, cfg, device, epoch_callback=None, log=print) -> list:
    """Fixed number of epochs. Returns the history (one dict per epoch).
    `epoch_callback(model, epoch)` runs after the validation of the epochs in debug_epochs()."""
    import torch
    import torch.nn.functional as F
    from checks import assert_densenet_input, assert_logits
    from data import to_densenet_input
    from model import enter_train_mode, freeze_backbone, make_optimizer

    set_seed(cfg.SEED)
    if cfg.FREEZE_BACKBONE:
        freeze_backbone(model)
    optimizer = make_optimizer(model, cfg)
    X, y = data.X_trainval.to(device), data.y_trainval.to(device)
    idx_train = torch.as_tensor(data.split.idx_train, device=device)
    idx_val = torch.as_tensor(data.split.idx_val, device=device)
    generator = torch.Generator().manual_seed(cfg.SEED)
    callback_epochs = debug_epochs(cfg.NUM_EPOCHS, cfg.GRADCAM_DEBUG_EPOCHS)
    history = []
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        enter_train_mode(model, cfg)
        t0 = time.time()
        perm = torch.randperm(len(idx_train), generator=generator).to(device)
        run_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(0, len(perm), cfg.BATCH_SIZE):
            b = idx_train[perm[i:i + cfg.BATCH_SIZE]]
            inputs = to_densenet_input(X[b], cfg.DENSENET_SIZE)
            out = model(inputs)
            if epoch == 1 and i == 0:                      # contract on the real first batch
                assert_densenet_input(inputs, cfg)
                assert_logits(out, cfg)
            loss = F.cross_entropy(out, y[b])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            run_loss += loss.detach() * len(b)
            correct += (out.argmax(1) == y[b]).sum()
        train_seconds = time.time() - t0
        val_loss, val_acc = evaluate(model, X, y, idx_val, cfg, device)
        history.append({
            "epoch": epoch, "train_loss": run_loss.item() / len(idx_train),
            "train_acc": correct.item() / len(idx_train), "val_loss": val_loss, "val_acc": val_acc,
            "train_seconds": train_seconds})
        log(f"epoch {epoch}/{cfg.NUM_EPOCHS}: train loss {history[-1]['train_loss']:.4f} "
            f"acc {history[-1]['train_acc']:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} "
            f"| {train_seconds:.0f}s")
        if epoch_callback is not None and epoch in callback_epochs:
            epoch_callback(model, epoch)
    return history


def get_trained_model(cfg, data, device, epoch_callback=None, log=print, pretrained=True) -> tuple:
    """Builds the model according to MODEL_MODE. Returns (model in eval mode, info dict).
    `pretrained=False` exists for the tests only (no ImageNet download)."""
    import torch
    from model import build_densenet, validate_layers, wrap_for_flag

    path = Path(cfg.CHECKPOINT_PATH)
    saved = read_checkpoint(path) if path.exists() else None
    mismatches = snapshot_mismatches(saved["snapshot"], cfg) if saved else []
    action, reason = resolve_mode(cfg, saved is not None, mismatches)
    log(f"MODE: {action} ({reason})")
    if action == "load":
        model = wrap_for_flag(build_densenet(cfg, pretrained=False), cfg)
        model.load_state_dict(saved["state_dict"])
        history = saved["history"]
    else:
        set_seed(cfg.SEED)                      # BEFORE building: the random head must be reproducible
        model = wrap_for_flag(build_densenet(cfg, pretrained=pretrained), cfg)
        model.to(device)
        history = train_model(model, data, cfg, device, epoch_callback=epoch_callback, log=log)
        save_checkpoint(path, model, cfg, history)
    model = model.to(device).eval()
    validate_layers(model, cfg)
    X, y = data.X_trainval.to(device), data.y_trainval.to(device)
    _, val_acc = evaluate(model, X, y, torch.as_tensor(data.split.idx_val, device=device), cfg, device)
    info = {
        "action": action, "reason": reason, "history": history, "sha256": file_sha256(path),
        "val_acc_reloaded": val_acc, "val_acc_history": history[-1]["val_acc"] if history else None,
        "device": str(device),
    }
    return model, info
