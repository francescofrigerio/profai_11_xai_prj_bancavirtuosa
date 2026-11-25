"""Stage 1 driver (TEMPORARY: the final notebook of stage 4 absorbs it).

Real run on Colab: MNIST -> DenseNet (train or load, per MODEL_MODE) -> LR baseline -> metrics ->
'Verifiche' table. Run it twice: the first time it trains (MODE: train) and writes the
checkpoint, the second time it must load it (MODE: load) and give the same numbers.

    %run run_stage1.py
"""
import json
import time
from pathlib import Path

import numpy as np

from baseline import fit_baseline
from checks import Verifications, assert_lr_input, assert_proba
from config import CONFIG
from data import load_mnist, to_lr_input
from metrics import accuracy, confusion, macro_f1, quadrants, seconds_per_image
from model import predict_logits, softmax_np
from training import get_trained_model, pick_device

LR_EXPECTED_VAL_ACC = (0.90, 0.94)       # design: about 0.92 (phase 0: 0.9222)
GAP_EXPECTED = (0.03, 0.07)              # design: 4-6 points (phase 0: 4.78 on validation)


def main(cfg=CONFIG):
    import torch
    out = Path(cfg.OUTPUT_DIR) / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    v = Verifications()
    t_start = time.time()

    device, warning = pick_device(cfg)
    if warning:
        print("WARNING:", warning)
        v.record("device", "WARN", warning)
    data = load_mnist(cfg)
    split = data.split
    v.expect("split sizes", len(split.idx_val) == cfg.VAL_SIZE and len(split.idx_train) == 60000 - cfg.VAL_SIZE,
             f"train {len(split.idx_train)}, validation {len(split.idx_val)}, test {len(data.y_test)}")
    v.expect("split is disjoint", set(split.idx_train).isdisjoint(set(split.idx_val)), "no image in both sets")

    # ---- DenseNet
    model, info = get_trained_model(cfg, data, device)
    v.record("model mode", "INFO", f"{info['action']} ({info['reason']})")
    v.expect("checkpoint round trip", abs(info["val_acc_reloaded"] - info["val_acc_history"]) < 1e-3,
             f"reloaded validation accuracy {info['val_acc_reloaded']:.4f} vs training run {info['val_acc_history']:.4f}")
    val_acc_d = info["val_acc_reloaded"]
    v.expect("DenseNet validation accuracy >= MIN_VAL_ACCURACY", val_acc_d >= cfg.MIN_VAL_ACCURACY,
             f"{val_acc_d:.4f} vs {cfg.MIN_VAL_ACCURACY}")
    logits = predict_logits(model, data.X_test, cfg, device)
    assert_proba(softmax_np(logits), cfg)
    pred_d = logits.argmax(1)
    y_test = data.y_test.numpy()
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    sec_d = seconds_per_image(lambda: predict_logits(model, data.X_test[:1000], cfg, device), 1000, sync)

    # ---- Logistic regression (train indices only)
    X_trainval = to_lr_input(data.X_trainval)
    assert_lr_input(X_trainval, cfg)
    y_trainval = data.y_trainval.numpy()
    lr = fit_baseline(X_trainval, y_trainval, split.idx_train, cfg)
    X_test = to_lr_input(data.X_test)
    assert_lr_input(X_test, cfg)
    assert_proba(lr.predict_proba(X_test[:1000]), cfg)
    pred_l = lr.predict(X_test)
    val_acc_l = accuracy(y_trainval[split.idx_val], lr.predict(X_trainval[split.idx_val]))
    sec_l = seconds_per_image(lambda: lr.predict(X_test[:1000]), 1000)
    facts = lr.check_exactness(X_trainval[split.idx_val][:2000])
    v.expect("LR converged", not lr.diagnostics["convergence_warning"],
             f"{lr.diagnostics['n_iter']} iterations (LR_MAX_ITER={cfg.LR_MAX_ITER})")
    v.expect("LR validation accuracy in the expected range", LR_EXPECTED_VAL_ACC[0] <= val_acc_l <= LR_EXPECTED_VAL_ACC[1],
             f"{val_acc_l:.4f} (design expects about 0.92)")
    v.expect("LR logit is exactly w.x + b", facts["decision_exact_max_err"] < 1e-8,
             f"max error {facts['decision_exact_max_err']:.1e}")
    v.expect("LR coefficients sum to ~0 over the classes", facts["coef_colsum_max"] < 1e-3,
             f"max {facts['coef_colsum_max']:.1e} (l2: relative = absolute coefficients)")
    v.expect("LR classes aligned with the digits", facts["classes_ok"], "classes_ == 0..9")

    # ---- Comparison on the test set
    acc_d, acc_l = accuracy(y_test, pred_d), accuracy(y_test, pred_l)
    gap = acc_d - acc_l
    v.expect("DenseNet beats the LR on the test set", gap > 0, f"test accuracy {acc_d:.4f} vs {acc_l:.4f}")
    v.expect("gap between the models as the design expects", GAP_EXPECTED[0] <= gap <= GAP_EXPECTED[1],
             f"{gap * 100:.2f} points (design: 4-6 points on validation)")
    q = quadrants(y_test, pred_d, pred_l)
    v.expect("the four quadrants sum to the test size", sum(q.values()) == len(y_test), str(q))

    metrics = {
        "densenet": {"val_acc": val_acc_d, "test_acc": acc_d, "test_macro_f1": macro_f1(y_test, pred_d, cfg.NUM_CLASSES),
                     "seconds_per_image": sec_d},
        "lr": {"val_acc": val_acc_l, "test_acc": acc_l, "test_macro_f1": macro_f1(y_test, pred_l, cfg.NUM_CLASSES),
               "seconds_per_image": sec_l, "fit_seconds": lr.diagnostics["fit_seconds"], "n_iter": lr.diagnostics["n_iter"]},
        "quadrants": q, "gap_test_points": gap * 100,
        "model": {k: info[k] for k in ("action", "reason", "sha256", "device")},
        "history": info["history"], "total_seconds": time.time() - t_start,
    }
    (out / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(out / "confusion_densenet.csv", confusion(y_test, pred_d, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    np.savetxt(out / "confusion_lr.csv", confusion(y_test, pred_l, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    (out / "verifications.md").write_text(v.to_markdown() + "\n")

    print("\n" + v.to_markdown())
    print(f"\nDenseNet: val {val_acc_d:.4f} | test {acc_d:.4f} | F1 {metrics['densenet']['test_macro_f1']:.4f} | {sec_d * 1000:.2f} ms/img")
    print(f"LR:       val {val_acc_l:.4f} | test {acc_l:.4f} | F1 {metrics['lr']['test_macro_f1']:.4f} | {sec_l * 1000:.3f} ms/img")
    print(f"gap on test: {gap * 100:.2f} points | quadrants: {q}")
    print(f"checkpoint sha256: {info['sha256']}\nworst status: {v.worst()} | total {time.time() - t_start:.0f}s")
    return metrics, v


if __name__ == "__main__":
    main()
