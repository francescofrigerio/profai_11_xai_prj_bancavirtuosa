"""Run From the final notebook).

Real run on Colab: MNIST -> DenseNet (train or load, per MODEL_MODE) 
                         -> LR baseline -> metrics ->
    %run run_main.py
"""


import os
import sys
# import random
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

import warnings

# 1. Ignore warning based on deprecate args of models
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")

# 2. Ignore FutureWarning of library (ex. Pandas o NumPy on Torchvision)
warnings.filterwarnings('ignore', category=FutureWarning)

from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# computer vision with pytorch
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
from torchvision.models import DenseNet121_Weights, densenet121

# preprocessd dataset with pytorch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from config import CONFIG
from logger import LOGGER
from train import get_trained_model, get_device
# from baseline import baseline
from baseline import fit_baseline
from data import load_data_baseline

# from data import load_mnist, to_lr_input
# from metrics import accuracy, confusion, macro_f1, quadrants, seconds_per_image
# from model import predict_logits, softmax_np
# from training import get_trained_model, pick_device
# import torch
# 
# LR_EXPECTED_VAL_ACC = (0.90, 0.94)       # design: about 0.92 (phase 0: 0.9222)
# GAP_EXPECTED = (0.03, 0.07)              # design: 4-6 points (phase 0: 4.78 on validation)

def main(cfg=CONFIG,logger=LOGGER):
    
    out = Path(cfg.OUTPUT_DIR) / "prod"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    device, warning = get_device(cfg)
    if warning:
        logger.info("WARNING:", warning)


    X_train ,X_val, X_test , y_train ,y_val ,  y_test = load_data_baseline()
    logger.debug(f" X_train numpy {X_train.shape}")
    logger.debug(f" X_test numpy {X_test.shape}")

    # Logistic regression (train indices only)
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)
    
    # data = load_mnist(cfg)
    # split = data.split
    
    # ---- DenseNet
    model, info = get_trained_model(cfg, data, device)
    
    val_acc_d = info["val_acc_reloaded"]
    
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
    lr = fit_baseline(X_trainval, y_trainval, split.idx_train, cfg,logger)
    X_test = to_lr_input(data.X_test)
    assert_lr_input(X_test, cfg)
    assert_proba(lr.predict_proba(X_test[:1000]), cfg)
    pred_l = lr.predict(X_test)
    val_acc_l = accuracy(y_trainval[split.idx_val], lr.predict(X_trainval[split.idx_val]))
    sec_l = seconds_per_image(lambda: lr.predict(X_test[:1000]), 1000)
    facts = lr.check_exactness(X_trainval[split.idx_val][:2000])
    

    # ---- Comparison on the test set
    acc_d, acc_l = accuracy(y_test, pred_d), accuracy(y_test, pred_l)
    gap = acc_d - acc_l
   
    q = quadrants(y_test, pred_d, pred_l)
    

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