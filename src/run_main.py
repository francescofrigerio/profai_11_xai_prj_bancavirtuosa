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
# import time

# computer vision with pytorch
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
from torchvision.models import DenseNet121_Weights, densenet121

# preprocessd dataset with pytorch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image

from config import CONFIG
from logger import LOGGER
from train import get_trained_model, get_device
# from baseline import baseline
from baseline import fit_baseline
# from data import load_data_baseline
from data import load_datasets
from data import data_to_input_lr
from output import plot_logistic_graph , plot_confusion_matrix_lr
from sklearn.metrics import accuracy_score
from metrics import confusion, macro_f1, quadrants

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

    datasets_dict , loaders_dict = load_datasets()

    trainval_datasets = datasets_dict["trainval"]
    test_datasets = datasets_dict["test"]

    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]

    # Load tensor for train and test cnn
    train_loader_cnn = loaders_dict["cnn"]["train"]
    val_loader_cnn = loaders_dict["cnn"]["val"]
    test_loader_cnn = loaders_dict["cnn"]["test"]
    
    X_train , y_train = data_to_input_lr(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val = data_to_input_lr(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_test , y_test = data_to_input_lr(test_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    
    logger.info(f" X_train numpy {X_train.shape}")
    logger.info(f" X_test numpy {X_test.shape}")

    # Logistic regression (train indices only)
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)

    val_accuracy_lr = accuracy_score(y_val, lr.predict(X_val))
    test_accuracy_lr = accuracy_score(y_test, lr.predict(X_test))
    logger.info(f"Accuratezza di test (Logistic Regression): {test_accuracy_lr}")

    # plot_logistic_graph(lr, CONFIG, CONFIG.OUTPUT_DIR, c=CONFIG.LR_C)
    # plot_confusion_matrix_lr(lr,X_test,y_test, "CLASSIFIER LR ON MNIST DATASET " , "TEST",CONFIG)

    y_pred_val_lr=model.predict(X_val)
    y_pred_test_lr=model.predict(X_test)
    report = classification_report(y_test,y_pred_test_lr, target_names=[str(i) for i in range(cfg.NUM_CLASSES)],digits=4)
    print(report)
    
    # plot barcode for class
    
    # Cnn DenseNet
    model, info = get_trained_model(cfg,
                                    logger, 
                                    test_datasets,
                                    train_loader_cnn,
                                    val_loader_cnn,
                                    test_loader_cnn, 
                                    device)
    
    y_pred_cnn = info["y_pred"]
    test_accuracy_cnn = info["test_accuracy"]
    val_acc_cnn = info["val_acc_history"]
    train_sec_cnn = info["train_seconds"]
    val_sec_cnn = info["val_seconds"]
    
    
    gap = test_accuracy_cnn - test_accuracy_lr
   
    q = quadrants(y_test, y_pred_cnn, y_pred_test_lr)
    

    metrics = {
        "densenet": {
                     "val_acc": val_acc_cnn, 
                     "test_acc": test_accuracy_cnn, 
                     "test_macro_f1": macro_f1(y_test, y_pred_cnn, cfg.NUM_CLASSES),
                     "train_sec_cnn": train_sec_cnn,
                     "val_sec_cnn": val_sec_cnn},
        "lr": {"val_acc": val_accuracy_lr, 
               "test_acc": test_accuracy_lr, 
               "test_macro_f1": macro_f1(y_test, y_pred_test_lr, cfg.NUM_CLASSES),
               "fit_seconds": lr.diagnostics["fit_seconds"], 
               "n_iter": lr.diagnostics["n_iter"]},
        "quadrants": q, "gap_test_points": gap * 100,
        "model": {k: info[k] for k in ("action", "reason", "path", "device")},
        "history": info["history"], "total_seconds": time.time() - t_start 
    }
    
    (out / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(out / "confusion_densenet.csv", confusion(y_test, y_pred_cnn, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    np.savetxt(out / "confusion_lr.csv", confusion(y_test, y_pred_test_lr, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    (out / "verifications.md").write_text(v.to_markdown() + "\n")

    print(f"\nDenseNet: val {val_acc_cnn:.4f}   | test {test_accuracy_cnn:.4f} | F1 {metrics['densenet']['test_macro_f1']:.4f} | {sec_d * 1000:.2f} ms/img")
    print(f"LR:         val {val_accuracy_lr:.4f} | test {test_accuracy_lr:.4f} | F1 {metrics['lr']['test_macro_f1']:.4f} | {sec_l * 1000:.3f} ms/img")
    print(f"gap on test: {gap * 100:.2f} points | quadrants: {q}")
    print(f"checkpoint: {info['path']}\n | total {time.time() - t_start:.0f}s")
    return metrics


if __name__ == "__main__":
    main()