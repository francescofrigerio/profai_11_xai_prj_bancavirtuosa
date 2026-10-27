"""
    Run From the final notebook).

    Real run on Colab: MNIST -> DenseNet (train or load, per MODEL_MODE) 
                         -> LR baseline -> metrics ->
    %run run_main.py
""" 
import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.metrics import classification_report

from baseline import fit_baseline
from config import CONFIG
from data import data_to_input_lr, load_datasets
from logger import LOGGER
from metrics import accuracy, confusion, macro_f1, quadrants
from training import get_device, get_trained_model
from output import plot_logistic_graph , plot_confusion_matrix_lr

# Configurazione dei Warnings (spostata DOPO tutti gli import) ---
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def main(cfg=CONFIG,logger=LOGGER):
    """
        Main Function Of the project
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Xai per Classificazione di Cifre in ambito Finance")

    # Definisci l'argomento (es. --mode, accettando solo LOAD o PROD)
    parser.add_argument( '--mode',
                        type=str,
                        choices=['PROD','LOAD'],
                        default='PROD',
                        help="Modalità di esecuzione (default: PROD)"
                    )
    
    # Effettua il parse degli argomenti lanciati da terminale
    args = parser.parse_args()

    flag_pretrained = False
    # Assegna la configurazione globale corretta in base all'input
    
    if args.mode == 'LOAD':
        flag_pretrained = True
        print("Training avviato in modalità: PRODUCTION")
   
    # definizione/creazione dir e file di di output
    out = Path(cfg.OUTPUT_DIR) 
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
    
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)

    val_accuracy_lr = accuracy(y_val, lr.predict(X_val))
    test_accuracy_lr = accuracy(y_test, lr.predict(X_test))
    logger.info(f"Accuratezza di test (Logistic Regression): {test_accuracy_lr}")

    plot_logistic_graph(lr, CONFIG, CONFIG.OUTPUT_DIR, c=CONFIG.LR_C)
    plot_confusion_matrix_lr(lr,X_test,y_test, "CLASSIFIER LR ON MNIST DATASET " , "TEST",CONFIG)
    # plot barcode for class
    
    y_pred_val_lr = lr.predict(X_val)
    y_pred_test_lr = lr.predict(X_test)
    report = classification_report(y_test,y_pred_test_lr, target_names=[str(i) for i in range(cfg.NUM_CLASSES)],digits=4)
    print(report)
    
    
    
    # Cnn DenseNet
    model, info = get_trained_model(cfg,
                                    logger, 
                                    test_datasets,
                                    train_loader_cnn,
                                    val_loader_cnn,
                                    test_loader_cnn, 
                                    device,
                                    pretrained=flag_pretrained)
    
    y_pred_cnn = info["y_pred"]
    test_accuracy_cnn = info["test_accuracy"]
    val_acc_cnn = info["val_acc_history"]
    train_sec_cnn = info["train_seconds"]
    val_sec_cnn = info["val_seconds"]

    sec_l = lr.diagnostics["fit_seconds"]
    sec_d = val_sec_cnn
    
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
               "fit_seconds": sec_l, 
               "n_iter": lr.diagnostics["n_iter"]},
        "quadrants": q, "gap_test_points": gap * 100,
        "model": {k: info[k] for k in ("action", "reason", "path", "device")},
        "history": info["history"], "total_seconds": time.time() - t_start 
    }
    
    (out / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(out / "confusion_densenet.csv", confusion(y_test, y_pred_cnn, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    np.savetxt(out / "confusion_lr.csv", confusion(y_test, y_pred_test_lr, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    
    # (out / "verifications.md").write_text(v.to_markdown() + "\n")

    print(f"\nDenseNet: val {val_acc_cnn:.4f}   | test {test_accuracy_cnn:.4f} | F1 {metrics['densenet']['test_macro_f1']:.4f} | {sec_d * 1000:.2f} ms/img")
    print(f"LR:         val {val_accuracy_lr:.4f} | test {test_accuracy_lr:.4f} | F1 {metrics['lr']['test_macro_f1']:.4f} | {sec_l * 1000:.3f} ms/img")
    print(f"gap on test: {gap * 100:.2f} points | quadrants: {q}")
    print(f"checkpoint: {info['path']}\n | total {time.time() - t_start:.0f}s")
    return metrics


if __name__ == "__main__":
    main()
