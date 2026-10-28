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

from baseline import fit_baseline
from config import CONFIG
from data import data_to_input_lr, load_datasets
from logger import LOGGER
from metrics import accuracy, confusion, macro_f1, quadrants
from training import get_device, get_trained_model
from output import plot_logistic_graph , plot_predict_info

# Configurazione dei Warnings (spostata DOPO tutti gli import) ---
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def main(cfg=CONFIG,logger=LOGGER):
    """
        Main Function Of the project
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Xai per Classificazione di Cifre in ambito Finance")

    # Definisci l'argomento (--mode, accettando solo LOAD o PROD)
    parser.add_argument( '--mode',
                        type=str,
                        choices=['train','load'],
                        default='train',
                        help="Modalità di esecuzione (default: PROD)"
                    )

    # Definisci l'argomento (--cnn, accettando solo True o False)
    parser.add_argument( '--cnn',
                            type=bool,
                            choices=[True,False],
                            default=True,
                            help="Modalità di esecuzione (default: PROD)"
                        )
    
    # Effettua il parse degli argomenti lanciati da terminale
    args = parser.parse_args()

    flag_pretrained = False
    # Assegna la configurazione globale corretta in base all'input
    
    if args.mode == 'load':
        flag_pretrained = True
        logger.info("Training avviato in modalità: LOAD")
   
    # definizione/creazione dir e file di di output
    out = Path(cfg.OUTPUT_DIR) 
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info(f"Output dir: {cfg.OUTPUT_DIR}")

    device, warning = get_device(cfg)
    if warning:
        logger.info("WARNING:", warning)

    datasets_dict , loaders_dict = load_datasets()
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
    logger.info(f"Val Accuracy (Logistic Regression): {val_accuracy_lr}")
    logger.info(f"Test Accuracy (Logistic Regression): {test_accuracy_lr}")

    plot_logistic_graph(lr.get_model(), CONFIG, CONFIG.OUTPUT_DIR, c=CONFIG.LR_C)
    
    # y_pred_val_lr = lr.predict(X_val)
    y_pred_test_lr = lr.predict(X_test)
    plot_predict_info(y_test, 
                      y_pred_test_lr,
                      "CLASSIFIER LR ON MNIST DATASET " , 
                      "TEST",
                      CONFIG,
                      "lr")
    
    logger.info(f" RUN_TESTS FLAG {cfg.RUN_TESTS}")
    if cfg.RUN_TESTS is False:
        # Cnn DenseNet
        info_cnn = get_trained_model(cfg,
                                    logger, 
                                    test_datasets,
                                    train_loader_cnn,
                                    val_loader_cnn,
                                    test_loader_cnn, 
                                    device,
                                    pretrained=flag_pretrained)

        plot_predict_info(info_cnn["y_true"], 
                          info_cnn["y_pred"],
                          "CLASSIFIER CNN ON MNIST DATASET " , 
                          "TEST",
                          CONFIG,
                          "cnn")
    
        y_pred_cnn = info_cnn["y_pred"]
        test_accuracy_cnn = info_cnn["test_accuracy"]
        val_acc_cnn = info_cnn["val_acc_history"]
        
        train_sec_lr = lr.diagnostics["fit_seconds"]
        train_sec_cnn = info_cnn["train_seconds"]
        val_sec_cnn = info_cnn["val_seconds"]
    
        gap = test_accuracy_cnn - test_accuracy_lr
   
        q = quadrants(y_test, y_pred_cnn, y_pred_test_lr)
    

        metrics = {  "densenet": {
                     "val_acc": val_acc_cnn, 
                     "test_acc": test_accuracy_cnn, 
                     "test_macro_f1": macro_f1(y_test, y_pred_cnn, cfg.NUM_CLASSES),
                     "train_sec_cnn": train_sec_cnn,
                     "val_sec_cnn": val_sec_cnn},
                     "lr": {"val_acc": val_accuracy_lr, 
                     "test_acc": test_accuracy_lr, 
                     "test_macro_f1": macro_f1(y_test, y_pred_test_lr, cfg.NUM_CLASSES),
                     "fit_seconds": train_sec_lr, 
                     "n_iter": lr.diagnostics["n_iter"]},
                     "quadrants": q, "gap_test_points": gap * 100,
                     "model": {k: info_cnn[k] for k in ("action", "reason", "path", "device")},
                     "history": info_cnn["history"], "total_seconds": time.time() - t_start 
        }
    
        # default=str converte gli oggetti Path in str
        (out / "lr_cnn_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        
        np.savetxt(out / "confusion_densenet.csv", confusion(y_test, y_pred_cnn, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
        np.savetxt(out / "confusion_lr.csv", confusion(y_test, y_pred_test_lr, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    
        # (out / "verifications.md").write_text(v.to_markdown() + "\n")

        print(f"\nDenseNet: val {val_acc_cnn:.4f}   | " \
              f"test {test_accuracy_cnn:.4f} | " \
              f"F1 {metrics['densenet']['test_macro_f1']:.4f} | {train_sec_cnn * 1000:.2f} ms/img")
        
        print(f"LR:          val {val_accuracy_lr:.4f} | " \
              f"test {test_accuracy_lr:.4f} | " \
              f"F1 {metrics['lr']['test_macro_f1']:.4f} | {train_sec_lr * 1000:.3f} ms/img")
        
        print(f"gap on test: {gap * 100:.2f} points | quadrants: {q}")
        print(f"checkpoint: {info_cnn['path']}\n | total {time.time() - t_start:.0f}s")
        return metrics
    
    logger.info(
        "Programma Terminato")


if __name__ == "__main__":
    main()
