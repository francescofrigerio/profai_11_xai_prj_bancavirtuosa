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
from training import get_device, get_trained_model ,get_evaluate_model
from output import plot_logistic_graph , plot_predict_info , save_dict_to_json , load_dict_from_json
from utility import run_create_list_images , run_check_class_ok_ko

# Configurazione dei Warnings (spostata DOPO tutti gli import) ---
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def phase_2( cfg,
             logger,
             out_path,
             flag_pretrained = True ) -> None :
    """
        Evaluate Logistic Regression
        with n_images = 50/100 e list_class = [2, 3, 5, 8],
        Vorresti selezionare almeno un'immagine per ciascuna classe, se disponibile.
        Questo significa che potresti restituire 4 immagini anche se ne hai richieste 3.

        con n_images = 10 e 3 classi, la divisione intera assegna 3 posti
        per classe: ne utilizzi al massimo 9, non 10.
        Non è il nostro caso visto che abbiamo 1_images=100 e len(list_class)=4
    """

     
    list_class = cfg.IMAGES_CHECK_CLASSES
    
    if len(list_class) == 0 :
        raise ValueError("list_class non può essere vuota")
    else:
        n_for_class = max(1, cfg.NUM_IMAGES_FOR_CLASS // len(list_class))

    logger.debug(f"Immagini per classe: {n_for_class}")
    device, warning = get_device(cfg)
    if warning:
        logger.debug("WARNING:", warning)

    # con n_images = 10 e 3 classi, la divisione intera assegna 3 posti
    # per classe: ne utilizzi al massimo 9, non 10.
    # START LOAD DATA
    datasets_dict , loaders_dict = load_datasets()
    test_datasets = datasets_dict["test"]
        
    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]
        
    # Load tensor for train and test cnn
    # train_loader_cnn = loaders_dict["cnn"]["train"]
    # val_loader_cnn = loaders_dict["cnn"]["val"]
    test_loader_cnn = loaders_dict["cnn"]["test"]
            
    X_train , y_train = data_to_input_lr(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val = data_to_input_lr(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)

    X_test , y_test = data_to_input_lr(test_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    # END LOAD DATA

    # START PROCESS LR MODEL
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)
    y_pred =  lr.predict(X_test)

    logger.debug("Indici KO/OK di controllo per classe:")

    run_check_class_ok_ko( logger, 
                            y_test,
                            y_pred , 
                            list_class,
                            "LR")

    list_images_lr_ok , list_images_lr_ko  = run_create_list_images( logger,
                                                                     X_test,
                                                                     y_test,
                                                                     y_pred ,
                                                                     list_class,
                                                                     n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                                     flag_before_ko=True)
    if (len(list_images_lr_ok) < cfg.NUM_IMAGES_MIN_OK ):
        list_images_lr_ok , _  = run_create_list_images( logger,
                                                        X_test,
                                                        y_test,
                                                        y_pred ,
                                                        list_class,
                                                        n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                        flag_before_ko=False)

    logger.info(f"Risultato Finale LR: {len(list_images_lr_ko)} {len(list_images_lr_ok)}")
    # END PROCESS LR MODEL
    
    # START PROCESS CNN MODEL 
    X_test , y_test , y_pred = get_evaluate_model( cfg,
                                                   logger,
                                                   flag_pretrained,
                                                   test_loader_cnn,
                                                   device )
    
    logger.debug("Indici KO di controllo per classe:")
    
    run_check_class_ok_ko( logger, 
                            y_test,
                            y_pred , 
                            list_class,
                            "CNN")

    list_images_cnn_ok , list_images_cnn_ko  = run_create_list_images(  logger,
                                                                        X_test,
                                                                        y_test,
                                                                        y_pred ,
                                                                        list_class,
                                                                        n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                                        flag_before_ko=True)
    if (len(list_images_cnn_ok) < cfg.NUM_IMAGES_MIN_OK ):
            list_images_cnn_ok , _  = run_create_list_images( logger,
                                                              X_test,
                                                              y_test,
                                                              y_pred ,
                                                              list_class,
                                                              n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                              flag_before_ko=False)
    logger.debug(f"Risultato Finale CNN: {len(list_images_cnn_ko)} {len(list_images_cnn_ok)}")
    # END PROCESS LR MODEL

    # Save files for phase 3
    save_dict_to_json(out_path / "images_lr_ok.json" , list_images_lr_ok)
    save_dict_to_json(out_path / "images_lr_ko.json" , list_images_lr_ko)
    save_dict_to_json(out_path / "images_cnn_ok.json" , list_images_cnn_ok)
    save_dict_to_json(out_path / "images_cnn_ko.json" , list_images_cnn_ko)

def phase_1(cfg,
           logger,
           out_path,
           flag_pretrained = False) :
    """
        This is the first run.
        - TO begin with loads data , train and test the models lr and cnn.
        - After that show save checkpoint cnn model.
        - At The end show/save different outputs (confusion matrix , classification repors
        plot bar based and order on F1_Score metrics for different classes ) 
    """

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
        
    # To run on codespace set cfg.RUN_TESTS=True
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
        (out_path / "lr_cnn_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            
        np.savetxt(out_path / "confusion_densenet.csv", confusion(y_test, y_pred_cnn, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
        np.savetxt(out_path / "confusion_lr.csv", confusion(y_test, y_pred_test_lr, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
        
        print(f"\nDenseNet: val {val_acc_cnn:.4f}   | " \
                  f"test {test_accuracy_cnn:.4f} | " \
                  f"F1 {metrics['densenet']['test_macro_f1']:.4f} | {train_sec_cnn * 1000:.2f} ms/img")
            
        print(f"LR:       val {val_accuracy_lr:.4f}   | " \
                  f"test {test_accuracy_lr:.4f} | " \
                  f"F1 {metrics['lr']['test_macro_f1']:.4f} | {train_sec_lr * 1000:.3f} ms/img")
            
        print(f"gap on test: {gap * 100:.2f} points | quadrants: {q}")
        # print(f"checkpoint: {info_cnn['path']}\n | total {time.time() - t_start:.0f}s")
        print(f"checkpoint: {info_cnn['path']}\n | total {time.perf_counter() - t_start:.0f}s")
        return metrics
    
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

    if flag_pretrained is False :
        phase_1(cfg,logger,out,flag_pretrained)
    else:
        phase_2(cfg,logger,out,flag_pretrained)

    logger.info("Programma Terminato")

if __name__ == "__main__":
    main()
