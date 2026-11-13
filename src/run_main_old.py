"""
    Run From the final notebook).

    Real run on Colab: MNIST -> DenseNet (train or load, per RUN_MODE) 
                         -> LR baseline -> metrics ->
    %run run_main.py
""" 
import argparse
import json
from pathlib import Path
import time
import warnings
import torch

import numpy as np

from baseline import fit_baseline
from config import CONFIG 

from data import data_to_input_lr, load_datasets
from data import  load_images_by_class

from logger import LOGGER
from metrics import accuracy, confusion, macro_f1, quadrants
from output import plot_predict_info 
from training import get_device, get_trained_model 

from utility import create_and_save_images_lists  
from xai.run_xai import run_exp 
from xai.run_xai import make_gradcam_callback


# Configuration Warnings 
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def phase_2_xai( cfg,
                 logger,
                 out_path,
                 flag_pretrained = True ) -> None :
    """
        Run xai Explainer : 
        - In this phase the model will be alwais pretrained

        - For each XAI technique, it calculates an explanation 
          with respect to the predicted class, pred.
          This means that the result maps answer the question:
          Which parts of the image contributed 
          to the prediction made by the model?
          Not necessarily to the image's actual class.
        
        - For Integrated Gradients, it also saves the completeness error:
    """
    t_start = time.time()
    
    device, warning = get_device(cfg)
    if warning:
        logger.debug(f"WARNING: {warning}")
    
    datasets_dict , loaders_dict , transform_cnn_dict = load_datasets()
    # trainval_dataset = datasets_dict["trainval"]
    # test_dataset = datasets_dict["test"]
    transform_val_cnn = transform_cnn_dict["val"]
    transform_test_cnn = transform_cnn_dict["test"]
    
    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]
    
    # Load tensor for train and test cnn
    train_loader_cnn = loaders_dict["cnn"]["train"]
    val_loader_cnn = loaders_dict["cnn"]["val"]
    test_loader_cnn = loaders_dict["cnn"]["test"]

    X_train_lr , y_train_lr , _  = data_to_input_lr(train_loader_lr,1, 
                                                    CONFIG.IMAGE_SIZE, 
                                                    CONFIG.IMAGE_SIZE)
    X_val_lr , y_val_lr , idx_val_lr = data_to_input_lr(val_loader_lr,1, 
                                               CONFIG.IMAGE_SIZE, 
                                               CONFIG.IMAGE_SIZE)

    X_test_lr , y_test_lr , idx_test = data_to_input_lr(test_loader_lr,1, 
                                                 CONFIG.IMAGE_SIZE, 
                                                 CONFIG.IMAGE_SIZE)

    # Logistic regression (train indices only)
    lr = fit_baseline(X_train_lr, y_train_lr, X_val_lr, y_val_lr,cfg,logger)
    # from here is different compare to phase 1
   
    list_img = []
    if cfg.LIST_IMAGES_SEL_PHASE_2 == cfg.LIST_IMAGES_SEL_CNN_OK:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_CNN_OK)
    elif cfg.LIST_IMAGES_SEL_PHASE_2 == cfg.LIST_IMAGES_SEL_CNN_KO:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_CNN_KO)

    if len(list_img) == 0 :
        raise ValueError("list_img non può essere vuota")

    info_cnn = {}
    if cfg.FLAG_LIST_VAL :
        callback = make_gradcam_callback(cfg,val_loader_cnn, device, out_path , list_img , logger)
    else:
        callback = make_gradcam_callback(cfg,test_loader_cnn, device, out_path , list_img , logger)
                                        
    # Cnn DenseNet
    model , info_cnn , history = get_trained_model(cfg,
                                            logger, 
                                            train_loader_cnn,
                                            val_loader_cnn,
                                            test_loader_cnn, 
                                            device,
                                            pretrained=flag_pretrained,
                                            epoch_callback=callback)
    
    if cfg.FLAG_LIST_VAL :
        xai_param_dict = {  "t_start": t_start,
                            "X_data": X_val_lr,
                            "y_data": y_val_lr,
                            "info": info_cnn,
                            "list_img": list_img,  
                            "out_path": out_path,   
                            "transform_cnn": transform_val_cnn,   
                            "dataloader_cnn": val_loader_cnn
                        }
    else:
        xai_param_dict = { "t_start": t_start,
                            "X_data": X_test_lr,
                            "y_data": y_test_lr,
                            "info": info_cnn,
                            "list_img": list_img,  
                            "out_path": out_path,      
                             "transform_cnn": transform_test_cnn,   
                             "dataloader_cnn": test_loader_cnn
                        }

    return run_exp( cfg, logger, model, lr, device , xai_param_dict) 
 
def phase_1_train(cfg,
                logger,
                out_path,
                flag_pretrained = False) -> None :
    """
        This is the first run.
        - TO begin with loads data , train and test the models lr and cnn.
        - After that show save checkpoint cnn model.
        - At The end show/save different outputs (confusion matrix , classification repors
        plot bar based and order on F1_Score metrics for different classes ) 
    """

    metrics = {}
    t_start = time.time()
    
    logger.debug(f"Output dir: {cfg.OUTPUT_DIR}")
    
    device, warning = get_device(cfg)
    if warning:
        logger.debug(f"WARNING: {warning}")
    
    _ , loaders_dict , _ = load_datasets()
    
    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]
    
    # Load tensor for train and test cnn
    train_loader_cnn = loaders_dict["cnn"]["train"]
    val_loader_cnn = loaders_dict["cnn"]["val"]
    test_loader_cnn = loaders_dict["cnn"]["test"]

    X_train_lr , y_train_lr , _  = data_to_input_lr(train_loader_lr,1, 
                                                    CONFIG.IMAGE_SIZE, 
                                                    CONFIG.IMAGE_SIZE)
    X_val_lr , y_val_lr , idx_val_lr = data_to_input_lr(val_loader_lr,1, 
                                               CONFIG.IMAGE_SIZE, 
                                               CONFIG.IMAGE_SIZE)
        
    X_test_lr , y_test_lr , idx_test = data_to_input_lr(test_loader_lr,1, 
                                                 CONFIG.IMAGE_SIZE, 
                                                 CONFIG.IMAGE_SIZE)
        
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train_lr, y_train_lr, X_val_lr, y_val_lr,cfg,logger)

    y_pred_val_lr = lr.predict(X_val_lr)
    val_accuracy_lr = accuracy(y_val_lr,y_pred_val_lr)

    y_pred_test_lr = lr.predict(X_test_lr)
    test_accuracy_lr = accuracy(y_test_lr, y_pred_test_lr)

    logger.debug(f"Val Accuracy (Logistic Regression): {val_accuracy_lr}")
    logger.debug(f"Test Accuracy (Logistic Regression): {test_accuracy_lr}")

    # plot_logistic_graph(lr.get_model(), CONFIG, CONFIG.OUTPUT_DIR, c=CONFIG.LR_C)
    plot_predict_info(y_test_lr, 
                      y_pred_test_lr,
                      "CLASSIFIER LR ON MNIST DATASET " , 
                      "TEST",
                      CONFIG,
                      "lr")

    
    if cfg.FLAG_LIST_VAL :
        create_and_save_images_lists( logger, 
                                          # X_val_lr,
                                          y_val_lr, 
                                          y_pred_val_lr, 
                                          idx_val_lr,
                                          cfg.CHECK_CLASSES, 
                                          out_path,
                                          cfg.FILE_IMG_LR_OK, 
                                          cfg.FILE_IMG_LR_KO,
                                          n_images=cfg.NUM_IMAGES_FOR_CLASS,
                                          flag_before_ko=True)
    else:
        create_and_save_images_lists( logger, 
                                  # X_test_lr,
                                  y_test_lr, 
                                  y_pred_test_lr, 
                                  idx_test,
                                  cfg.CHECK_CLASSES, 
                                  out_path,
                                  cfg.FILE_IMG_LR_OK, 
                                  cfg.FILE_IMG_LR_KO,
                                  n_images=cfg.NUM_IMAGES_FOR_CLASS,
                                  flag_before_ko=True)
    
    list_img = []
    if cfg.LIST_IMAGES_SEL_PHASE_1 == cfg.LIST_IMAGES_SEL_LR_OK:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_OK)
    elif cfg.LIST_IMAGES_SEL_PHASE_1 == cfg.LIST_IMAGES_SEL_LR_KO:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_KO)

    if len(list_img) == 0 :
        raise ValueError("list_img non può essere vuota")

    info_cnn = {}
    if cfg.FLAG_LIST_VAL :
        callback = make_gradcam_callback(cfg,val_loader_cnn, device, out_path , list_img , logger)
    else:
        callback = make_gradcam_callback(cfg,test_loader_cnn, device, out_path , list_img , logger)
                                        
    # Cnn DenseNet
    _ , info_cnn , history = get_trained_model(cfg,
                                                logger, 
                                                train_loader_cnn,
                                                val_loader_cnn,
                                                test_loader_cnn, 
                                                device,
                                                pretrained=flag_pretrained,
                                                epoch_callback=callback)

    last = history[-1] if history else None
    if last is None or "y_pred_val" not in last:
        raise ValueError("history senza predizioni di validation: rifare il training")

    if cfg.FLAG_LIST_VAL :
        create_and_save_images_lists(  logger, 
                                           last["y_true_val"],
                                           last["y_pred_val"],
                                           last["idx_val"],
                                           cfg.CHECK_CLASSES, 
                                           out_path,
                                           cfg.FILE_IMG_CNN_OK, 
                                           cfg.FILE_IMG_CNN_KO,
                                           n_images=cfg.NUM_IMAGES_FOR_CLASS,
                                           flag_before_ko=True)
    else:
        create_and_save_images_lists(  logger, 
                                       # info_cnn["X_test"],
                                       info_cnn["y_true"],
                                       info_cnn["y_pred"], 
                                       info_cnn["idx_test"],
                                       cfg.CHECK_CLASSES, 
                                       out_path,
                                       cfg.FILE_IMG_CNN_OK, 
                                       cfg.FILE_IMG_CNN_KO,
                                       n_images=cfg.NUM_IMAGES_FOR_CLASS,
                                       flag_before_ko=True)

    # alwais with test data
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
       
    q = quadrants(y_test_lr, y_pred_cnn, y_pred_test_lr)
        
    
    metrics = {  "densenet": {
                         "val_acc": val_acc_cnn, 
                         "test_acc": test_accuracy_cnn, 
                         "test_macro_f1": macro_f1(y_test_lr, y_pred_cnn, cfg.NUM_CLASSES),
                         "train_sec_cnn": train_sec_cnn,
                         "val_sec_cnn": val_sec_cnn},
                         "lr": {"val_acc": val_accuracy_lr, 
                         "test_acc": test_accuracy_lr, 
                         "test_macro_f1": macro_f1(y_test_lr, y_pred_test_lr, cfg.NUM_CLASSES),
                         "fit_seconds": train_sec_lr, 
                         "n_iter": lr.diagnostics["n_iter"]},
                         "quadrants": q, "gap_test_points": gap * 100,
                         "model": {k: info_cnn[k] for k in ("action", "reason", "path", "device")},
                         "history": info_cnn["history"], "total_seconds": time.time() - t_start 
                    }
        
    # default=str converte gli oggetti Path in str
    (out_path / "lr_cnn_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            
    np.savetxt(out_path / "confusion_densenet.csv", confusion(y_test_lr, y_pred_cnn, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
    np.savetxt(out_path / "confusion_lr.csv", confusion(y_test_lr, y_pred_test_lr, cfg.NUM_CLASSES), fmt="%d", delimiter=",")
        
    logger.info(f"\nDenseNet: val {val_acc_cnn:.4f}   | " \
                  f"test {test_accuracy_cnn:.4f} | " \
                  f"F1 {metrics['densenet']['test_macro_f1']:.4f} | {train_sec_cnn * 1000:.2f} ms/img")
            
    logger.info(f"LR:       val {val_accuracy_lr:.4f}   | " \
                  f"test {test_accuracy_lr:.4f} | " \
                  f"F1 {metrics['lr']['test_macro_f1']:.4f} | {train_sec_lr * 1000:.3f} ms/img")
            
    logger.info(f"gap on test: {gap * 100:.2f} points | quadrants: {q}")
    # print(f"checkpoint: {info_cnn['path']}\n | total {time.time() - t_start:.0f}s")
    logger.info(f"checkpoint: {info_cnn['path']}\n | total {time.perf_counter() - t_start:.0f}s")
    return metrics

def main(cfg=CONFIG, logger=LOGGER):
    """
        Main Function Of the project
    """

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Xai per Classificazione di Cifre in ambito Finance")

    # Define argument (--mode)
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'load'],
        default='train',
        help="Modalità di esecuzione (train o load)"
    )

    # Define argument (--phase) 9 = default 
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 9],
        default=9,
        help="Fase di esecuzione (9 per tutte le fasi - default, oppure 1, 2, 3)"
    )
    
    # parse args
    args = parser.parse_args()

    flag_pretrained = False
    
    if args.mode == 'load':
        # we can pass load also with phase 1
        flag_pretrained = True
        logger.info("Training avviato in modalità: LOAD")
    else:
        logger.info("Training avviato in modalità: TRAIN")

    # Gestione della fase da eseguire
    flag_phase_to_run = args.phase
    
    if flag_phase_to_run == cfg.RUN_PHASE:
        logger.info("Fase da eseguire: TUTTE LE FASI (Default - 9)")
    elif flag_phase_to_run in (1, 2, 3):
        logger.info("Fase da eseguire: %d",flag_phase_to_run)
    else:
        raise ValueError(f"Attenzione!! Numero fase da eseguire non valido: {flag_phase_to_run}")

    # dir and file output
    out = Path(cfg.OUTPUT_DIR) 
    out.mkdir(parents=True, exist_ok=True)

    # Esecuzione condizionale o sequenziale basata sulla fase
    if flag_phase_to_run in ( cfg.RUN_PHASE , cfg.RUN_PHASE_1 ):
        logger.info("------------------------------------------------------------------")
        phase_1_train(cfg, logger, out, flag_pretrained)
        logger.info("------------------------------------------------------------------")
        
    if flag_phase_to_run in ( cfg.RUN_PHASE , cfg.RUN_PHASE_2 ):
        logger.info("------------------------------------------------------------------")
        phase_2_xai(cfg, logger, out)
        logger.info("------------------------------------------------------------------")

    if flag_phase_to_run in ( cfg.RUN_PHASE , cfg.RUN_PHASE_3 ):
        logger.info("------------------------------------------------------------------")
        # phase_2_xai(cfg, logger, out,flag_pretrained)
        
        logger.info("------------------------------------------------------------------")

    logger.debug("Programma Terminato")

if __name__ == "__main__":
    main()
