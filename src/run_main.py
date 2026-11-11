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
from utility import save_dict_to_json

from logger import LOGGER
from metrics import accuracy, confusion, macro_f1, quadrants
from output import plot_predict_info 
from training import get_device, get_trained_model 
from training import get_evaluate_model


from utility import  run_check_class_ok_ko , verify_output_files_exist
from utility import create_and_save_images_lists  , run_create_list_images
from xai.run_xai import run_exp , make_gradcam_callback


# Configuration Warnings 
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def phase_2_xai( cfg,
                 logger,
                  out_path ) -> None :
    """
        Run xai Explainer

        - For each XAI technique, it calculates an explanation 
          with respect to the predicted class, pred.
          This means that the result maps answer the question:
          Which parts of the image contributed 
          to the prediction made by the model?
          Not necessarily to the image's actual class.
        
        - For Integrated Gradients, it also saves the completeness error:


    """
    if len(cfg.CHECK_CLASSES) == 0 :
        raise ValueError("list_class mancante {len(cfg.CHECK_CLASSES)} ")
    
    n_for_class = max(1, cfg.NUM_IMAGES_FOR_CLASS // len(cfg.CHECK_CLASSES))

    t_start = time.time()
    
    logger.debug(f"Immagini per classe: {n_for_class}")
    device, warning = get_device(cfg)
    if warning:
        logger.debug(f"WARNING: {warning}")

    datasets_dict , loaders_dict , transform_cnn_dict = load_datasets()
    # trainval_dataset = datasets_dict["trainval"]
    test_datasets = datasets_dict["test"]
    # train_idx = datasets_dict["train_idx"]

    train_loader_cnn = loaders_dict["cnn"]["train"]
    val_loader_cnn = loaders_dict["cnn"]["val"]
    test_loader_cnn = loaders_dict["cnn"]["test"]

    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]
       
    X_train_lr , y_train_lr , _  = data_to_input_lr(train_loader_lr,
                                              1, 
                                              CONFIG.IMAGE_SIZE,
                                              CONFIG.IMAGE_SIZE)
    X_val_lr , y_val_lr , _ = data_to_input_lr(val_loader_lr,
                                         1,
                                         CONFIG.IMAGE_SIZE,
                                         CONFIG.IMAGE_SIZE)

    # test_loader_lr has shuffle=False 
    # and uses datasettest_base directly, test_idx
    # corresponds to the original idx of the test set.
    X_test_lr, y_test_lr, test_idx_lr = data_to_input_lr(test_loader_lr,
                                                1,
                                                CONFIG.IMAGE_SIZE,
                                                CONFIG.IMAGE_SIZE
                                                )

    expected_idx = torch.arange(len(test_idx_lr))

    assert torch.equal(test_idx_lr, expected_idx), ( "Errore: non contiene indici consecutivi "
                                                    "da 0 a len(test_idx_lr)-1 nell'ordine previsto."
                                                )

    logger.debug("Controllo test_idx superato.")
    logger.info(f" test_loader_cnn {type(test_loader_cnn)}")
    logger.info(f"has dataset:{hasattr(test_loader_cnn, "dataset")}")

    # Logistic regression (train indices only)
    lr = fit_baseline(X_train_lr, y_train_lr, X_val_lr, y_val_lr,cfg,logger)

    list_img = []
    # It could be pass an input value
    if cfg.LIST_IMAGES_SEL == cfg.LIST_IMAGES_SEL_LR_OK:
        list_img = load_images_by_class(logger,                                 
                                        out_path / cfg.FILE_IMG_LR_OK)
    elif cfg.LIST_IMAGES_SEL == cfg.LIST_IMAGES_SEL_LR_KO:
        list_img = load_images_by_class(logger,
                                        out_path / cfg.FILE_IMG_LR_KO)
    elif cfg.LIST_IMAGES_SEL == cfg.LIST_IMAGES_SEL_CNN_OK:
        list_img = load_images_by_class(logger,                              out_path / cfg.FILE_IMG_CNN_OK)
    elif cfg.LIST_IMAGES_SEL == cfg.LIST_IMAGES_SEL_CNN_KO:
        list_img = load_images_by_class(logger,
                                        out_path / cfg.FILE_IMG_CNN_KO)
    else:
        list_img = load_images_by_class(logger,
                                        out_path / cfg.FILE_IMG_CNN_KO)

    if len(list_img) == 0 :
        raise ValueError("list_img non può essere vuota")

    # The test DataLoader has shuffle=False, and the indices 
    # # it returns correspond to those in test_base.
    # So we can pass test_loader_cnn
    logger.info(f" test_loader_cnn {type(test_loader_cnn)}")
    logger.info(f"has dataset: {hasattr(test_loader_cnn, "dataset")}")
    callback = make_gradcam_callback(cfg,test_loader_cnn, device, out_path , list_img , logger)

    model, info = get_trained_model(cfg,
                            logger, 
                            train_loader_cnn,
                            val_loader_cnn,
                            test_loader_cnn, 
                            device,
                            pretrained=True,    
                            epoch_callback=callback )

    xai_param_dict = { "test_datasets" : test_datasets,
                        "t_start": t_start,
                        "X_test_lr": X_test_lr,
                        "y_test_lr": y_test_lr,
                        "info": info,
                        "list_img": list_img,  
                        "out_path": out_path,      
                        "transform_train_cnn": transform_cnn_dict["train"]
                     }
    
    return run_exp( cfg, logger, model, lr, device , xai_param_dict) 
       
def phase_1_load( cfg,
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

     
    
    
    
    if len(cfg.CHECK_CLASSES) == 0 :
        raise ValueError(f"La lista di classi da controllare mancante {cfg.CHECK_CLASSES}")
    
    n_for_class = max(1, cfg.NUM_IMAGES_FOR_CLASS // len(cfg.CHECK_CLASSES))

    logger.debug(f"Immagini per classe: {n_for_class}")
    device, warning = get_device(cfg)
    if warning:
        logger.debug(f"WARNING: {warning}")

    # START LOAD DATA
    _ , loaders_dict , _ = load_datasets()
    
    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]
        
    # Load tensor for train and test cnn
    test_loader_cnn = loaders_dict["cnn"]["test"]
            
    X_train , y_train , _ = data_to_input_lr(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val , _ = data_to_input_lr(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)

    X_test , y_test , idx_test = data_to_input_lr(test_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    # END LOAD DATA

    # START PROCESS LR MODEL
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)
    y_pred =  lr.predict(X_test)

    logger.debug("Indici KO/OK di controllo per classe:")

    num_ok , num_ko = run_check_class_ok_ko( logger, 
                            y_test,
                            y_pred , 
                            cfg.CHECK_CLASSES,
                            "LR")
    if num_ok == 0 or num_ko == 0 :
        raise ValueError(f"num ok o ko non possono essere {num_ok} {num_ko} per LR")

    list_images_lr_ok , list_images_lr_ko  = run_create_list_images( logger,
                                                                     X_test,
                                                                     y_test,
                                                                     y_pred,
                                                                     idx_test,
                                                                     cfg.CHECK_CLASSES,
                                                                     n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                                     flag_before_ko=True)
    if (len(list_images_lr_ok) < cfg.NUM_IMAGES_MIN_OK ):
        list_images_lr_ok , _  = run_create_list_images( logger,
                                                        X_test,
                                                        y_test,
                                                        y_pred ,
                                                        idx_test,
                                                        cfg.CHECK_CLASSES,
                                                        n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                        flag_before_ko=False)

    logger.debug(f"Risultato Finale LR: {len(list_images_lr_ko)} {len(list_images_lr_ok)}")
    # END PROCESS LR MODEL
    
    # START PROCESS CNN MODEL 
    X_test , y_test , y_pred , idx_test = get_evaluate_model( cfg,
                                                   logger,
                                                   flag_pretrained,
                                                   test_loader_cnn,
                                                   device )
    
    logger.debug("Indici KO di controllo per classe:")
    
    num_ok , num_ko = run_check_class_ok_ko( logger, 
                            y_test,
                            y_pred , 
                            cfg.CHECK_CLASSES,
                            "CNN")
    if num_ok == 0 or num_ko == 0 :
        raise ValueError(f"num ok o ko non possono essere {num_ok} {num_ko} per CNN")

    list_images_cnn_ok , list_images_cnn_ko  = run_create_list_images(  logger,
                                                                        X_test,
                                                                        y_test,
                                                                        y_pred,
                                                                        idx_test,
                                                                        cfg.CHECK_CLASSES,
                                                                        n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                                        flag_before_ko=True)
    if (len(list_images_cnn_ok) < cfg.NUM_IMAGES_MIN_OK ):
        list_images_cnn_ok , _  = run_create_list_images( logger,
                                                              X_test,
                                                              y_test,
                                                              y_pred,
                                                              idx_test,
                                                              cfg.CHECK_CLASSES,
                                                              n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                              flag_before_ko=False)
    logger.debug(f"Risultato Finale CNN: {len(list_images_cnn_ko)} {len(list_images_cnn_ok)}")
    # END PROCESS LR MODEL

    # Save files for phase 3
    save_dict_to_json(out_path / cfg.FILE_IMG_LR_OK , list_images_lr_ok)
    save_dict_to_json(out_path / cfg.FILE_IMG_LR_KO , list_images_lr_ko)
    save_dict_to_json(out_path / cfg.FILE_IMG_CNN_OK , list_images_cnn_ok)
    save_dict_to_json(out_path / cfg.FILE_IMG_CNN_KO , list_images_cnn_ko)

    return verify_output_files_exist(out_path,cfg)

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
    
    # datasets_dict , loaders_dict , transform_dict = load_datasets()
    # test_datasets = datasets_dict["test"]
    # idx_dataset = transform_dict["train"]
    _ , loaders_dict , _ = load_datasets()
    
    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]
    
    # Load tensor for train and test cnn
    train_loader_cnn = loaders_dict["cnn"]["train"]
    val_loader_cnn = loaders_dict["cnn"]["val"]
    test_loader_cnn = loaders_dict["cnn"]["test"]

    logger.info(f"test_loader_cnn type: {type(test_loader_cnn)}")

    X_train_lr , y_train_lr , _  = data_to_input_lr(train_loader_lr,1, 
                                                    CONFIG.IMAGE_SIZE, 
                                                    CONFIG.IMAGE_SIZE)
    X_val_lr , y_val_lr , _ = data_to_input_lr(val_loader_lr,1, 
                                               CONFIG.IMAGE_SIZE, 
                                               CONFIG.IMAGE_SIZE)
        
    X_test_lr , y_test_lr , idx_test = data_to_input_lr(test_loader_lr,1, 
                                                 CONFIG.IMAGE_SIZE, 
                                                 CONFIG.IMAGE_SIZE)
        
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train_lr, y_train_lr, X_val_lr, y_val_lr,cfg,logger)
    
    val_accuracy_lr = accuracy(y_val_lr, lr.predict(X_val_lr))
    y_pred_test_lr = lr.predict(X_test_lr)
    test_accuracy_lr = accuracy(y_test_lr, y_pred_test_lr)
    logger.info(f"Val Accuracy (Logistic Regression): {val_accuracy_lr}")
    logger.info(f"Test Accuracy (Logistic Regression): {test_accuracy_lr}")

    # plot_logistic_graph(lr.get_model(), CONFIG, CONFIG.OUTPUT_DIR, c=CONFIG.LR_C)
    plot_predict_info(y_test_lr, 
                      y_pred_test_lr,
                      "CLASSIFIER LR ON MNIST DATASET " , 
                      "TEST",
                      CONFIG,
                      "lr")

    
    
    # logger.info(f"y_pred_test_lr shape: {y_pred_test_lr.shape}")
    # logger.info(f"y_test_lr shape: {y_test_lr.shape}")
    # logger.info(f"idx_dataset type: {type(idx_dataset)}")
    create_and_save_images_lists( logger, 
                                  X_test_lr,
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
    if cfg.LIST_IMAGES_SEL == cfg.LIST_IMAGES_SEL_LR_OK:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_OK)
    elif cfg.LIST_IMAGES_SEL == cfg.LIST_IMAGES_SEL_LR_KO:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_KO)

    if len(list_img) == 0 :
            raise ValueError("list_img non può essere vuota")

    logger.info(f"Pass param test_loader_cnn type: {type(test_loader_cnn)}")

    callback = make_gradcam_callback(cfg,test_loader_cnn, device, out_path , list_img , logger)
                                        
    # Cnn DenseNet
    _ , info_cnn = get_trained_model(cfg,
                                        logger, 
                                        train_loader_cnn,
                                        val_loader_cnn,
                                        test_loader_cnn, 
                                        device,
                                        pretrained=flag_pretrained,
                                        epoch_callback=callback)

    create_and_save_images_lists(  logger, 
                                   info_cnn["X_test"],
                                   info_cnn["y_true"],
                                   info_cnn["y_pred"], 
                                   info_cnn["idx_test"],
                                   cfg.CHECK_CLASSES, 
                                   out_path,
                                   cfg.FILE_IMG_CNN_OK, 
                                   cfg.FILE_IMG_CNN_KO,
                                   n_images=cfg.NUM_IMAGES_FOR_CLASS,
                                   flag_before_ko=True)
    
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

import argparse
from pathlib import Path

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
        flag_pretrained = True
        logger.info("Training avviato in modalità: LOAD")
    else:
        logger.info("Training avviato in modalità: TRAIN")

    # Gestione della fase da eseguire
    flag_phase_to_run = args.phase
    
    if flag_phase_to_run == cfg.RUN_PHASE:
        logger.info("Fase da eseguire: TUTTE LE FASI (Default - 9)")
    elif flag_phase_to_run in (1, 2, 3):
        logger.info(f"Fase da eseguire: {flag_phase_to_run}")
    else:
        raise ValueError(f"Attenzione!! Numero fase da eseguire non valido: {flag_phase_to_run}")

    # dir and file output
    out = Path(cfg.OUTPUT_DIR) 
    out.mkdir(parents=True, exist_ok=True)

    # Esecuzione condizionale o sequenziale basata sulla fase
    if flag_phase_to_run == cfg.RUN_PHASE or flag_phase_to_run == cfg.RUN_PHASE_1:
        logger.info("------------------------------------------------------------------")
        phase_1_train(cfg, logger, out, flag_pretrained)
        logger.info("------------------------------------------------------------------")
        
    if flag_phase_to_run == cfg.RUN_PHASE or flag_phase_to_run == cfg.RUN_PHASE_2:
        logger.info("------------------------------------------------------------------")
        # phase_2_xai(cfg, logger, out)
        logger.info("------------------------------------------------------------------")

    if flag_phase_to_run == cfg.RUN_PHASE or flag_phase_to_run == cfg.RUN_PHASE_3:
            logger.info("------------------------------------------------------------------")
            # phase_2_xai(cfg, logger, out)
            logger.info("------------------------------------------------------------------")

    # load module pretrained and write json file
    # phase_1_load(cfg,logger,out,flag_pretrained)

    logger.debug("Programma Terminato")

def main_old(cfg=CONFIG,logger=LOGGER):
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

    # arg --cnn only True o False
    parser.add_argument( '--phase',
                            type=int,
                            choices=[1,2,3],
                            default=0,
                            help="Modalità di esecuzione (default: PROD)"
                        )
    
    # parse args
    args = parser.parse_args()

    flag_pretrained = False
    
    if args.mode == 'load':
        flag_pretrained = True
        logger.info("Training avviato in modalità: LOAD")

    flag_phase_to_run = cfg.RUN_PHASE
    logger.info(f"Fase da eseguire: DEFAULT {cfg.RUN_PHASE}")
    if args.phase in (1,2,3) :
        flag_phase_to_run = args.phase
        logger.info(f"Fase da eseguire: DEFAULT {flag_phase_to_run}")
    else:
         raise ValueError(f"attenzione!! numero fase da eseguire non valido {args.phase}")
    

    # dir and file output
    out = Path(cfg.OUTPUT_DIR) 
    out.mkdir(parents=True, exist_ok=True)

    if flag_phase_to_run == cfg.RUN_PHASE or flag_phase_to_run == cfg.RUN_PHASE_1 :
        phase_1_train(cfg,logger,out,flag_pretrained)
    elif flag_phase_to_run == cfg.RUN_PHASE or flag_phase_to_run == cfg.RUN_PHASE_2 :
        phase_2_xai(cfg,logger,out)

    # load module pretrained and write json file
    # phase_1_load(cfg,logger,out,flag_pretrained)

    logger.debug("Programma Terminato")

if __name__ == "__main__":
    main()
