"""
    Run From the final notebook).

    Real run on Colab: MNIST -> DenseNet (train or load, per RUN_MODE) 
                         -> LR baseline -> metrics ->
    %run run_main.py
""" 
import argparse
import json
import csv
from pathlib import Path
import time
import warnings
# import torch
import numpy as np

# Conflict between colab and last numy version
# where could ben substitide from np.trapezoid
# AttributeError: module 'numpy' has no attribute 'trapz'
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from baseline import fit_baseline
from config import CONFIG 
from constants import GROUPS_RESULT_OK_KO , GROUPS_RESULT_BOTH_OK , GROUPS_RESULT_BOTH_KO
from constants import GROUPS_RESULT_LR_OK_CNN_KO , GROUPS_RESULT_LR_KO_CNN_OK 

from data import data_to_input_lr, load_datasets
from utility import  load_images_by_class , read_text_for_html


from logger import LOGGER
from metrics import accuracy, confusion, macro_f1, quadrants 
from outputs import plot_predict_info , plot_quadrants_balance_info
from outputs import section_display_title , section_display_content 
from training import get_device, get_trained_model , evaluate_model

from utility import create_and_save_images_lists , require , save_quadrant_balance_to_json
from utility select_thin_thick_examples ,select_thin_thick_from_maps_json , validate_maps_json_positions

from xai.explore_xai import run_exp , make_gradcam_callback
from xai.compare_xai import run_compare_group_xai
from xai.compare_xai import run_compare_group_xai_1 , run_compare_group_xai_2
from xai.outputs_xai import plot_mask_sensitivity , plot_masks_for_image
from xai.outputs_xai import plot_metrics_all , plot_spearman_vs_iou , inspect_digit_masks

from xai import build_techniques
from xai.shap_explainer import make_background


# Configuration Warnings 
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def compare_2_xai( cfg,
                 logger,
                 out_path,
                 flag_pretrained = True ) -> None :
    """
        Run xai Explainer to compare: 
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
    
    _ , loaders_dict , transform_cnn_dict = load_datasets()
    transform_val_cnn = transform_cnn_dict["val"]
    
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
    
    logger.debug("Test di riproducibilità superato con successo!")
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train_lr, y_train_lr, X_val_lr, y_val_lr,cfg,logger)

    logger.debug( f"LogisticRegression fitted | "
                    f"n_iter={lr.diagnostics['n_iter']} | "
                    f"val_acc={lr.diagnostics['val_accuracy']:.6f}"
                )
    
    # run reproducibility check
    repro_results = lr.check_reproducibility(X_train_lr, y_train_lr, cfg)
    if not all(repro_results.values()):
        failed_checks = [key for key, value in repro_results.items() if not value]
        raise RuntimeError(f"Attenzione: test di riproducibilità fallito! {failed_checks}")
    
    y_pred_val_lr = lr.predict(X_val_lr)

    # from here is different compare to phase 1
    list_img = []
    
    list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_CNN_KO,n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS)

    logger.debug( f"after load_images_by_class len list images ={len(list_img)}")
    
    if len(list_img) == 0 :
        raise ValueError("list_img non può essere vuota")

    callback = None
    if ( cfg.PLOT_DEBUG_ON ):
        callback = make_gradcam_callback(cfg,val_loader_cnn, device, out_path , list_img , logger)
                                      
    # Cnn DenseNet
    model , info_cnn , history = get_trained_model(cfg,
                                            logger, 
                                            train_loader_cnn,
                                            val_loader_cnn,
                                            test_loader_cnn, 
                                            device,
                                            pretrained=flag_pretrained,
                                            epoch_callback=callback)

    _ ,  y_pred_cnn_val , y_true_cnn_val, idx_val_cnn = evaluate_model(model, 
                                                                val_loader_cnn,
                                                                device)
    
    # here we verifiy the alignment
    # It works because because valloader_lr e valloader_cnn has shuffle=False
    # not works :TypeError list indices must be integers or slices, not str
    # require((idx_val_lr == history['idx_val']).all(),f"Attenzione!! 1 indici non allineati tra cnn ed lr")
    require((idx_val_lr == idx_val_cnn).all(),f"Attenzione!! 2 indici non allineati tra cnn ed lr")
    # assert (y_val_lr == y_true_cnn_val).all()
    quadrants_info = save_quadrant_balance_to_json( cfg,
                                                    logger, 
                                                    y_val_lr,
                                                    y_pred_val_lr, 
                                                    y_pred_cnn_val,
                                                    # history['y_pred_val'], 
                                                    idx_val_lr,
                                                    cfg.CHECK_CLASSES, 
                                                    out_path )
    
    if ( cfg.PLOT_DEBUG_ON ):
        plot_quadrants_balance_info(out_path / "quadrants_distribution.png", 
                                    quadrants_info, 
                                    logger)

    list_img_ok = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_KO,n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS)
    
    list_img_ko = load_images_by_class(logger,out_path / cfg.FILE_IMG_CNN_KO ,n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS )

    if len(list_img_ok) == 0 or len(list_img_ko) == 0:
            raise ValueError("list_img non possono essere vuote")
    
    
    

    # built ONCE and shared by all groups
    background = make_background(transform_val_cnn, cfg, logger, device)
    techniques = build_techniques(model, cfg, background)

    ctx_param_dict = {"t_start": t_start, 
                        "X_data": X_val_lr, 
                        "y_data": y_val_lr,
                        "y_pred_lr": y_pred_val_lr, 
                        "info": info_cnn, 
                        "out_path": out_path,
                        "dataloader_cnn": val_loader_cnn,
                        "org_idx_cmp_shap_ig": 3161,
                        "background" : background
                        }
    
    all_records_1, all_results_1 = [], {}
    all_records_2, all_results_2 = [], {}

    
    
    GROUP_SETS_BY_MODE = { "all": GROUPS_RESULT_OK_KO,
                            "both_ok": GROUPS_RESULT_BOTH_OK,
                            "both_ko": GROUPS_RESULT_BOTH_KO,
                            "lr_ok_cnn_ko": GROUPS_RESULT_LR_OK_CNN_KO,
                            "lr_ko_cnn_ok": GROUPS_RESULT_LR_KO_CNN_OK,
                        }

    selected_mode = getattr(cfg, "SAMPLE_GROUP_SELECTION", "all")
    if selected_mode not in GROUP_SETS_BY_MODE:
        raise ValueError(f"Valore non valido per cfg.SAMPLE_GROUP_SELECTION: {selected_mode!r}. "
                        f"Valori ammessi: {list(GROUP_SETS_BY_MODE.keys())}"
                        )

    active_groups = GROUP_SETS_BY_MODE[selected_mode]
    logger.info("Selezione gruppi XAI attiva: mode=%s -> %s", selected_mode, list(active_groups.keys()))
    for group in active_groups:
        list_images = load_images_by_class(logger, 
                                           out_path / f"images_{group}.json",
                                           n_images_for_class=cfg.XAI_IMAGES_FOR_CLASS)
        if not list_images:
            logger.warning("Gruppo %s vuoto: salto", group)
            continue

        # logger.info("START run_compare_group_xai_1 3 return params")
        results_1, records_1 , source_row_1 = run_compare_group_xai_1( cfg, 
                                                                          logger, 
                                                                          model, 
                                                                          device,
                                                                          techniques,
                                                                          lr, 
                                                                          ctx_param_dict,
                                                                          group,
                                                                          list_images ) 
                                                            
            all_results_1[group] = results_1
            all_records_1 += records_1

            # logger.info("START run_compare_group_xai_2 3 return params")
            results_2, records_2 = run_compare_group_xai_2( cfg, 
                                                        logger, 
                                                        results_1,
                                                        source_row_1,
                                                        model, 
                                                        device,
                                                        techniques,
                                                        ctx_param_dict,
                                                        group )

            all_results_2[group] = results_2
            all_records_2 += records_2
    
    if all_records_1:
        with open(out_path / "xai_metrics_all_1.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_records_1[0]))
            w.writeheader()
            w.writerows(all_records_1)
            
    if all_records_2:
        with open(out_path / "xai_metrics_all_2.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_records_2[0]))
            w.writeheader()
            w.writerows(all_records_2)

    return X_val_lr , y_val_lr ,idx_val_lr


def explore_2_xai( cfg,
                   logger,
                   out_path,
                   flag_pretrained = True ) -> None :
    """
        Run xai Explainer to explore: 
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
    
    _ , loaders_dict , transform_cnn_dict = load_datasets()
    transform_val_cnn = transform_cnn_dict["val"]
    
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

       
    logger.debug("Test di riproducibilità superato con successo!")
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train_lr, y_train_lr, X_val_lr, y_val_lr,cfg,logger)

    # run reproducibility check
    repro_results = lr.check_reproducibility(X_train_lr, y_train_lr, cfg)
    if not all(repro_results.values()):
        failed_checks = [key for key, value in repro_results.items() if not value]
        raise RuntimeError(f"Attenzione: test di riproducibilità fallito! {failed_checks}")
    
    y_pred_val_lr = lr.predict(X_val_lr)
    
    # from here is different compare to phase 1
    ÿ st 훱翴  = []
    list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_CNN_KO, n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS)

    if len(list_img) == 0 :
        raise ValueError("list_img non può essere vuota")

    callback = make_gradcam_callback(cfg,val_loader_cnn, device, out_path , list_img , logger)
                                      
    # Cnn DenseNet
    model , info_cnn , history = get_trained_model(cfg,
                                            logger, 
                                            train_loader_cnn,
                                            val_loader_cnn,
                                            test_loader_cnn, 
                                            device,
                                            pretrained=flag_pretrained,
                                            epoch_callback=callback)
    
    # here we verifiy the alignment
    # It works because because valloader_lr e valloader_cnn has shuffle=False
    # not works very well : TypeError list indices must be integers or slices, not str
    # require((idx_val_lr == history['idx_val']).all(),f"Attenzione!! indici non allineati tra cnn ed lr")
    
    list_img_ok = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_KO,n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS)
    list_img_ko = load_images_by_class(logger,out_path / cfg.FILE_IMG_CNN_KO,n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS)

    if len(list_img_ok) == 0 or len(list_img_ko) == 0:
            raise ValueError("list_img non possono essere vuote")
    
    xai_param_dict_ok = {  "t_start": t_start,
                        "X_data": X_val_lr,
                        "y_data": y_val_lr,
                        "info": info_cnn,
                        "list_img": list_img_ok,  
                        "out_path": out_path,   
                        "transform_cnn": transform_val_cnn,   
                        "dataloader_cnn": val_loader_cnn,
                        "flag_error_pred_cnn": False,
                        "type_list":  cfg.FILE_IMG_LR_OK.split(".")[0]
                    }
   

    result_ok = run_exp( cfg, logger, model, lr, device , xai_param_dict_ok) 

    xai_param_dict_ko = {  "t_start": t_start,
                            "X_data": X_val_lr,
                            "y_data": y_val_lr,
                            "info": info_cnn,
                            "list_img": list_img_ko,  
                            "out_path": out_path,   
                            "transform_cnn": transform_val_cnn,   
                            "dataloader_cnn": val_loader_cnn,
                            "flag_error_pred_cnn": True ,
                            "type_list":  cfg.FILE_IMG_LR_OK.split(".")[0]
                        }

       
    result_ko = run_exp( cfg, logger, model, lr, device , xai_param_dict_ko) 

    return result_ok , result_ko

def inspect_2_xai(cfg, X_data = None , y_data = None , idx_val = None , quantiles=(0.2, 0.8)):
    
    """
        analysis pre compare group to choice Treshold mask
        The final print score is a useful estimate for selecting examples, 
        not a perfect measure of significance: it depends on the threshold.
        This function start selecting thin and thick examples.
    """
    
    # X_data, y_data, = ctx["X_data"], ctx["y_data"]
    X_val_lr = x_data
    y_val_lr = y_data
    idx_val_lr = idx_val
    if X_val_lr is None and :
        _ , loaders_dict , _ = load_datasets()
        
        # Load tensor for train and test baseline
        val_loader_lr = loaders_dict["lr"]["val"]
        
        # Load tensor 
        X_val_lr , y_val_lr , idx_val_lr = data_to_input_lr(val_loader_lr,1, 
                                                CONFIG.IMAGE_SIZE, 
                                                 CONFIG.IMAGE_SIZE)

    selected = select_thin_thick_examples(X_val_lr, y_val_lr, cfg)
    
    # positions to pass at view function 
    positions = [ position
                    for examples in selected.values()
                        for position in (examples["thin"], examples["thick"])
                ]
    inspect_digit_masks(positions, X_val_lr, cfg)
    print("START SELECTED DIGIT MASK")
    print(selected)
    print("END SELECTED DIGIT MASK")
    return selected , positions
    
def analysis_2_xai(cfg,
                 logger,
                 out_path,
                 x_data=None,
                 idx_val_lr=None ):
    """
          X_data = X_val_lr must be the same value of the function compare 2 xai
          In the next version we can modify both to pass argument instead of
          read again the dataset
    """

    
    X_val_lr = x_data
    idx_to_pos = None
    # idx_val_lr = None
    # This check is appropriate if the default value is None. 
    # It's important Don't use `if not X_val_lr with a NumPy array or a tensor, 
    # as it may lead to an ambiguity error.
    if X_val_lr is None:
        datasets_dict , loaders_dict , _ = load_datasets()
        trainval_dataset = datasets_dict["trainval"]
        
        # Load tensor for train and test baseline
        val_loader_lr = loaders_dict["lr"]["val"]
        
        # Load tensor 
        X_val_lr , y_val_lr , idx_val_lr = data_to_input_lr(val_loader_lr,1, 
                                                            cfg.IMAGE_SIZE, 
                                                            cfg.IMAGE_SIZE)
                                                 
        idx_to_pos = { int(sample_id): pos
                        for pos, sample_id in enumerate(idx_val_lr)
                     }

    
    if (cfg.SECTION_DISPLAY_ON):
            section_display_title( "Fase2:Analisi XAI -", 
                                    "Comparazione tra Grad-CAM, Integrated Gradients , Occlusion Map Lime Shap sulla griglia 28x28."
                                   )

    GROUP_SETS_BY_MODE = { "all": GROUPS_RESULT_OK_KO,
                            "both_ok": GROUPS_RESULT_BOTH_OK,
                            "both_ko": GROUPS_RESULT_BOTH_KO,
                            "lr_ok_cnn_ko": GROUPS_RESULT_LR_OK_CNN_KO,
                            "lr_ko_cnn_ok": GROUPS_RESULT_LR_KO_CNN_OK,
                         }
    
    selected_mode = getattr(cfg, "ANALYSIS_GROUP_SELECTION", "all")
    if selected_mode not in GROUP_SETS_BY_MODE:
        raise ValueError(f"Valore non valido per cfg.SAMPLE_GROUP_SELECTION: {selected_mode!r}. "
                            f"Valori ammessi: {list(GROUP_SETS_BY_MODE.keys())}"
                        )
    
    active_groups = GROUP_SETS_BY_MODE[selected_mode]
    logger.debug("Selezione gruppi XAI attiva: mode=%s -> %s", selected_mode, list(active_groups.keys()))
    
    for group in active_groups:
        logger.info(f"start {group}")     
        _json_path = out_path / f"stage2_maps_{group}.json"  
               
        plot_metrics_all(filename=out_path / "xai_metrics_all_2.csv",
                    save_path=out_path / f"xai_cnn_deletion_summary_{group}.png")
                             
        summary_df, masks = plot_mask_sensitivity( json_path=_json_path,
                                                            X_data=X_val_lr,
                                                            cfg=cfg,
                                                            save_path=out_path / f"mask_sensitivity_{group}.png",
                                                        )
    
        print(summary_df.to_string(index=False))
    
        # check json map file 
        is_valid = validate_maps_json_positions(_json_path, dataset,idx_to_pos)
        assert(is_valid == True,f"Attenzione!! json file {_json_path} non valido o non coerente")
        
        # Substitude image_index with a index presente into the JSON and csv
        # static selected img after reading json file
        img_idx=56664
        flag_single_img = False
        if flag_single_img:
    
            plot_masks_for_image(json_path=_json_path,
                            image_index=img_idx,
                            idx_to_pos_lr=idx_to_pos,
                            X_data=X_val_lr,
                            cfg=cfg,
                            masks=masks,
                            thresholds=(0.1, 0.2),
                            dilations=(0, 1),
                            save_path=out_path / f"mask_examples_{img_idx}.png",
                        )
                        
        
        
        
        # dinamic selected img from current group
        selected = select_thin_thick_from_maps_json(json_path=_json_path,
                                                    X_data=X_val_lr,
                                                    cfg=cfg)
                    
    
        # iterate a dict element
        for digit , examples in (selected.items()):
            # iterate a thickness element
            for thickness in ("thin", "thick"):
                example = examples[thickness]
                # ID JSON record , not pos X_data
                example_idx = example["index"]
                
                plot_masks_for_image(json_path=_json_path,
                                image_index=example_idx,  
                                X_data=X_val_lr,
                                cfg=cfg,
                                masks=masks,
                                thresholds=(0.1, 0.2),
                                dilations=(0, 1),
                                save_path=out_path / f"mask_{group}_digit{digit}_{thickness}_{example_idx}.png"
                            )

                                   
def phase_3_xai(cfg,
                 logger,
                 inp_path,
                 out_path):
                 
    if (cfg.SECTION_DISPLAY_ON):
        section_display_title( "Fase3:Processo Finale XAI - Modello da Training LR", 
                     "Confronto tra metriche Xai / LR."
                        )
        section_display_title( "Fase3:Comparazione Metriche - Immagini da Quadrante", 
                     "Confronto metriche Xai"
                        )
        results_txt = read_text_for_html(inp_path / cfg.FILE_TXT_RESULTS_XAI)
        section_display_content("Fase3:Confronto Metriche Xai", content_html=results_txt)
        
        compare_txt = read_text_for_html(inp_path / cfg.FILE_TXT_COMPARE_XAI)
        section_display_content("Fase3:Confronto Metriche Xai", content_html=compare_txt)
        
                 
def phase_2_xai( cfg,
                 logger,
                 inp_path,
                 out_path,
                 flag_pretrained = True ) -> None :
    """
         This function runs phase_2 as two steps
         1 - explore and debug images with Xai ( not valid for decision)
         2 - compare images with Xai (valid for decision)
    """
    # EXPLORE AND COMPARE
    if (cfg.SECTION_DISPLAY_ON):
        section_display_title( "Fase2:Esplorazione Tecniche XAI - Immagini da Training LR", 
                     "Confronto visivo tra Grad-CAM, Integrated Gradients , Occlusion Map Lime Shap sulla griglia 28x28."
                        )
        section_display_title( "Fase2:Comparazione Tecniche XAI - Immagini da Quadrante", 
                     "Confronto metriche Xai"
                        )
        content_txt = read_text_for_html(inp_path / cfg.FILE_TXT_CONTENTS_XAI)
        section_display_content("Fase2:Metriche Xai", content_html=content_txt)
   
    x_data , y_data , idx_val = None , None , None
                    
    if cfg.FLAG_FINAL_REPORT:
        x_data,y_data,idx_val = compare_2_xai(cfg,logger,out_path,flag_pretrained)
    
    if cfg.FLAG_INSPECT_REPORT:
        selected_images , position_images = inspect_2_xai( cfg, x_data,y_data,idx_val)    
       
    print(f"selected_img {type(selected_images)}") 
    print(f"selected_img {len(selected_images)}") 
    print(f"selected_img {type(position_images)}") 
    print(f"selected_img {len(position_images)}") 
    
    print(f"selected_img {type(idx_val)}") 
    print(f"selected_img {len(idx_val)}") 
    # print(f"selected_img {idx_val}")
    # raise ValueError("Mi fermo per debug")
    
    if cfg.FLAG_ANALYSIS_REPORT:
        analysis_2_xai(cfg,logger,out_path,x_data,idx_val)
    
    

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

    # run reproducibility check
    repro_results = lr.check_reproducibility(X_train_lr, y_train_lr, cfg)
    if not all(repro_results.values()):
        failed_checks = [key for key, value in repro_results.items() if not value]
        raise RuntimeError(f"Attenzione: test di riproducibilità fallito! {failed_checks}")
        
    logger.debug("Test di riproducibilità superato con successo!")
    
    y_pred_val_lr = lr.predict(X_val_lr)
    val_accuracy_lr = accuracy(y_val_lr,y_pred_val_lr)

    y_pred_test_lr = lr.predict(X_test_lr)
    test_accuracy_lr = accuracy(y_test_lr, y_pred_test_lr)

    logger.debug(f"Val Accuracy (Logistic Regression): {val_accuracy_lr}")
    logger.debug(f"Test Accuracy (Logistic Regression): {test_accuracy_lr}")

    if ( cfg.PLOT_DEBUG_ON ):
        plot_predict_info(y_test_lr, 
                            y_pred_test_lr,
                            "CLASSIFIER LR ON MNIST DATASET " , 
                            "TEST",
                            CONFIG,
                            "lr")

    create_and_save_images_lists( logger, 
                                y_val_lr, 
                                y_pred_val_lr, 
                                idx_val_lr,
                                cfg.CHECK_CLASSES, 
                                out_path,
                                cfg.FILE_IMG_LR_OK, 
                                cfg.FILE_IMG_LR_KO,
                                n_images=cfg.NUM_IMAGES_FOR_CLASS,
                                flag_before_ko=True)
    
    
    list_img = []
    if cfg.LIST_IMAGES_SEL_PHASE_1 == cfg.LIST_IMAGES_SEL_LR_OK:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_OK,n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS)
    elif cfg.LIST_IMAGES_SEL_PHASE_1 == cfg.LIST_IMAGES_SEL_LR_KO:
        list_img = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_KO,n_images_for_class = cfg.NUM_IMAGES_LOAD_FOR_CLASS)

    if len(list_img) == 0 :
        raise ValueError("list_img non può essere vuota")

    callback = None
    if ( cfg.PLOT_DEBUG_ON ):
        callback = make_gradcam_callback(cfg,
                                         val_loader_cnn, 
                                         device, 
                                         out_path , 
                                         list_img , 
                                         logger)
                                    
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
    
       

    # alwais with test data
    if ( cfg.PLOT_DEBUG_ON ):
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
        logger.debug("Training avviato in modalità: LOAD")
    else:
        logger.debug("Training avviato in modalità: TRAIN")

    # Gestione della fase da eseguire
    flag_phase_to_run = args.phase
    
    if flag_phase_to_run == cfg.RUN_PHASE:
        logger.debug("Fase da eseguire: TUTTE LE FASI (Default - 9)")
    elif flag_phase_to_run in (1, 2, 3):
        logger.debug("Fase da eseguire: %d",flag_phase_to_run)
    else:
        raise ValueError(f"Attenzione!! Numero fase da eseguire non valido: {flag_phase_to_run}")

    # dir and file input/output
    inp = Path(cfg.INPUT_DIR) 
    inp.mkdir(parents=True, exist_ok=True)
    
    out = Path(cfg.OUTPUT_DIR) 
    out.mkdir(parents=True, exist_ok=True)

 

    logger.debug("Start Check Fase 1")
    # Esecuzione condizionale o sequenziale basata sulla fase
    if flag_phase_to_run in ( cfg.RUN_PHASE , cfg.RUN_PHASE_1 ):
        if (cfg.SECTION_DISPLAY_ON):
            section_display_title( "Fase2:Training CNN - Eseguire solo con GPU", 
                                   "Debug GradCam , Metriche , Matrice di Confusione , Grafico a Barre F1_SCORE per Classi")
                                   
        logger.debug("Before Run Fase 1")                       
        phase_1_train(cfg, logger, out, flag_pretrained)
        logger.debug("After Run Fase 1")   
      
    logger.debug("Start Check Fase 2")  
    if flag_phase_to_run in ( cfg.RUN_PHASE , cfg.RUN_PHASE_2 ):
        phase_2_xai(cfg, logger, inp, out)

    logger.debug("Start Check Fase 3")
    if flag_phase_to_run in ( cfg.RUN_PHASE , cfg.RUN_PHASE_3 ):
        logger.debug("Before Run Fase 3") 
        phase_3_xai(cfg, logger, inp , out)
        logger.debug("Before Run Fase 3") 
        
    logger.debug("Programma Terminato")

if __name__ == "__main__":
    main()
