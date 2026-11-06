"""
    Run From the final notebook).

    Real run on Colab: MNIST -> DenseNet (train or load, per MODEL_MODE) 
                         -> LR baseline -> metrics ->
    %run run_main.py
""" 
import argparse
import os
import json
from pathlib import Path
import time
import warnings
import copy
import numpy as np
import torch
import torch.nn.functional as F

from baseline import fit_baseline
from config import CONFIG , CONFIG_CHECK_CLASSES , TITLES
from data import data_to_input_lr, load_datasets
from data import save_dict_to_json , load_images_by_class
from data import to_densenet_input

from logger import LOGGER
from metrics import accuracy, confusion, macro_f1, quadrants
from output import plot_logistic_graph , plot_predict_info 
from training import get_device, get_trained_model ,get_trained_model_callback
from training import get_evaluate_model
from xai import TECHNIQUE_NAMES, build_techniques

from utility import run_create_list_images , run_check_class_ok_ko , ls_dir
from xai.gradcam import make_gradcam_debug_callback
from xai.shap_explainer import make_background

from matplotlib.figure import Figure

# Configurazione dei Warnings (spostata DOPO tutti gli import) ---
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def digit_mask(input_img28, threshold, dilation):
    """
        This function creates a binary mask of the image.

        It compares each pixel with a threshold:
        - pixel above the threshold: True;
        - pixel at or below the threshold: False.
        In this way, it approximately identifies 
        the pixels belonging to the digit,
        and so identify the area occupied by the digit

        In the short:
        This function is used to distinguish the digit area 
        from the background, potentially including 
        a small margin around the strokes(tratti).

        Limits:THRESHOLD` must be consistent 
        with the pixel values ​​in X_train. 
        If the images are normalized or standardized, 
        the threshold must be chosen accordingly.
    """
    from scipy.ndimage import binary_dilation
    mask = np.asarray(input_img28) > threshold

    # expands the mask, slightly enlarging the identified area.
    # The dilation parameter determines how many times 
    # this expansion is performed. If it is set to zero, 
    # the mask remains unchanged.
    return binary_dilation(mask, 
                           structure=np.ones((3, 3), bool), 
                           iterations=dilation
                           ) if dilation else mask


def to_grid28(map2d, size, side=28):
    """
        This Function convert XAI 2dmap 
        to mnist risolution 28x28 to compare
        xai map with the digit mask

        This is the step:
    """

    # - Convert map2d to tensor and Adds two dimensions—batch 
    # and channel—resulting 
    # in the shape (1, 1, H, W).
    t = torch.as_tensor(np.asarray(map2d), 
                        dtype=torch.float32)[None, None]
    if t.shape[-1] < size:
        # If the map is smaller than the specified size, 
        # it scales it up to size × size.
        t = F.interpolate(t, 
                          size=(size, size), 
                          mode="bilinear", 
                          align_corners=False)

    # Resizes it to 28 × 28 using area interpolation.
    # and removes the batch and channel dimensions 
    # and returns a NumPy array.  
    return F.interpolate(t, 
                         size=(side, side), 
                         mode="area")[0, 0].numpy()


def phase_2_xai( cfg,
                 logger,
                 dataset,
                 t_start,
                 techniques,
                 model,
                 lr,
                 X_train,
                 y_train,
                 device,
                 info,
                 out_path,
                 list_images ) -> None :
    """
    This function implements Hook that are a tool 
    for observing the internal 
    workings of a network during processing.
    
    A hook is a function that allows you to attach 
    to a specific point during the execution 
    of a PyTorch model, enabling you to intercept 
    and observe information—such as during the 
    forward or backward pass—without 
    having to directly modify the model's code.

    In the context of XAI (Explainable AI), 
    hooks are primarily used to retrieve layer 
    activations and their corresponding gradients, 
    which are necessary for calculating explanations.
    """
    
    rows, ig_errors = [], []
    n_images = len(list_images)

    # for i in picks:
    for i, id_class in enumerate(sorted(list_images.keys())):

        item = list_images[id_class]
        idx_dataset = item["id_dataset"]

        logger.info(f"Start Loop {i} {idx_dataset}")

        # The function to_densenet_input() has to receive a batch with 4 dim. 
        # So we add the size batch here 
        # Original Image MNIST: (1, 28, 28)
        input_image, input_label = dataset[idx_dataset]
                       
        # Add size of the batch: (1, 1, 28, 28) to the input
        x28 = input_image.unsqueeze(0)
        
        # Convert to Input DenseNet: (1, 3, 128, 128)
        # x1 = to_densenet_input(data.X_trainval[i:i + 1], cfg.DENSENET_SIZE).to(device)
        input_img = to_densenet_input(x28,cfg.DENSENET_SIZE).to(device)
        
        with torch.no_grad():
            # pred = int(model(input_img).argmax(1))
            pred = int(model(input_img).argmax(dim=1).item())

        maps = {}
        for name, technique in techniques.items():
            logger.info(f"Start Xai {name} {technique}")
            result = technique.explain(input_img, pred)
            maps[name] = result
            
            if name == "integrated_gradients":
                ig_errors.append(result.meta["completeness_error"])
            logger.info(f"End Xai {name} {technique}")

        # sostituzione i con idx_dataset
        rows.append({"index": int(idx_dataset), 
                     "true": int(y_train[idx_dataset]), 
                     "pred": pred, 
                     "maps": maps,
                     "lr_map": lr.local_map(X_train[idx_dataset], pred)})
        
        logger.info(f"End Loop {i} image {idx_dataset} (true {y_train[idx_dataset]}, predicted {pred}): ")

    # This selects the first index to
    # compare explanation getting from 
    # this first with that already in rows[0] 
    first_idx = rows[0]["index"]
    # These select and transform the first image
    first_image, _ = dataset[first_idx]
    first = to_densenet_input(first_image.unsqueeze(0),
                              cfg.DENSENET_SIZE
                              ).to(device)

    # LIME and SHAP can have stochastic components: 
    # a new execution might produce a slightly different map.                         
    for name in ("lime", "shap"):
        logger.info(f"Start Again Calculate Xai {name} {techniques[name]}")
        # This gets the new calculated map;
        again = techniques[name].explain(first, rows[0]["pred"]).map
        # This gest the previously saved calculated map, 
        ref = rows[0]["maps"][name].map

        diff = float(np.abs(again - ref).max())
        
        logger.info(f"{name} Check Difference {diff <= cfg.RELU_MAX_REF_DIFF}", 
                            f"max difference {diff:.2e}")
        
        logger.info(f"End Again Calculate Xai {name} {technique}")
       
    # This model's independent copy is used to perform checks 
    # without affecting the original trained model
    patched = copy.deepcopy(model).eval()

    # Saliency Map Normalization (Min-Max [0, 1]):
    # in order to make the maps generated 
    # by different techniques
    # homogeneous and directly comparable 
    # (e.g., Grad-CAM and Integrated Gradients
    #     can produce raw attribution values 
    #     on very different scales).

    # Just verify that the recalculation using the copy 
    # of the model produces results sufficiently close 
    # to the original ones.
    # it is a consistency check
    # We can cancel this test to have a more simple code
    for name in ("gradcam", "integrated_gradients", "saliency"):
        # getting the map calculate in the main loop
        a = rows[0]["maps"][name].map
        # getting the map recalculated on the same image, 
        # using the copied model.
        b = build_techniques(patched, cfg, names=(name,))[name].explain(first, rows[0]["pred"]).map
        
        # it measures the maximum absolute difference 
        # between the two maps, normalized by the 
        # maximum absolute value of the second map.
        diff = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
        
        logger.info(f"{name} Check on trained model: "
                    f"{diff <= cfg.RELU_MAX_REL_DIFF}, "
                    f"relative difference {diff:.1e}"
                    )
         
    shares = []

    # this section calculating two values ​​for each image:
    # 1) It represents the proportion 
    # of the total saliency that falls within the digit mask.
    # If first value = 0.85, 
    # it would mean that 85% of the total saliency 
    # under consideration falls within the area 
    # identified as the digit.

    # 2) It represents the fraction of the 
    # image occupied by the mask space digit.
    # shares contains a pair of indicators for each image, 
    # rather than the maps themselves.
    for row in rows:
        mask = digit_mask(X_train[row["index"], 0].numpy(), 
                          cfg.MASK_THRESHOLD, 
                          cfg.MASK_DILATION)
        
        g = to_grid28(np.abs(row["maps"]["saliency"].map), 
                      cfg.DENSENET_SIZE)

        # shares.append((float(g[mask].sum() / g.sum()), 
        #                       float(mask.mean() )))
        share = float(g[mask].sum() / max(g.sum(), 1e-12))
        shares.append(share,float(mask.mean())
        
    
    # figure plot for all techniques
    names = list(techniques)
    fig = Figure(figsize=(2.1 * (len(names) + 2), 2.1 * len(rows)))
    axes = fig.subplots(len(rows), len(names) + 2)
    for r, row in enumerate(rows):
        # original image
        axes[r, 0].imshow(X_train[row["index"], 0], cmap="gray")
        axes[r, 0].set_ylabel(f"true {row['true']}\npred {row['pred']}", fontsize=8)
        # Prepare the cell's maps for visualization:
        # col1 original map
        # col2 for each XAI technique;
        # col3 for the local map of the Logistic Regression
        cells = [(row["maps"][n].map, 
                  row["maps"][n].signed) for n in names] + [(row["lr_map"], True)]

        for c, (m, signed) in enumerate(cells, start=1):
            # Graphical normalization is performed using 
            # the maximum absolute value of each individual map. 
            # This makes the maps easier to read, 
            # but be careful: it does not mean that 
            # the colors can be directly compared 
            # in quantitative terms across different techniques.
            a = float(np.abs(m).max()) or 1.0
            
            # seismic for maps with both positive and negative value
            # viridis for those that are non-negative.
            axes[r, c].imshow(m, cmap="seismic" if signed else "viridis", 
                              vmin=-a if signed else 0, 
                              vmax=a,
                              interpolation="nearest" if m.shape[0] <= 8 else "bilinear")
        for c in range(len(names) + 2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, title in enumerate(["image"] + [TITLES[n] for n in names] + ["LR local\n(w * x)"]):
        axes[0, c].set_title(title, fontsize=9)
    fig.savefig(out_path / "maps_grid.png", 
                dpi=90, bbox_inches="tight")
    try:
        from IPython.display import Image, display
        display(Image(str(out_path / "maps_grid.png")))
    except Exception:
        pass

    # save result to json file
    results = {
        "model": {k: info[k] for k in ("action", "reason", "sha256", "device")},
        "ig_completeness_errors": ig_errors,
        "images": [{"index": r["index"], "true": r["true"], "pred": r["pred"],
                    "native_shapes": {n: list(r["maps"][n].map.shape) for n in names}} for r in rows],
        "total_seconds": time.time() - t_start,
    }
    (out_path / "stage2_results.json").write_text(json.dumps(results, indent=2))
    
    logger.info(f"all tecniques total time {time.time() - t_start:.0f}s")
    return results

def phase_2_main( cfg,
                  cfg_list_class,
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
    if len(cfg_list_class) == 0 :
        raise ValueError("list_class non può essere vuota")
    else:
        n_for_class = max(1, cfg.NUM_IMAGES_FOR_CLASS // len(cfg_list_class))

    t_start = time.time()
    
    logger.debug(f"Immagini per classe: {n_for_class}")
    device, warning = get_device(cfg)
    if warning:
        logger.debug("WARNING:", warning)

    datasets_dict , loaders_dict = load_datasets()
    trainval_dataset = datasets_dict["trainval"]
    test_datasets = datasets_dict["test"]
    train_idx = datasets_dict["train_idx"]

    train_loader_cnn = loaders_dict["cnn"]["train"]
    val_loader_cnn = loaders_dict["cnn"]["val"]
    test_loader_cnn = loaders_dict["cnn"]["test"]

    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
              
    X_train , y_train , _  = data_to_input_lr(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val , _ = data_to_input_lr(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
                            
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)
    
    list_images_sample = load_images_by_class(logger,out_path / cfg.FILE_IMG_LR_OK)
    callback = make_gradcam_debug_callback(cfg,logger,test_datasets, device, out_path , list_images_sample)
    # callbac_lr_ko = make_gradcam_debug_callback(cfg,logger,test_datasets, device, out_path , cfg.FILE_IMG_LR_KO)
    # callback_cnn_ok = make_gradcam_debug_callback(cfg,logger,test_datasets, device, out_path , cfg.FILE_IMG_CNN_OK)
    # callback_cnn_ko = make_gradcam_debug_callback(cfg,logger,test_datasets, device, out_path , cfg.FILE_IMG_CNN_KO)

    model , info = get_trained_model(cfg,
                                    logger, 
                                    test_datasets,
                                    train_loader_cnn,
                                    val_loader_cnn,
                                    test_loader_cnn, 
                                    device,
                                    pretrained=True,    
                                    epoch_callback=callback )

    background = make_background(trainval_dataset,train_idx, cfg, device)
    techniques = build_techniques(model, cfg, background)

    phase_2_xai( cfg, 
                logger, 
                t_start, 
                techniques, 
                model, 
                lr, 
                X_train,
                y_train,
                device, 
                info, 
                out_path , 
                list_images_sample ) 
    

    
    
   
    
def phase_1_load( cfg,
                cfg_list_class,
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

     
    
    
    if len(cfg_list_class) == 0 :
        raise ValueError("list_class non può essere vuota")
    else:
        n_for_class = max(1, cfg.NUM_IMAGES_FOR_CLASS // len(cfg_list_class))

    logger.debug(f"Immagini per classe: {n_for_class}")
    device, warning = get_device(cfg)
    if warning:
        logger.debug("WARNING:", warning)

    # START LOAD DATA
    datasets_dict , loaders_dict = load_datasets()
    
    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
    test_loader_lr = loaders_dict["lr"]["test"]
        
    # Load tensor for train and test cnn
    test_loader_cnn = loaders_dict["cnn"]["test"]
            
    X_train , y_train , _ = data_to_input_lr(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val , _ = data_to_input_lr(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)

    X_test , y_test , idx_dataset = data_to_input_lr(test_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    # END LOAD DATA

    # START PROCESS LR MODEL
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)
    y_pred =  lr.predict(X_test)

    logger.debug("Indici KO/OK di controllo per classe:")

    run_check_class_ok_ko( logger, 
                            y_test,
                            y_pred , 
                            cfg_list_class,
                            "LR")

    list_images_lr_ok , list_images_lr_ko  = run_create_list_images( logger,
                                                                     X_test,
                                                                     y_test,
                                                                     y_pred,
                                                                     idx_dataset,
                                                                     cfg_list_class,
                                                                     n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                                     flag_before_ko=True)
    if (len(list_images_lr_ok) < cfg.NUM_IMAGES_MIN_OK ):
        list_images_lr_ok , _  = run_create_list_images( logger,
                                                        X_test,
                                                        y_test,
                                                        y_pred ,
                                                        idx_dataset,
                                                        cfg_list_class,
                                                        n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                        flag_before_ko=False)

    logger.debug(f"Risultato Finale LR: {len(list_images_lr_ko)} {len(list_images_lr_ok)}")
    # END PROCESS LR MODEL
    
    # START PROCESS CNN MODEL 
    X_test , y_test , y_pred , idx_dataset = get_evaluate_model( cfg,
                                                   logger,
                                                   flag_pretrained,
                                                   test_loader_cnn,
                                                   device )
    
    logger.debug("Indici KO di controllo per classe:")
    
    run_check_class_ok_ko( logger, 
                            y_test,
                            y_pred , 
                            cfg_list_class,
                            "CNN")

    list_images_cnn_ok , list_images_cnn_ko  = run_create_list_images(  logger,
                                                                        X_test,
                                                                        y_test,
                                                                        y_pred,
                                                                        idx_dataset,
                                                                        cfg_list_class,
                                                                        n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                                        flag_before_ko=True)
    if (len(list_images_cnn_ok) < cfg.NUM_IMAGES_MIN_OK ):
            list_images_cnn_ok , _  = run_create_list_images( logger,
                                                              X_test,
                                                              y_test,
                                                              y_pred,
                                                              idx_dataset,
                                                              cfg_list_class,
                                                              n_images_=cfg.NUM_IMAGES_FOR_CLASS,
                                                              flag_before_ko=False)
    logger.debug(f"Risultato Finale CNN: {len(list_images_cnn_ko)} {len(list_images_cnn_ok)}")
    # END PROCESS LR MODEL

    # Save files for phase 3
    save_dict_to_json(out_path / cfg.FILE_IMG_LR_OK , list_images_lr_ok)
    save_dict_to_json(out_path / cfg.FILE_IMG_LR_KO , list_images_lr_ko)
    save_dict_to_json(out_path / cfg.FILE_IMG_CNN_OK , list_images_cnn_ok)
    save_dict_to_json(out_path / cfg.FILE_IMG_CNN_KO , list_images_cnn_ko)
    

def phase_1_train(cfg,
                logger,
                out_path,
                flag_pretrained = False):
    """
        This is the first run.
        - TO begin with loads data , train and test the models lr and cnn.
        - After that show save checkpoint cnn model.
        - At The end show/save different outputs (confusion matrix , classification repors
        plot bar based and order on F1_Score metrics for different classes ) 
    """

    t_start = time.time()
    
    logger.debug(f"Output dir: {cfg.OUTPUT_DIR}")
    
    device, warning = get_device(cfg)
    if warning:
        logger.debug("WARNING:", warning)
    
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
        
    X_train , y_train , _  = data_to_input_lr(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val , _ = data_to_input_lr(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
        
    X_test , y_test , _ = data_to_input_lr(test_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
        
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)
    
    val_accuracy_lr = accuracy(y_val, lr.predict(X_val))
    test_accuracy_lr = accuracy(y_test, lr.predict(X_test))
    print(f"Val Accuracy (Logistic Regression): {val_accuracy_lr}")
    print(f"Test Accuracy (Logistic Regression): {test_accuracy_lr}")

    # bella ma non sembra molto utile alla comprensione
    # plot_logistic_graph(lr.get_model(), CONFIG, CONFIG.OUTPUT_DIR, c=CONFIG.LR_C)
        
    # y_pred_val_lr = lr.predict(X_val)
    y_pred_test_lr = lr.predict(X_test)
    plot_predict_info(y_test, 
                      y_pred_test_lr,
                      "CLASSIFIER LR ON MNIST DATASET " , 
                      "TEST",
                      CONFIG,
                      "lr")

    # False is mandatory To run on codespace 
    flag_run_cnn = cfg.RUN_CNN_MAN
    # on codespace github alwais down
    if cfg.IN_NOTEBOOK is False :
        flag_run_cnn = False 

    if flag_run_cnn is True:
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
    
def main(cfg=CONFIG,cfg_list_class=CONFIG_CHECK_CLASSES,logger=LOGGER):
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
        phase_1_train(cfg,logger,out,flag_pretrained)
    else:
        phase_1_load(cfg,cfg_list_class,logger,out,flag_pretrained)

    phase_2_main(cfg,cfg_list_class,logger,out)

    logger.debug("Programma Terminato")

if __name__ == "__main__":
    main()
