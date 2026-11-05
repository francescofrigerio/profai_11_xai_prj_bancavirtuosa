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

import numpy as np

from baseline import fit_baseline
from config import CONFIG , CONFIG_CHECK_CLASSES
from data import data_to_input_lr, load_datasets
from data import save_dict_to_json 
from logger import LOGGER
from metrics import accuracy, confusion, macro_f1, quadrants
from output import plot_logistic_graph , plot_predict_info 
from training import get_device, get_trained_model ,get_trained_model_callback
from training import get_evaluate_model
from xai import TECHNIQUE_NAMES, build_techniques

from utility import run_create_list_images , run_check_class_ok_ko , ls_dir
from xai.gradcam import make_gradcam_debug_callback
from xai.shap_explainer import make_background

# Configurazione dei Warnings (spostata DOPO tutti gli import) ---
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")
warnings.filterwarnings('ignore', category=FutureWarning)

def phase_2_old( cfg,
             cfg_list_class,
             logger,
             out_path ) -> None :
    
    
    out = Path(cfg.OUTPUT_DIR) / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    
    t_start = time.time()

    device, warning = pick_device(cfg)
    if warning:
        print("WARNING:", warning)
        
    data = load_mnist(cfg)
    split = data.split
    callback = make_gradcam_debug_callback(data, cfg, device, out_path)
    model, info = get_trained_model(cfg, data, device, epoch_callback=callback)
    
    background = make_background(data.X_trainval, split.idx_train, cfg, device)
    techniques = build_techniques(model, cfg, background)

    y = data.y_trainval.numpy()
    X_flat = to_lr_input(data.X_trainval)
    lr = fit_baseline(X_flat, y, split.idx_train, cfg)
    picks = debug_indices(y, split.idx_val, cfg.NUM_DEBUG_IMAGES)

    rows, seconds, ig_errors = [], {n: [] for n in techniques}, []
    for i in picks:
        x1 = to_densenet_input(data.X_trainval[i:i + 1], cfg.DENSENET_SIZE).to(device)
        with torch.no_grad():
            pred = int(model(x1).argmax(1))
        maps = {}
        for name, technique in techniques.items():
            result = technique.explain(x1, pred)
            maps[name] = result
            seconds[name].append(result.seconds)
            if name == "integrated_gradients":
                ig_errors.append(result.meta["completeness_error"])
        rows.append({"index": int(i), "true": int(y[i]), "pred": pred, "maps": maps,
                     "lr_map": lr.local_map(X_flat[i], pred)})
        print(f"image {i} (true {y[i]}, predicted {pred}): " +
              ", ".join(f"{n} {maps[n].seconds:.2f}s" for n in maps))

    first = torch.as_tensor(to_densenet_input(data.X_trainval[picks[0]:picks[0] + 1], cfg.DENSENET_SIZE)).to(device)
    for name in ("lime", "shap"):
        again = techniques[name].explain(first, rows[0]["pred"]).map
        ref = rows[0]["maps"][name].map
        v.expect(f"{name} is reproducible", bool(np.allclose(ref, again, rtol=1e-3, atol=1e-4 * np.abs(ref).max())),
                 "same seed, same map on the first image")
   
    patched = SafeDenseNet(copy.deepcopy(model)).eval()
    for name in ("gradcam", "integrated_gradients", "saliency"):
        a = rows[0]["maps"][name].map
        b = build_techniques(patched, cfg, names=(name,))[name].explain(first, rows[0]["pred"]).map
        diff = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))
        
    shares = []
    for row in rows:
        mask = digit_mask(data.X_trainval[row["index"], 0].numpy(), cfg.MASK_THRESHOLD, cfg.MASK_DILATION)
        g = to_grid28(np.abs(row["maps"]["saliency"].map), cfg.DENSENET_SIZE)
        shares.append((float(g[mask].sum() / g.sum()), float(mask.mean())))
    
    # ---- figure
    names = list(techniques)
    fig = Figure(figsize=(2.1 * (len(names) + 2), 2.1 * len(rows)))
    axes = fig.subplots(len(rows), len(names) + 2)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(data.X_trainval[row["index"], 0], cmap="gray")
        axes[r, 0].set_ylabel(f"true {row['true']}\npred {row['pred']}", fontsize=8)
        cells = [(row["maps"][n].map, row["maps"][n].signed) for n in names] + [(row["lr_map"], True)]
        for c, (m, signed) in enumerate(cells, start=1):
            a = float(np.abs(m).max()) or 1.0
            axes[r, c].imshow(m, cmap="seismic" if signed else "viridis", vmin=-a if signed else 0, vmax=a,
                              interpolation="nearest" if m.shape[0] <= 8 else "bilinear")
        for c in range(len(names) + 2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, title in enumerate(["image"] + [TITLES[n] for n in names] + ["LR local\n(w * x)"]):
        axes[0, c].set_title(title, fontsize=9)
    fig.savefig(out / "maps_grid.png", dpi=90, bbox_inches="tight")
    try:
        from IPython.display import Image, display
        display(Image(str(out / "maps_grid.png")))
    except Exception:
        pass

    results = {
        "model": {k: info[k] for k in ("action", "reason", "sha256", "device")},
        "median_seconds_per_image": median, "ig_completeness_errors": ig_errors,
        "images": [{"index": r["index"], "true": r["true"], "pred": r["pred"],
                    "native_shapes": {n: list(r["maps"][n].map.shape) for n in names}} for r in rows],
        "per_image_all_techniques_seconds": float(sum(median.values())), "total_seconds": time.time() - t_start,
    }
    (out / "stage2_results.json").write_text(json.dumps(results, indent=2))
    (out / "verifications.md").write_text(v.to_markdown() + "\n")
    print("\n" + v.to_markdown())
    print(f"\nmedian seconds per image: " + ", ".join(f"{n} {m:.2f}" for n, m in median.items()))
    print(f"all six techniques: {sum(median.values()):.1f}s per image | worst status: {v.worst()} | total {time.time() - t_start:.0f}s")
    return results, v

def phase_2( cfg,
             cfg_list_class,
             logger,
             out_path ) -> None :
    """
        Run xai Explainer
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

    callback_lr_ok = make_gradcam_debug_callback(cfg,logger,test_datasets, device, out_path , cfg.FILE_IMG_LR_OK)
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
                                    epoch_callback=callback_lr_ok )

    background = make_background(trainval_dataset,train_idx, cfg, device)
    techniques = build_techniques(model, cfg, background)

    # Load tensor for train and test baseline
    train_loader_lr = loaders_dict["lr"]["train"]
    val_loader_lr = loaders_dict["lr"]["val"]
          
    X_train , y_train , _  = data_to_input_lr(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val , _ = data_to_input_lr(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
                        
    # Logistic regression (train indices only)
    lr = fit_baseline(X_train, y_train, X_val, y_val,cfg,logger)
    

    
    
   
    
def phase_1_2( cfg,
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
    

def phase_1_1(cfg,
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
        phase_1_1(cfg,logger,out,flag_pretrained)
    else:
        phase_1_2(cfg,cfg_list_class,logger,out,flag_pretrained)

    phase_2(cfg,cfg_list_class,logger,out)

    logger.debug("Programma Terminato")

if __name__ == "__main__":
    main()
