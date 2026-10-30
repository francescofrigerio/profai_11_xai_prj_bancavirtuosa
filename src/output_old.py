
import numpy as np
import pandas as pd
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
# from PIL import Image
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def save_dict_to_json(file_images , data_dict):
    with open(file_images, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)

def load_dict_from_json(logger,file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        list_load_dict = json.load(f)

    logger.info(f"Caricati {len(list_load_dict)} elementi con successo!")
    logger.info("Primo elemento:", list_load_dict[0])
    return list_load_dict


def save_fig(fig, cfg, path: Path) -> None:
    """
         Colab (IN_NOTEBOOK=True)	
         Codespace (IN_NOTEBOOK=False)
         return None to avoid print(Figure size)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    # print(f"Figure saved: {path}")

    if cfg.IN_NOTEBOOK:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return None

def imshow(img):
    """
        Trasformazione inversa dell'immagine nello spazio dei colori.
        Faccio una trasposta nello spazio dei colori altrimenti
        l'immagine non avrà il colore corretto.
        la normalizzazione è diversa da quella sul dataset
        ma influisce solo sulla visualizzazione
    """

    # Definizione dei valori ImageNet per i 3 canali RGB
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Se il tensore è su GPU, lo portiamo su CPU prima di convertirlo in NumPy
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    # Portiamo i canali alla fine: (C, H, W) -> (H, W, C)
    # Esempio: da (3, 128, 128) diventa (128, 128, 3)
    npimg = np.transpose(img, (1, 2, 0))
    # 2. Denormalizzazione corretta: x = (y * std) + mean
    # NumPy applica automaticamente l'operazione su ogni canale lungo l'ultimo asse
    npimg = (npimg * std) + mean
    plt.imshow(npimg)

    plt.show()

def back_to_image(img):
    # Definizione dei valori ImageNet per i 3 canali RGB
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Se il tensore è su GPU, lo portiamo su CPU prima di convertirlo in NumPy
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    # 1. Portiamo i canali alla fine: (C, H, W) -> (H, W, C)
    # Esempio: da (3, 128, 128) diventa (128, 128, 3)
    npimg = np.transpose(img, (1, 2, 0))

    # 2. Denormalizzazione corretta: x = (y * std) + mean
    # NumPy applica automaticamente l'operazione su ogni canale lungo l'ultimo asse
    npimg = (npimg * std) + mean

    # 3. Clip per evitare che imprecisioni nei float uscano dal range valido [0, 1] per matplotlib
    npimg = np.clip(npimg, 0, 1)

    return npimg

def plot_predict_info(y_true_,
                      y_pred_, 
                      title , 
                      str_train_test,
                      cfg,
                      suffix_file):
    """
         Plot Confusion Matrix Table
         Classification Report
         BarPlot F1_score
         suffix_file -> lr or cnn
    """

    

    # confusion_matrix
    cm = confusion_matrix(y_true_,y_pred_)

    fig, ax = plt.subplots(figsize=(7, 5))
    # ax=ax associates heatmap to axis
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False , ax=ax)
    ax.set_xlabel('Predicted ' + str_train_test)
    ax.set_ylabel('True ' + str_train_test)
    ax.set_title(title + str_train_test)
    
    save_fig(fig,cfg,Path(cfg.OUTPUT_DIR) / f"confusion_matrix_{suffix_file}_{str_train_test}.png")

    # Classification report
    class_labels = list(range(cfg.NUM_CLASSES))

    report = classification_report( y_true_,
                                    y_pred_,
                                    labels=class_labels,
                                    target_names=[str(i) for i in class_labels],
                                    output_dict=True,
                                    zero_division=0
                                )

    df_report = pd.DataFrame(report).T
    print(df_report.round(4))

    # Extraction f1 scores from report
    f1_scores = [ report[str(i)]["f1-score"]
                    for i in class_labels
                ]

    # Order class for F1-score ascending
    sorted_indices = np.argsort(f1_scores)

    class_labels_sorted = [str(class_labels[i]) for i in sorted_indices]
    f1_scores_sorted = [f1_scores[i] for i in sorted_indices]


    # plot bar
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = plt.bar(class_labels_sorted,
                    f1_scores_sorted,
                    color="steelblue",
                    edgecolor="black"
                )

    ax.set_xlabel("MNIST Class (digit)")
    ax.set_ylabel("F1-score")
    ax.set_title(f"F1-score for class - {title} {str_train_test}")
    ax.set_xticks(class_labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Numeric Value upon the bars
    ax.bar_label(bars, fmt="%.3f", padding=3)
    save_fig(fig,cfg,Path(cfg.OUTPUT_DIR) / f"f1_score_{suffix_file}_{str_train_test}.png")

def plot_confusion_matrix_lr(model,X,y_true, title , str_train_test,cfg):
    """
         Plot Confusion Matrix Table
    """

    y_pred =  model.predict(X)
    cm = confusion_matrix(y_true,y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted ' + str_train_test)
    plt.ylabel('True ' + str_train_test)
    plt.title(title + str_train_test)

    report = classification_report(y_true,y_pred, target_names=[str(i) for i in range(cfg.NUM_CLASSES)],digits=4)
    print(report)

def plot_predict_info_cnn(y_true_,y_pred_, title , str_train_test,cfg):
    """
        Plot Confusion Matrix Table
        Classification Report
        BarPlot F1_score
    """

    cm = confusion_matrix(y_true_, y_pred_)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    ax.set_xlabel('Predicted ' + str_train_test)
    ax.set_ylabel('True ' + str_train_test)
    ax.set_title(title + str_train_test)
        
    save_fig(fig,cfg,Path(cfg.OUTPUT_DIR) / f"confusion_matrix_cnn_{str_train_test}.png")
    

    # Classification report
    class_labels = list(range(cfg.NUM_CLASSES))
    
    report = classification_report( y_true_,
                                    y_pred_,
                                    labels=class_labels,
                                    target_names=[str(i) for i in class_labels],
                                    output_dict=True,
                                    zero_division=0
                                    )
    
    df_report = pd.DataFrame(report).T
    print(df_report.round(4))

    # Extraction f1 scores from report
    f1_scores = [ report[str(i)]["f1-score"]
                        for i in class_labels
                    ]
    
    # plot bar
    fig, ax = plt.subplots(figsize=(9, 5))
    
    bars = plt.bar(class_labels,
                        f1_scores,
                        color="steelblue",
                        edgecolor="black"
                    )
    
    ax.set_xlabel("MNIST Class (digit)")
    ax.set_ylabel("F1-score")
    ax.set_title(f"F1-score for class - {title} {str_train_test}")
    ax.set_xticks(class_labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Numeric Value upon the bars
    ax.bar_label(bars, fmt="%.3f", padding=3)
    save_fig(fig,cfg,Path(cfg.OUTPUT_DIR) / f"f1_score_cnn_{str_train_test}.png")

def plot_confusion_matrix_cnn(y_true_,y_pred_, title , str_train_test,cfg):
    """
         Plot Confusion Matrix Table
    """

    cm = confusion_matrix(y_true_, y_pred_)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted ' + str_train_test)
    plt.ylabel('True ' + str_train_test)
    plt.title(title + str_train_test)

    report = classification_report(y_true_,y_pred_, target_names=[str(i) for i in range(cfg.NUM_CLASSES)],digits=4)
    print(report)

def plot_logistic_graph(model_,cfg , out_dir , c ):
    out_dir_path = Path(out_dir)  # Converte la stringa in Path

    # The matrix model_.coef_ has the next shape [10 ,784]:
    # so 10 classes = digits and 784 pixels for every classes
    # Every element is a coefficient w  learned from model.
    # The model use these coefficients to get the score for each classes.

    # È la differenza tra il coefficiente di ciascuna classe
    # e la media dei coefficienti dello stesso pixel calcolata su tutte le classi.
    # Per mettere in evidenza ciò che distingue una classe dalle altre,
    # anziché mostrare semplicemente il coefficiente originale.
    # Un valore positivo indica un coefficiente superiore alla media delle classi per quel pixel;
    # un valore negativo indica un coefficiente inferiore.
    rel = model_.coef_ - model_.coef_.mean(axis=0, keepdims=True)

    # 2d-matrix with single graph
    # axes[0,0], axes[0,1], ..., axes[0,4],
    # axes[1,0], axes[1,1], ..., axes[1,4]
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))

    # axes.ravel() convert 2d-matrix in a 1d-array , reading
    # elements from left to right from top to bottom .
    # loop1: k = 0 e ax is the first graph at top left side
    # loop2: k = 1 e ax is the second graph and so on .
    for k, ax in enumerate(axes.ravel()):

        # rel[k] seleziona tutti i coefficienti relativi della classe k: nel nostro esempio, 784 valori=pixels.
        # np.abs(rel[k]) calcola il valore assoluto di ciascun coefficiente. Per esempio, -0.7 diventa 0.7.
        # .max() trova il valore assoluto più grande tra tutti i pixel di quella classe.
        a = max(np.abs(rel[k]).max(), 1e-12)

        # Con reshape i pixel da 1d-array sono disposti in a 2d matrix,
        # che può essere rappresentata come un'immagine.
        ax.imshow(rel[k].reshape(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), cmap="seismic", vmin=-a, vmax=a)
        ax.set_title(f"class {k}")
        ax.axis("off")

    fig.suptitle(f"LR global maps (relative coefficients), C={c}")
    save_fig(fig,cfg,Path(cfg.OUTPUT_DIR) / "phase0_lr_global_maps.png")
    # save_fig(fig, cfg, out_dir_path)
