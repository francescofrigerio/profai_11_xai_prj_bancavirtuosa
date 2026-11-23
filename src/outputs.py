"""
    Plot graph on main
"""
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display , HTML
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from IPython.display import Image, display , HTML

def section_display_title(title, description=""):
    html = f"""<div style="margin-top: 35px; margin-bottom: 15px; padding: 12px 16px; 
                border-left: 5px solid #1a73e8; background-color: #f1f3f4; 
                border-radius: 4px; color: #202124;">
                <h3 style="margin: 0 0 5px 0; font-family: sans-serif;">{title}</h3>
                <p style="margin: 0; font-size: 14px; color: #5f6368;">{description}</p>
            </div>
            """
    display(HTML(html))

def section_display_content(title, content_html=""):
    """
         view html content txt 
    """
    html = f"""
            <style>
                .xai-box table 
                {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    font-size: 14px;
                    background-color: #ffffff;
                }}
                .xai-box th, .xai-box td 
                {{
                    border: 1px solid #dadce0;
                    padding: 8px 12px;
                    text-align: left;
                }}
                .xai-box th 
                {{
                    background-color: #e8f0fe;
                    color: #1967d2;
                    font-weight: bold;
                }}
                .xai-box code 
                {{
                    background-color: #f1f3f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: monospace;
                    color: #d93025;
                }}     
            </style>

            <div class="xai-box" style="margin-top: 30px; margin-bottom: 20px; padding: 18px 22px; 
                border-left: 5px solid #1a73e8; background-color: #f8f9fa; 
                border-radius: 6px; color: #202124; font-family: sans-serif; line-height: 1.5;">
                <h3 style="margin: 0 0 12px 0; color: #1a73e8;">{title}</h3>
                <div style="font-size: 14.5px; color: #3c4043;">
                    {content_html}
                </div>
            </div>
    """
    display(HTML(html))





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

def plot_quadrants_info(path , counts_info , logger):
    """
        quadrants plot graph
    """
    names = list(counts_info)
    candidates = [counts_info[n]["candidates"] for n in names]
    selected = [counts_info[n]["selected"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width/2, candidates, width, label="Immagini disponibili")
    ax.bar(x + width/2, selected, width, label="Immagini selezionate")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Numero immagini")
    ax.set_title("Distribuzione delle immagini nei quattro quadranti")
    ax.legend()

    try:
        fig.tight_layout()
        # fig.savefig(out_path / "quadrants_distribution.png", dpi=150)
        fig.savefig(path, dpi=150)
        try:
            display(Image(str(path)))
        except Exception:
            logger.exception("Impossibile mostrare %s", path)
    finally:
        fig.clear()

def plot_quadrants_balance_info(path, counts_info, logger):
    """
        Two-panel chart:
        - Left: Available vs. Selected (logarithmic scale to allow for reading both)
        - Right: Focus exclusively on the selected images to verify the balance across quadrants

    """
    names = list(counts_info)
    candidates = [counts_info[n]["candidati"] for n in names]
    selected = [counts_info[n]["selezionati"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: general comparison with logaritmic scale
    b1 = ax1.bar(x - width / 2, candidates, width, label="Immagini disponibili")
    b2 = ax1.bar(x + width / 2, selected, width, label="Immagini selezionate")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylabel("Numero immagini (scala log)")
    ax1.set_title("Disponibili vs Selezionate")
    ax1.legend()
    ax1.bar_label(b1, padding=2, fontsize=8)
    ax1.bar_label(b2, padding=2, fontsize=8)

    # Panel 2: Zoom on only selected images (effective balance)
    b3 = ax2.bar(x, selected, width=0.45, color="#ff7f0e", label="Immagini selezionate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel("Numero immagini selezionate")
    ax2.set_title("Bilanciamento immagini selezionate per quadrante")
    ax2.set_ylim(0, max(selected) * 1.2 if selected else 10)
    ax2.bar_label(b3, padding=2, fontsize=9, fontweight="bold")

    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        try:
            display(Image(str(path)))
        except Exception:
            logger.exception("Impossibile mostrare %s", path)
    finally:
        fig.clear()

