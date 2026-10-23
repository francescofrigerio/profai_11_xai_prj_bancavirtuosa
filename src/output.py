
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
# from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

    # img = img / 2 + 0.5     # unnormalize
    # print(f"type {type(img)} ")
    # print(f"img {img.shape} ")
    # img = (img * std) + mean # unnormalize
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # Se il tensore è su GPU, lo portiamo su CPU prima di convertirlo in NumPy
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    # 1. Portiamo i canali alla fine: (C, H, W) -> (H, W, C)
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


# Funzione per visualizzare la matrice di confusione
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

# Funzione per visualizzare la matrice di confusione
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

def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"  figure saved: {path}")
    plt.show() if CONFIG.IN_NOTEBOOK else plt.close(fig)




def plot_logistic_graph(model_,config , out_dir , c ):
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
        ax.imshow(rel[k].reshape(config.IMAGE_SIZE, config.IMAGE_SIZE), cmap="seismic", vmin=-a, vmax=a)
        ax.set_title(f"class {k}")
        ax.axis("off")

    fig.suptitle(f"LR global maps (relative coefficients), C={c}")
    save_fig(fig, out_dir_path / f"phase0_lr_global_maps.png")
