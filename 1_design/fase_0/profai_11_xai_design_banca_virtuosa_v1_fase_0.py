

"""**GIT CLONE DAL REPOSITORY IN PRODUZIONE OPPURE MOUNT DRIVE IN SVILUPPO**"""

# Commented out IPython magic to ensure Python compatibility.
def run_mount_drive(flag_mount=True):
   """
        Mount Drive for load dataset
   """

   if flag_mount :
      from google.colab import drive
      # drive.mount('/gdrive'
      drive.mount('/content/drive')
      #Colab Notebooks\ProfAi\Corso 09 Generative AI\ProjectCyberEyeSolution
      # %cd drive/MyDrive/'Colab Notebooks'/ProfAi/'Corso 11 ExplainedAi'/ProjectCyberEyeSolution/
#       %cd drive/MyDrive/'Colab Notebooks'/ProfAi/'Corso 11 ExplainedAi'/ProjectBancaVirtuosaXai/

   else:
#       %cd ProjectBancaVirtuosaXai/
#    %ls

run_mount_drive(flag_mount=True)

# installazione pacchett
!pip -q install captum

from tqdm.auto import tqdm
import logging

#DEBUG
# 1. Configurazione di base
# Qui decidiamo il livello minimo (WARNING) e il formato del messaggio
# nel codice con setTrace attivo il debug
logging.basicConfig(
    level=logging.WARNING,
    # level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

# Creazione di un oggetto logger
logger = logging.getLogger("AiAppLogger")
logger.setLevel(logging.INFO)


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Librera specifica di pytorch per la computer vision
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
from torchvision.models import DenseNet121_Weights, densenet121

# Libreria specfica di pytorch per effettuare trasformazioni
# su tensori che rappresentano delle immagini.
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
# usata per validare i nostri conti e
# calcolare parametri e dimensioni della rete
# from torchsummary import summary

# Non serve pandas perchè non andiamo a leggere
# il dataset da un csv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import sys
import random


# Aggiunti rispetto al codice originale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Altrimenti con la LogisticRegression
# non si vede il warning relativo alla mancata convergenza dopo max_iter
# import warnings
# warnings.filterwarnings('ignore')
import warnings

# 1. Ignora i warning legati alla deprecazione di argomenti nei modelli
warnings.filterwarnings('ignore', category=UserWarning, message=".*pretrained.*")

# 2. Ignora i FutureWarning di librerie terze (es. Pandas o NumPy interni a Torchvision)
warnings.filterwarnings('ignore', category=FutureWarning)
print("Script avviato con warning filtrati in modo selettivo!")


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


import pandas as pd
from PIL import Image

# Modelli di Base Saliency e Occlusion
from captum.attr import Saliency, Occlusion
# Modelli Avanzati Integrtared Gradience
from captum.attr import IntegratedGradients

"""**1. FUNZIONE DI UTILITA'***

**2. CONFIGURAZIONE E COSTANTI**

**3. RIPRODUCIBILITA'**

**4. CONTROLLO RUNTIME TYPE**

"""

def sync(device) -> None:
    """
        Syncronizer CPU and GPU to get correct train and val time
        Only Running with CUDA=True
        Run this function before get time to make cpu wait mode until GPU
        has finisched his work.
        Run this function before get time to avoid measure time of previous epoch
    """
    if torch is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device):
    """
        Wrapper to get time
    """
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return out, time.perf_counter() - t0

def set_seed(seed: int) -> None:
    """
        fix seed for every library random,environment,numpy,torch
    """
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


from dataclasses import dataclass

@dataclass(frozen=True)
class ConfigConstantsDefault:
    """Global config. Replaces magic numbers with descriptive names."""

    # --- Reproducibility and environment
    SEED: int = 42
    # "auto" -> cuda if available, else cpu (warn on cpu)
    DEVICE: str = "auto"

    # --- Data
    # MNIST side (1x28x28): input of the LR baseline
    IMAGE_SIZE: int = 28
    # DenseNet input side (3x128x128)
    DENSENET_SIZE: int = 128
    NUM_CLASSES: int = 10
    # stratified split of the 60k train set
    VAL_SIZE: int = 6000
    # fixed validation images (one per class)
    NUM_DEBUG_IMAGES: int = 10

    # --- Model and training
    CKPT_PATH: str = "./models/checkpoint_fase_0.pt"
    BACKBONE_WEIGHTS: str = "IMAGENET1K_V1"
    # True -> head only, BatchNorm of backbone in eval
    FREEZE_BACKBONE: bool = True
    # warning (not stop) if final validation accuracy is below
    MIN_VAL_ACCURACY: float = 0.95
    NUM_EPOCHS: int = 5
    # to confirm in phase zero
    BATCH_SIZE: int = 64
    # used when FREEZE_BACKBONE=True
    OPTIMIZER: str = "adam"
    LEARNING_RATE_HEAD: float = 1e-3
    # used when FREEZE_BACKBONE=False
    LEARNING_RATE_FULL: float = 5e-5
    # "auto" | "train" | "load"
    MODEL_MODE: str = "auto"
    CHECKPOINT_PATH: str = "checkpoints/densenet121_mnist.pt"
    # BatchNorm of the backbone in eval during training. Effective only when FREEZE_BACKBONE=True
    # (otherwise the backbone trains normally). False reproduces plain model.train().
    FLAG_BATCHNORM_EVAL: bool = True
    # True: keep torchvision's in-place ReLU (less VRAM). False: patch them (non in-place).
    # Nothing switches automatically: if an explanation technique needs False, say so via config.
    FLAG_RELU_INPLACE: bool = True

    # --- Grad-CAM
    GRADCAM_LAYER: str = "features.norm5"
    # for the comparison figure
    GRADCAM_ALT_LAYER: str = "features.denseblock3"
    GRADCAM_DEBUG_EPOCHS: str = "first_last"

    # --- Logistic regression baseline
    LR_PENALTY: str = "l2"
    # optimize and default algoritm based on gradient
    # support L2 regolarize method
    LR_SOLVER: str = "lbfgs"
    # check the power of regolarize.
    # It is inverse value of alpha cioè C = 1 \ alpha .
    # t is the standard trade off between
    # model flexibility and the penalization of complexity.
    LR_C: float = 1.0
    # check ConvergenceWarning in phase zero
    LR_MAX_ITER: int = 1000

    # --- XAI techniques
    # "lime" | "captum"  (provisional, phase zero)
    LIME_BACKEND: str = "lime"
    # "shap" | "captum"  (provisional, phase zero)
    SHAP_BACKEND: str = "shap"
    IG_STEPS: int = 50
    OCCLUSION_WINDOW: int = 16
    OCCLUSION_STRIDE: int = 8
    LIME_NUM_SAMPLES: int = 1000
    # 8x8 grid segmentation (to check visually)
    LIME_GRID_CELLS: int = 8
    SHAP_NUM_BACKGROUND: int = 50
     # gradient samples of Gradient SHAP (phase zero)
    SHAP_NUM_SAMPLES: int = 200

    # --- Sampling
    GALLERY_SIZE: int = 24
    # quantitative sample; fixed after timing in phase zero
    N_QUANT: int = 100

    # --- Comparison
    # Spearman and IoU
    COMPUTE_AGREEMENT: bool = True
    # "absolute" | "positive"
    SIGN_MODE: str = "absolute"
    # "digit" | "full"
    MASK_MODE: str = "digit"
    # intensity above which a pixel belongs to the digit
    MASK_THRESHOLD: float = 0.1
    MASK_DILATION: int = 1
    TOPK_FRACTION: float = 0.2
    # deletion curves
    COMPUTE_FAITHFULNESS: bool = False
    DELETION_STEPS: int = 10

    # --- Execution and outputs
    RUN_TESTS: bool = False
    OUTPUT_DIR: str = "outputs"
    IN_NOTEBOOK = "ipykernel" in sys.modules



CONFIG = ConfigConstantsDefault()

THRESHOLDS = {
    "backend_max_seconds_per_image": 15.0,    # LIME / SHAP backend accepted if <= (T4)
    "quant_stage_max_minutes": 20.0,          # time budget of the quantitative stage
    "run_all_load_max_minutes": 30.0,         # "Esegui tutto" with MODEL_MODE=load
    "run_all_train_max_minutes": 45.0,        # "Esegui tutto" with MODEL_MODE=train
    "ig_completeness_pass_rel": 0.05,         # |sum(attr) - (f(x)-f(b))| / |f(x)-f(b)|
    "ig_completeness_warn_rel": 0.20,
    "logits_patch_atol": 1e-4,                # logits before/after the ReLU patch
    "preproc_max_abs_diff": 0.05,             # tensor path vs torchvision PIL path (normalized units)
    "lr_expected_acc_range": (0.90, 0.94),    # design expects ~92 %
    "lr_colsum_warn": 1e-3,                   # sum over classes of LR coefficients, per pixel
    "loss_still_dropping_rel": 0.05,          # last-epoch relative val-loss drop => "still dropping"
    "mask_fraction_range": (0.05, 0.60),      # sane share of pixels in the digit mask
    "determinism_rtol": 1e-3,
}

# Random seed for reproducibility




# SEED = 42
SEED = CONFIG.SEED
set_seed(SEED)


# BATCH_SIZE=32
BATCH_SIZE=CONFIG.BATCH_SIZE

print(torch.cuda.is_available())
# settiamo 'cuda' per la GPU altrimenti verra settata la 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(CONFIG.IN_NOTEBOOK)
# !ls ./models/checkpoint_fase_0.pt
!ls ./models



"""### Modello
- DenseNet121 con pesi `IMAGENET1K_V1`; `classifier` sostituito da `Linear(1024, 10)`.
- **ReLU in-place:** i blocchi usano `ReLU(inplace=True)` e il `forward` finale usa `F.relu(..., inplace=True)` in forma funzionale. Questo rompe gli hook di Grad-CAM e il backward di Captum/SHAP. Va gestito (moduli a `inplace=False`; per il `forward` finale un wrapper sottile). **Da verificare in fase zero.**
- `FREEZE_BACKBONE=True`: `requires_grad=False` su `features`; l'ottimizzatore riceve solo i parametri del `classifier`; **le BatchNorm del backbone restano in `eval`**.

### Perché le BatchNorm in eval
`requires_grad=False` ferma l'aggiornamento dei **pesi**, ma non quello delle **statistiche** (`running_mean`, `running_var`) delle BatchNorm, che in modalità `train()` si aggiornano a ogni batch. MNIST è molto diverso da ImageNet (sfondo quasi tutto nero), quindi le statistiche si sposterebbero e le feature "congelate" cambierebbero senza che nessun peso sia stato modificato. `model.train()` rimette tutti i moduli in train, quindi il freeze va **riapplicato a ogni inizio di epoca** (anche dopo la validation).

**FUNCTION PLOT GRAPH AND IMAGES** <br>
"""

# funzione per convertire un'immagine da tensore ad un formato visualizzabile
def back_to_image_05(img):
    # DeNormalizzazione
    # faccio l'operazione inversa della normalizzazione
    # domanda su ProfAi
    img = img / 2 + 0.5
    # trasformo in un vettore numpy
    npimg = img.numpy()
    # print("npimg",type(npimg))
    # Facciamo una trasposta perchè , per poter essere
    # visualizzati i canali devono essere all'ultimo posto
    # Non avremo più un tensore ma una matrice numpy
    # avente shape 28x28x1

    # La funzione squeeze di NumPy viene utilizzata per rimuovere le dimensioni
    # di un array che hanno una dimensione pari a 1. In altre parole, "squeeze"
    # elimina le dimensioni singolari di un array, riducendo il numero di dimensioni
    # senza perdere dati.
    # return img.squeeze() # -> soluzione del tutor
    return np.transpose(npimg, (1, 2, 0)) # soluzione esercizio il risultato è lo stesso

def imshow_05(img):
    """
        Trasformazione inversa dell'immagine nello spazio dei colori.
        Faccio una trasposta nello spazio dei colori altrimenti
        l'immagine non avrà il colore corretto.
        la normalizzazione è diversa da quella sul dataset
        ma influisce solo sulla visualizzazione
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
def plot_confusion_matrix_lr(model,X,y, title , str_train_test):
    """
         Plot Confusion Matrix Table
    """

    cm = confusion_matrix(y, model.predict(X))


    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted ' + str_train_test)
    plt.ylabel('True ' + str_train_test)
    plt.title(title + str_train_test)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"  figure saved: {path}")
    plt.show() if CONFIG.IN_NOTEBOOK else plt.close(fig)

"""**FUNCTION LOAD AND PREPROCESSING DATASET** <br>

"""

def load_dataset_lr():

    transform_lr = transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                    # Converti in tensore
                                    transforms.ToTensor(),
                                  ])

    mnist_trainvalset_lr = datasets.MNIST(root='./data', train=True, download=True, transform=transform_lr)
    mnist_testset_lr = datasets.MNIST(root='./data', train=False, download=False, transform=transform_lr)

    return mnist_trainvalset_lr,mnist_testset_lr

def load_dataset_cnn():
    # Prendo immagini a caso e le visualizzo
    transform_densenet = transforms.Compose([ # Aumenta altezza e larghezza delle immagini
                                     transforms.Resize((128, 128)),
                                     # Converti in tensore
                                     transforms.ToTensor(),
                                    # espando i canali da 1 a 3
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # Espande a 3 canali
                                    # normalizzazione delle img
                                    transforms.Normalize(
                                    mean=   [0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                    )
                                    ])

    mnist_trainvalset_cnn = datasets.MNIST(root='./data', train=True, download=True, transform=transform_densenet)
    mnist_testset_cnn = datasets.MNIST(root='./data', train=False, download=False, transform=transform_densenet)

    return mnist_trainvalset_cnn,mnist_testset_cnn

def split_dataset(trainvalset ,train_size = 54000,val_size = 6000):


    # (Funziona per i dataset Torch standard come MNIST)
    targets = trainvalset.targets

    # Calcoliamo la frazione di validazione desiderata (es. 6000 su 54000 = ~11.1%)
    val_fraction = val_size / (train_size + val_size)

    # 2. Otteniamo gli indici stratificati per train e validation
    # stratify=targets: Dice a scikit-learn di mantenere la stessa identica percentuale di ogni classe (da '0' a '9') sia in train_indices che in val_indices.

    train_indices, val_indices = train_test_split( range(len(trainvalset)),
                                                    test_size=val_fraction,
                                                    # stratificazione!
                                                    stratify=targets,
                                                    random_state=SEED
                                                   )

    # 3. Creiamo i Subset di PyTorch usando gli indici stratificati
    train_dataset = Subset(trainvalset, train_indices)
    val_dataset = Subset(trainvalset, val_indices)

    return train_dataset,val_dataset

def dataloader_to_tensor( dataloader_ , channels , w , h ):

    # Alloco i tensori 1, 28, 28
    # Dove sono allocati sulla GPU o la CPU ?
    X = torch.empty((len(dataloader_.dataset),  channels , w , h),
                dtype=torch.float32)
    y = torch.empty((len(dataloader_.dataset),),
                dtype=torch.int64)

    start = 0
    for images, labels in dataloader_:
        # images ha forma (64, 3, 128, 128) -> il tuo X
        # labels ha forma (64,)             -> il tuo y
        n = images.size(0)

        X[start:start+n].copy_(images)
        y[start:start+n].copy_(labels)

        start += n

    logger.debug(" X " , X.shape)
    logger.debug(" y " , y.shape)
    return X , y

def get_images_labels(dataset_,flag_data_loader=True,idx_dataset=0):

    if flag_data_loader:
        # sola volta in tutto il codice
        dataloader = DataLoader(dataset_, batch_size=32, shuffle=True)
        print(len(dataset_), "immagini,", len(dataset_.class_to_idx), "identità")
        dataiter = iter(dataloader)
        # i valori di images e labels sono tensori di pytorch
        images_, labels_ = next(dataiter)
    else:
        image, label = dataset_[idx_dataset]
        # images_ = [image]
        # labels_ = [label]
        # torch.stack() prende in input una lista di tensori e li impila lungo una nuova dimensione.
        images_ = torch.stack([image])
        # costruisce un nuovo tensore a partire da un valore (o lista di valori) Python nativi
        labels_ = torch.tensor([label])
    return images_, labels_


def load_data_lr():
    """
        load and process data for Logistic Regression
    """
    trainval_dataset_lr , test_dataset_lr = load_dataset_lr()
    train_size = len(trainval_dataset_lr) - CONFIG.VAL_SIZE
    val_size = CONFIG.VAL_SIZE

    logger.debug(f" train_size {train_size}")
    logger.debug(f" val_size {val_size}")

    train_dataset_lr , val_dataset_lr = split_dataset(trainval_dataset_lr ,train_size = train_size,val_size = val_size)

    train_loader_lr = DataLoader(train_dataset_lr, batch_size=BATCH_SIZE,shuffle=True)
    val_loader_lr = DataLoader(val_dataset_lr, batch_size=BATCH_SIZE,shuffle=False)
    test_loader_lr = DataLoader(test_dataset_lr, batch_size=BATCH_SIZE,shuffle=False)

    # Carico i tensori di train e test
    X_train , y_train = dataloader_to_tensor(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val = dataloader_to_tensor(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_test , y_test = dataloader_to_tensor(test_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)

    # appiattisco i tensori
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    logger.debug(f" X_train flat {X_train.shape}")
    logger.debug(f" X_val flat {X_val.shape}")
    logger.debug(f" X_test flat {X_test.shape}")

    # Converto il tensore nel formato numpy preferito da LogisticRegression
    X_train = X_train.numpy()
    X_val = X_val.numpy()
    X_test = X_test.numpy()
    return  X_train,X_val, X_test,  y_train ,y_val, y_test

"""**FUNZIONE SPECIFICHE PER LA REGRESSIONE LOGISTICA**"""

def plot_logistic_graph(model_,config , out_dir , c ):
    out_dir_path = Path(out_dir)  # Converte la stringa in Path
    rel = model_.coef_ - model_.coef_.mean(axis=0, keepdims=True)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
    for k, ax in enumerate(axes.ravel()):
        a = max(np.abs(rel[k]).max(), 1e-12)
        ax.imshow(rel[k].reshape(config.IMAGE_SIZE, config.IMAGE_SIZE), cmap="seismic", vmin=-a, vmax=a)
        ax.set_title(f"class {k}")
        ax.axis("off")

    fig.suptitle(f"LR global maps (relative coefficients), C={c}")
    # save_fig(fig, out_dir  + f"phase0_05_lr_global_maps.png")
    save_fig(fig, out_dir_path / f"phase0_lr_global_maps.png")

def lr_analyze(diag, cfg):
    """
        print result and score about Logistic Regression.
    """
    tag = "LogisticRegression"
    lo, hi = THRESHOLDS["lr_expected_acc_range"]

    logger.info( f"{tag}: fit {diag['fit_seconds']:.1f}s, "
           f"n_iter {diag['n_iter']}, "
           f"val acc {diag['val_accuracy']:.4f}"
         )

    # 1. Convergence
    if diag["convergence_warning"]:
        logger.info(
            f"[WARN] {tag}: ConvergenceWarning=True -> "
            f"consider raising LR_MAX_ITER (now {cfg.LR_MAX_ITER})"
        )
    else:
        logger.info(f"[OK] {tag}: ConvergenceWarning=False")

    # 2. Validation accuracy
    acc = diag["val_accuracy"]
    if lo <= acc <= hi:
        logger.info(
            f"[OK] {tag}: validation accuracy {acc:.4f} "
            f"(expected ~0.92, range {lo}-{hi})"
        )
    else:
        logger.info(
            f"[WARN] {tag}: validation accuracy {acc:.4f} "
            f"(expected ~0.92, range {lo}-{hi})"
        )

    # 3. Verifica della decision function
    err = diag["decision_exact_max_err"]
    if err < 1e-8:
        logger.info(
            f"[OK] {tag}: logit = sum(w*x)+b exactly "
            f"(max error {err:.1e})"
        )
    else:
        logger.info(
            f"[FAIL] {tag}: logit mismatch "
            f"(max error {err:.1e})"
        )

    # 4. Classi
    if diag["classes_ok"]:
        logger.info(f"[OK] {tag}: classes_ == 0..9 (coefficient rows aligned)")
    else:
        logger.info(f"[FAIL] {tag}: classes_ are not aligned with 0..9")

    # 5. Somma dei coefficienti per pixel
    colsum = diag["coef_colsum_max"]
    if colsum <= THRESHOLDS["lr_colsum_warn"]:
        logger.info(
            f"[OK] {tag}: per-pixel sum of coefficients over classes "
            f"max {colsum:.1e} "
            "(~0 with L2: relative and absolute coefficients coincide)"
        )
    else:
        logger.info(
            f"[WARN] {tag}: per-pixel sum of coefficients over classes "
            f"max {colsum:.1e}"
        )

    # 6. Altri warning
    for warning in diag["other_warnings"]:
        logger.info(f"[INFO] {tag}: warning raised -> {warning}")

def lr_fit_and_diagnose(X_train_, y_train_, X_val_, y_val_ , cfg) -> tuple:
    """Fit the LR and measure everything check 5 needs. Returns (model, diagnostics)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model = LogisticRegression(random_state=SEED,max_iter=cfg.LR_MAX_ITER).fit(X_train_, y_train_)
        fit_seconds = time.time() - t0

    categories = sorted({w.category.__name__ for w in caught})
    manual = X_val_ @ model.coef_.T + model.intercept_
    diag = {
        "fit_seconds": fit_seconds,
        "n_iter": int(np.max(model.n_iter_)),
        "convergence_warning": "ConvergenceWarning" in categories,
        "other_warnings": sorted({f"{w.category.__name__}: {str(w.message)[:140]}"
                                  for w in caught if w.category.__name__ != "ConvergenceWarning"}),
        "val_accuracy": float(model.score(X_val_, y_val_)),
        "coef_colsum_max": float(np.abs(model.coef_.sum(axis=0)).max()),
        "decision_exact_max_err": float(np.abs(manual - model.decision_function(X_val_)).max()),
        "classes_ok": bool(np.array_equal(model.classes_, np.arange(cfg.NUM_CLASSES))),
    }
    return model, diag



def run_train_lr(max_iter_=100):
    """
        Train Regressione Logistica
    """

    X_train ,X_val, X_test , y_train ,y_val ,  y_test = load_data_lr()
    logger.debug(f" X_train numpy {X_train.shape}")
    logger.debug(f" X_test numpy {X_test.shape}")

    # lr = LogisticRegression( random_state=SEED,max_iter=max_iter)
    lr , diag = lr_fit_and_diagnose(X_train, y_train, X_val, y_val ,CONFIG)

    # Calcolo dell'accuratezza di addestramento e test per la Logistic Regression
    test_accuracy_lr = accuracy_score(y_test, lr.predict(X_test))
    logger.info(f"Accuratezza di test (Logistic Regression): {test_accuracy_lr}")

    lr_analyze(diag,CONFIG)

    OUT_DIR="./output_lr"
    plot_logistic_graph(lr, CONFIG, OUT_DIR, c=CONFIG.LR_C)

    # stampo solo la confusion matrix sul test
    # plot_confusion_matrix(lr,X_train,y_train, "CLASSIFIER LR ON MNIST DATASET " , "TRAIN")
    plot_confusion_matrix_lr(lr,X_test,y_test, "CLASSIFIER LR ON MNIST DATASET " , "TEST")

"""**FUNZIONE SPECIFICHE PER LA CNN**"""

def set_batchnorm_eval(module: nn.Module) -> None:
    """Keep BatchNorm running statistics frozen (specification of the design)."""
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def enter_train_mode(model: nn.Module, config) -> None:
    """Single entry point to put the model in training mode. Call at the start of EVERY epoch."""
    model.train()
    if config.FREEZE_BACKBONE and CONFIG.FLAG_BATCHNORM_EVAL:
        model.features.apply(set_batchnorm_eval)


def check_model_params(model):
    """
         Check on model params
    """
    # Controlli sui parametri
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters()
        if p.requires_grad)

    logger.info(f"[build_densenet] Parametri totali: {total_params:,}")
    logger.info(f"[build_densenet] Parametri congelati: {total_params - trainable_params:,}")
    logger.info(f"[build_densenet] Parametri addestrabili: {trainable_params:,}")

    # for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        logger.info(f"{name} , {param.shape}")


def set_relu_inplace(module):
    """
         Apply modify to input model
         set_relu_inplace(model)
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            # Sostituisce il layer ReLU esistente con uno in-place
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            # Scende nei sottoblocchi (es. nei DenseBlock di DenseNet)
            set_relu_inplace(child)




#  model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
def build_densenet(cfg):
    """
         Build Model , optimizer and define loss Function
    """
    # weights = getattr(DenseNet121_Weights, CONFIG.BACKBONE_WEIGHTS) if pretrained else None
    # model = densenet121(weights=weights)
    # model.classifier = nn.Linear(base.classifier.in_features, cfg.NUM_CLASSES)

    # codice standard senza scaricare da github
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # permette di risparmiare memoria VRAM
    set_batchnorm_eval(model)

    # DA VERIFICARE
    for param in model.parameters():
         param.requires_grad = False
    # DA VERIFICARE

    # Modifichiamo l'ultimo layer per il numero corretto di classi
    # num_classes = len(trainval_dataset_cnn.class_to_idx)
    # logger.info(f"[build_densenet] num_classes {num_classes}")

    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, num_classes)
    model.classifier = nn.Linear(model.classifier.in_features, cfg.NUM_CLASSES)

    # Controlli sui parametri
    # total_params = sum(p.numel() for p in model.parameters())

    # trainable_params = sum(p.numel() for p in model.parameters()
    #     if p.requires_grad)

    # logger.info(f"[build_densenet] Parametri totali: {total_params:,}")
    # logger.info(f"[build_densenet] Parametri congelati: {total_params - trainable_params:,}")
    # logger.info(f"[build_densenet] Parametri addestrabili: {trainable_params:,}")

    # for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        logger.info(f"{name} , {param.shape}")

    check_model_params(model)

    loss_function = nn.CrossEntropyLoss()
    # Setto optimzers lo stocastic gradient descend
    # un learning rate=0.05
    # il parametro momentum=0.09
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model,optimizer,loss_function

def get_model(cfg):
    """Eval-mode model (in-place or patched ReLU, per FLAG_RELU_INPLACE).
    """
    model_base , _ , _ = build_densenet(cfg)
    # trained = cfg.CKPT_PATH.exists()
    trained = Path(cfg.CKPT_PATH).exists()
    if trained:
        # model_base.load_state_dict(torch.load(cfg.CKPT_PATH, map_location="cpu")["state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carichi il dizionario del checkpoint
        checkpoint = torch.load(cfg.CKPT_PATH, map_location=device)

        # Ripristini lo stato di modello ed ottimizzatore
        model_base.load_state_dict(checkpoint['model_state_dict'])
        model_base.to(device)

        check_model_params(model_base)
        logger.info("  NOTE: yes checkpoint found ")
    else:
        logger.info("  NOTE: no checkpoint found ")

    # for p in model_base.parameters():
    #    p.requires_grad_(False)

    return model_base

def save_model(cfg,model,optimizer, num_epochs):
    """
    """
    # logger.info(f"{type(loss_cnn)}")
    checkpoint_start = { 'epoch': num_epochs,                            # Epoca corrente
                         'model_state_dict': model.state_dict(),     # Pesi del modello
                         'optimizer_state_dict': optimizer.state_dict() # Stato dell'ottimizzatore (es. momentum, learning rate)
                         # 'loss': loss_cnn,                       # Ultimo valore di loss
                        }

    # trained = cfg.CKPT_PATH.exists()
    trained = Path(cfg.CKPT_PATH).exists()
    if trained == False :
        # Save dict into a file
        # torch.save(checkpoint_start, 'checkpoint_fase_0.pt')
        torch.save(checkpoint_start, cfg.CKPT_PATH)
        logger.info("Checkpoint iniziale salvato con successo!")

# Funzione per visualizzare la matrice di confusione
def plot_confusion_matrix_cnn(y_true_,y_pred_, title , str_train_test):
    """
         Plot Confusion Matrix Table
    """

    cm = confusion_matrix(y_true_, y_pred_)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted ' + str_train_test)
    plt.ylabel('True ' + str_train_test)
    plt.title(title + str_train_test)

def run_train_cnn(model_ ,
                  dataset_,
                  dataloader_train_ ,
                  dataloader_val_ ,
                  device_ ,
                  optimizer_ ,
                  criterion_ ,
                  num_epochs_=1):

    model_ = model_.to(device_)

    # Training loop
    for epoch in range(num_epochs_):
        # model_.train()
        enter_train_mode(model_,CONFIG)
        running_loss = 0.0
        sync(device)
        t0 = time.time()
        for i, (inputs, labels) in enumerate(dataloader_train_):
            inputs, labels = inputs.to(device_), labels.to(device_)

            optimizer_.zero_grad()
            outputs = model_(inputs)
            loss = criterion_(outputs, labels)
            loss.backward()
            optimizer_.step()

            running_loss += loss.item()
            if i % 1000 == 900:    # stampa ogni 100 mini-batch
                logger.info(f'[Epoca {epoch + 1}, Ciclo {i + 1:5d}] Loss: {running_loss / 10:.3f}')
                running_loss = 0.0
                # break

        sync(device)
        train_seconds = time.time() - t0
        t1 = time.time()

        # CICLO VALIDATION
        y_pred,y_true,y_pred_desc,y_true_desc = run_test_cnn(model_,
                                                             dataset_,
                                                             dataloader_val_,
                                                             device_)
        val_accuracy_cnn = accuracy_score(y_true,y_pred)
        if ( val_accuracy_cnn < CONFIG.MIN_VAL_ACCURACY ):
            logger.info(f"ATTENZIONE!! Accuratezza(DenseNet Cnn) Non Supera la Soglia({CONFIG.MIN_VAL_ACCURACY}): {val_accuracy_cnn}")

        sync(device)
        val_seconds = time.time() - t1
        logger.info(f'[Epoca {epoch + 1}, Ciclo {i + 1:5d}] train time: {train_seconds:.3f}  val time: {val_seconds:.3f} ')

    logger.info('Train completato')

def run_test_cnn(model_ ,
                dataset_,
                dataloader_ ,
                device_ ):


    model_ = model_.to(device_)

    model_.eval()
    y_pred_=[]
    y_true_=[]

    y_pred_desc_=[]
    y_true_desc_=[]
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_):
            images, labels = images.to(device_), labels.to(device_)

            # print images gid with relative prevision and ground truth

            # prendo l'output delle previsioni direttamente
            # dalle immagini caricate applicando la nostra rete neurale convolutiva
            # e addestrata .
            outputs_ = model_(images)
            # Ottengo il valore della classe prevista
            # usando le etichette per gli output ottenuti che abbiamo definito come prima.
            _, predicted = torch.max(outputs_, 1)

            y_true_.append(labels)
            y_pred_.append(predicted)

            # print("y_true",y_true.shape)
            # print("y_pred",y_pred.shape)
            # y_true (10000,)
            # y_pred (10000,)

            y_pred_desc_.append(dataset_.classes[lbl] for lbl in predicted.cpu())
            # senza convertire numpy avrei una lista di tensori
            # y_pred_ = [ tensor([...64 predizioni...]), tensor([...64 predizioni...]), ... ]
            y_true_desc_.append(dataset_.classes[lbl] for lbl in labels.cpu())



    y_true_ = torch.cat(y_true_).cpu().numpy()
    y_pred_ = torch.cat(y_pred_).cpu().numpy()

    return y_pred_,y_true_,y_pred_desc_,y_true_desc_



def run_predict_on_image(dataset_,images_,labels_,model_):

    images_, labels_ = images_.to(device), labels_.to(device)
    # print images gid with relative prevision and ground truth
    imshow(torchvision.utils.make_grid(images_.cpu().detach()))

    # prendo l'output delle previsioni direttamente
    # dalle immagini caricate applicando la nostra rete neurale convolutiva
    # e addestrata .
    outputs_ = model_(images_)
    # Ottengo il valore della classe prevista
    # usando le etichette per gli output ottenuti che abbiamo definito come prima.
    _, predicted_ = torch.max(outputs_, 1)

    gt_pred_desc_ = [dataset_.classes[lbl] for lbl in predicted_]
    # logger.info(f"gt_pred_desc_ {gt_pred_desc_}")

    gt_label_desc_ = [dataset_.classes[lbl] for lbl in labels_]
    # logger.info(f"gt_label_desc {gt_label_desc}")
    # logger.info(f"labels len {len(labels)}")
    # logger.info(f"predicted len {len(predicted)}")
    #                               for j in range(4)))

    df_ = pd.DataFrame({"Ground Truth": gt_label_desc_, "Prediction": gt_pred_desc_})
    return df_ , gt_pred_desc_ , gt_label_desc_ , predicted_
    # 3. (Opzionale) Visualizza il DataFrame nel log o a schermo
    # logger.info(f"\n{df.head()}")


def run_main_cnn(cfg,num_epochs_=5):

    trainval_dataset_cnn , test_dataset_cnn = load_dataset_cnn()

    train_size = len(trainval_dataset_cnn) - CONFIG.VAL_SIZE
    val_size = CONFIG.VAL_SIZE

    train_dataset_cnn , val_dataset_cnn = split_dataset(trainval_dataset_cnn ,train_size = train_size,val_size = val_size)

    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=BATCH_SIZE,shuffle=True)
    val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=BATCH_SIZE,shuffle=False)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=BATCH_SIZE,shuffle=False)

    # codice standard senza scaricare da github
    # model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # permette di risparmiare memoria VRAM
    # set_batchnorm_eval(model)

    # for param in model.parameters():
    #     param.requires_grad = False

    # Modifichiamo l'ultimo layer per il numero corretto di classi
    # num_classes = len(trainval_dataset_cnn.class_to_idx)

    # logger.info(f"num_classes {num_classes}")

    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, num_classes)

    # Controlli sui parametri
    # total_params = sum(p.numel() for p in model.parameters())

    # trainable_params = sum(p.numel() for p in model.parameters()
    #     if p.requires_grad)

    # logger.info(f"Parametri totali: {total_params:,}")
    # logger.info(f"Parametri congelati: {total_params - trainable_params:,}")
    # logger.info(f"Parametri addestrabili: {trainable_params:,}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"{name} , {param.shape}")

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model , optimizer , loss_function = get_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_train_cnn(model_=model,
                  dataset_=val_dataset_cnn,
                  dataloader_train_=train_loader_cnn,
                  dataloader_val_=val_loader_cnn,
                  device_=device,
                  optimizer_=optimizer,
                  criterion_=loss_function,
                  num_epochs_=num_epochs_)

    y_pred, y_true, y_pred_desc, y_true_desc = run_test_cnn(model,test_dataset_cnn,test_loader_cnn,device)

    test_accuracy_cnn = accuracy_score(y_true,y_pred)
    logger.info(f"Accuratezza di test (DenseNet Cnn): {test_accuracy_cnn}")
    plot_confusion_matrix_cnn(y_true,y_pred, "CLASSIFIER LR ON MNIST DATASET " , "TEST")

    save_model(CONFIG,model,optimizer, num_epochs_)

    return model,optimizer,loss_function



"""**BASELINE REGRESSIONE LOGISTICA** <br>

LogisticRegression di scikit-learn si aspetta normalmente una matrice bidimensionale: (numero_campioni, numero_feature). <br>
Quindi, prima ancora di discutere la RAM,
dobbiamo capire che cosa rappresenta ogni campione per il classificatore. <br>

Un'immagine RGB di dimensione 128×128 contiene tre canali. Volendo utilizzare ogni valore dei pixel come feature, quante feature avrà una singola immagine dopo averla appiattita? <br>
Un'immagine RGB di dimensione 128×128 contiene tre canali. Volendo utilizzare ogni valore dei pixel come feature, quante feature avrà una singola immagine dopo averla appiattita?  <br>
3x128x128=49152 perche le feature sono replicate sui tre canali quindi 49152  <br>


Ti propongo di ragionare su tre livelli: <br>

Attenzione alla RAM: il percorso a 28×28 riduce notevolmente la memoria rispetto a 128×128,
ma Logistic Regression può comunque richiedere memoria aggiuntiva durante l'addestramento.  <br>
Se necessario, valutare anche solver, regolarizzazione e una rappresentazione ancora più compatta. <br>

Il punto fondamentale è che possiamo mantenere le immagini a tre canali e 128×128 per DenseNet121,
usando una rappresentazione separata e più leggera per Logistic Regression.<br>


Dato originale: tensore PyTorch con quattro dimensioni. BATCH, CANALI , H , W <br>
Dato appiattito: matrice con 54.000 righe e 49.152 colonne. <br>
Modello: LogisticRegression, che durante il fit può richiedere
ulteriore memoria per le sue elaborazioni. <br>
l numero di elementi è: 54000x49152x4 byte ogni elemento un numero molto alto <br>

la matrice X_train in input al metodo fit
contiene circa  54000x49152=2,65 miliardi di valori.  <br>
Ma questo significa che la memoria necessaria è molto maggiore di quanto possa sembrare inizialmente. <br>

MNIST originale 1x28x28 = 784 <br>
MNIST modificato 3x128x128 = 49152 <br>

Posso utilizzare due diversi transform uno per il DenseNet e uno per per LogisticRegression rappresentando le immagini con 28x28 per LR e lasciare 3x128x128 solo per DenseNet

"""

run_train_lr(max_iter_=CONFIG.LR_MAX_ITER)

"""# Importa un modello DenseNet da torchvision <br>
[Documentazione](https://pytorch.org/hub/pytorch_vision_densenet/) <br>
**Descrizione Modello nella documentazione** <br>

Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections – one between each layer and its subsequent layer – our network has L(L+1)/2 direct connections. <br>
For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters <br>

**Dati utilizzati dal modello**<br>

All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].















**DEFINISCO LA CLASSE DEL MODELLO DI RETE CONV DA ADDESTRARE** <br>

### Dimensione in input/output
Input: $N$, $C_{in}$, H, W  \\
Layer Convoluzionali: $[N, n\_filters, H- (Kernel\_size -1), W-(Kernel\_size -1)]$ \\
Max Pool 2d: Output: $[N, C_{in}, \frac{H}{2}, \frac{W}{2}]$

### Parameteri

Layer Convoluzionali: $(((Kernel_{w} * Kernel_{h})*n_{canali})+1)*n_{filtri}$ \\
Layer lineari : $(input + 1) * output$ \\
Max Pool 2d: 0

#**Trasformazioni per il dataset** <br>


#**ARCHITETTURA DEL MODELLO XAI di tipo DenseNet Modificato** <br>

Partiamo dal modello DensNet addestrato su imagenet e facciamo le seguenti modifiche alla struttura : <br>


1. **Modifica il numero di classi dell'ultimo layer**
Trasformiamo quella rete in una rete che vada per ImageNet per questo tagliamo via l'ultimo strato di classificazione come si dovrà fare anche nel progetto finale che lavora con il Mnist.(dataset di cifre) e lo facciamo uscire invece che con il numero di classi di ImageNet con il numero di classi di LabelFaceINtheWild <br>

2. **Freeziamo tutti i parametri della rete in modo che non siano modificati dal nostro train**. <br>
Facciano transferLearning ovvero scarichiamo il modello DenseNet e usianmo i suoi parametri freezati perchè l'assunto è che pur essendo stato addestrato su un altro dataset potrebbe aver imparato qualcosa d'interessante su cosa e come vedere. Il modello è stato addestrato su ImageNet ma io lo voglio riaddestrare sul nostro dataset . <br>
Il modello viene tagliato a partire dall'ultimo strato convolutivo e verrà addestrato a imparare a classificare sul LabelFaceInTheWildPeople <br>
Per velocizzare il tutto tengo buoni i primi strati convolutivi originali del DenseNet che sono freezati non verranno addestrati e modificati tranne l'ultimo strato aggiunto da noi<br>

**APPUNTI INTERPRETAZIONE DEL CODICE** <br>
**1. DIMENSIONI DEI TENSORI IN INPUT** <br>

Il valore 1024 viene dall’architettura della DenseNet121 pre-addestrata, non dai pixel dell’immagine.

Quando dai alla DenseNet un tensore 3 × 128 × 128, l’immagine attraversa molti layer convoluzionali, pooling e dense blocks. A ogni passaggio le dimensioni spaziali (altezza × larghezza) vengono ridotte, mentre il numero di canali/feature viene trasformato e aumentato. Alla fine, la rete fa una sintesi delle feature visive e restituisce un vettore di dimensione fissa pari a 1024.

In forma semplificata:

3 × 128 × 128
      ↓
Convoluzioni e Dense Blocks
      ↓
Mappe di feature interne
      ↓
Global Average Pooling
      ↓
Vettore di 1024 feature
      ↓
Linear(1024, 10)
Il Global Average Pooling è il passaggio chiave: prende ciascuna mappa di feature finale e ne calcola una sintesi, normalmente una media. Se alla fine il backbone ha 1024 mappe di feature, il pooling produce un numero per ciascuna mappa, quindi un vettore di 1024 valori.

Per questo, il layer originale della DenseNet aveva una struttura concettuale simile a:

nn.Linear(1024, 1000)
per classificare le 1000 classi di ImageNet. Tu lo hai sostituito con:

nn.Linear(1024, 10)
per classificare le 10 cifre di MNIST. Hai cambiato solo il numero di output, mentre le 1024 feature in ingresso al classificatore sono determinate dal backbone DenseNet121.

Un punto utile: grazie al pooling finale, lo stesso backbone può spesso elaborare immagini con dimensioni spaziali differenti, purché sufficientemente grandi e compatibili con l’architettura. Il vettore finale rimane comunque lungo 1024.

**2. INTERPRETAZIONE DEI VALORI IN OUTPUT** <br>

Francesco. I 10 valori in uscita dal layer Linear(1024, 10) corrispondono effettivamente alle 10 classi MNIST, da 0 a 9.

La precisazione importante è questa: l’output diretto del modello non è ancora la probabilità di ogni classe, ma un insieme di logits.

Per esempio, il modello potrebbe produrre:

[-1.2, 0.4, 3.8, 0.1, -0.7, 1.0, -2.3, 0.5, 0.9, -0.4]
il primo valore è lo score associato alla classe 0;
il secondo alla classe 1;
...
il decimo alla classe 9.
Lo score più alto è 3.8, che si trova in posizione 2: la classe predetta è quindi la cifra 2.

Quando usi:

_, predicted_ = torch.max(outputs_, dim=1)
stai estraendo l’indice del logit più alto, cioè la classe scelta dal modello.

Se volessi interpretare quei valori come probabilità, dovresti applicare softmax: trasforma i 10 logits in valori compresi tra 0 e 1 la cui somma è 1. Ad esempio:

Classe 0 → 0,01
Classe 1 → 0,03
Classe 2 → 0,79
...
Con CrossEntropyLoss, però, non devi applicare softmax manualmente durante il training: la loss riceve direttamente i logits e include internamente il passaggio necessario, in modo numericamente stabile.

La scelta della classe resta identica sia usando logits sia usando probabilità softmax, perché softmax mantiene l’ordine dei valori: il valore più grande resta quello della classe predetta.

Nel tuo test, per calcolare accuracy e confusion matrix, ti bastano gli indici predetti ottenuti con torch.max: non serve calcolare le probabilità.

Quale informazione aggiuntiva potresti ottenere dalle probabilità softmax, oltre alla classe predetta

**TRAIN PRODUZIONE CON 5 EPOCHE** <br>


1- Creo la rete e la sposto su Gpu con il metodo to()
1. Setup del modello e dataset
2. Carichiamo DenseNet pre-addestrato
"""

num_epochs=5
model_cnn , _ ,_ = run_main_cnn(CONFIG,num_epochs_=num_epochs)
# 08:09:46 - INFO - [run_main_cnn] - Accuratezza di test (DenseNet Cnn): 0.9496
# 08:02:54 - INFO - [run_main_cnn] - Accuratezza di test (DenseNet Cnn): 0.9489
# RISULTATO CON FLAG BATCHNORM EVAL TRUE
# 08:13:23 - INFO - [run_main_cnn] - Accuratezza di test (DenseNet Cnn): 0.9732
# RISULTATO CON FLAG BATCHNORM EVAL TRUE e RELUINPLACE TRUE
# 08:30:57 - INFO - [run_main_cnn] - Accuratezza di test (DenseNet Cnn): 0.9732

"""**SALVO I PESI DEL MODELLO PER NON RIFARE IL TRAIN**
oppure il carico pesi e optimizer salvati prima
"""

model  = get_model(CONFIG)


