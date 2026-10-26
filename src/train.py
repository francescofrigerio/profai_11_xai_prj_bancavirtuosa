
import random
import time
import torch
import numpy as np
from pathlib import Path

import torch
import torchvision

from model import build_densenet, validate_layers
from model import enter_train_mode ,  freeze_backbone, make_optimizer , define_loss_function
from sklearn.metrics import accuracy_score

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(cfg) -> tuple:
    """(device, warning or None). A visible warning on CPU: the full run would be very slow."""
   
    want_cuda = cfg.DEVICE == "cuda" or (cfg.DEVICE == "auto" and torch.cuda.is_available())
    if cfg.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but no GPU is available")
    device = torch.device("cuda" if want_cuda else "cpu")
    warning = None if want_cuda else "NO GPU: 'Esegui tutto' would be very slow. Use a GPU runtime."
    return device, warning

def resolve_mode(cfg, ckpt_exists: bool) -> tuple:
    """
        (action, reason). action is "train" or "load".
    """
    if cfg.MODEL_MODE == "train":
        return "train", "MODEL_MODE=train"
    if cfg.MODEL_MODE == "load":
        if not ckpt_exists:
            raise FileNotFoundError(f"MODEL_MODE=load but there is no checkpoint at {cfg.CHECKPOINT_PATH}")
        return "load", "MODEL_MODE=load"
    
    return "error", "error"

def save_checkpoint(path, model, cfg, history) -> str:
    """
        Saves weights + history + version. 
        Returns the path of the file.
    """
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "history": history,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
    }, path)
    return path


def read_checkpoint(path) -> dict:
    return torch.load(path, map_location="cpu")

def evaluate_model( model_ ,
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

            y_pred_desc_.append(dataset_.classes[lbl] for lbl in predicted.cpu())
            y_true_desc_.append(dataset_.classes[lbl] for lbl in labels.cpu())

    y_true_ = torch.cat(y_true_).cpu().numpy()
    y_pred_ = torch.cat(y_pred_).cpu().numpy()

    return y_pred_,y_true_

def train_model(cfg,
                logger,
                model_ ,
                dataset_,
                dataloader_train_ ,
                dataloader_val_ ,
                device_ ,
                num_epochs_):

    model_ = model_.to(device_)

    set_seed(cfg.SEED)
    if cfg.FREEZE_BACKBONE:
        freeze_backbone(model_)
    optimizer = make_optimizer(model_, cfg)

    criterion = define_loss_function()
    history = []
    train_loss = 0 , val_accuracy = 0
    # Training loop
    for epoch in range(num_epochs_):
        enter_train_mode(model_,cfg)
        running_loss = 0.0
        sync(device_)
        t0 = time.time()
        for i, (inputs, labels) in enumerate(dataloader_train_):
            inputs, labels = inputs.to(device_), labels.to(device_)

            optimizer.zero_grad()
            outputs = model_(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            running_loss += train_loss
            if i % 1000 == 900:    # stampa ogni 100 mini-batch
                logger.info(f'[Epoca {epoch + 1}, Ciclo {i + 1:5d}] Loss: {running_loss / 10:.3f}')
                running_loss = 0.0
                # break
        sync(device_)
        train_seconds = time.time() - t0
        t1 = time.time()

        # CICLO VALIDATION
        y_pred,y_true = evaluate_model( model_,
                                        dataset_,
                                        dataloader_val_,
                                        device_ )
        val_accuracy = accuracy_score(y_true,y_pred)
        if ( val_accuracy < cfg.MIN_VAL_ACCURACY ):
            logger.info(f"WARNING!! Accuracy(DenseNet Cnn) not over the treshold({cfg.MIN_VAL_ACCURACY}): {val_accuracy_cnn}")

        
        sync(device_)
        val_seconds = time.time() - t1
        logger.info(f'[Epoca {epoch + 1}, loop {i + 1:5d}] train time: {train_seconds:.3f}  val time: {val_seconds:.3f} ')

        history.append({
                    "epoch": epoch, 
                    "train_loss": train_loss, 
                    "val_acc": val_accuracy,
                    "train_seconds": train_seconds,
                    "val_seconds": val_seconds})
        
    logger.info('Train finisched')
    return history



def get_trained_model(cfg,
                      logger,
                      test_datasets,
                      train_loader_cnn,
                      val_loader_cnn,
                      test_loader_cnn, 
                      device, 
                      pretrained=True) -> tuple:
    """
        Builds the model according to MODEL_MODE. 
        pretrained=False` exists for the tests only (no ImageNet download)
        Returns (model in eval mode, info dict).
    """
    
    path = Path(cfg.CHECKPOINT_PATH)
    saved = read_checkpoint(path) if path.exists() else None
    action, reason = resolve_mode(cfg, saved is not None)
    logger.info(f"MODE: {action} ({reason})")
    if action == "load":
        model = build_densenet(cfg, pretrained=False)
        model.load_state_dict(saved["state_dict"])
        history = saved["history"]
    else:
        # BEFORE building: the random head must be reproducible
        set_seed(cfg.SEED)                      
        model = build_densenet(cfg, pretrained=pretrained)
        model.to(device)
        history = train_model(cfg,
                              logger,
                              model,
                              test_datasets,
                              train_loader_cnn ,
                              val_loader_cnn ,
                              device ,
                              cfg.NUM_EPOCH)
        
        save_checkpoint(path, model, cfg, history)

    # model = model.to(device).eval()
    validate_layers(model, cfg)

    y_pred, y_true = evaluate_model( model ,test_datasets,test_loader_cnn ,device)
    test_accuracy = accuracy_score(y_true,y_pred)
    logger.info(f"Accuratezza di test (DenseNet Cnn): {test_accuracy_cnn}")

    # X, y = data.X_trainval.to(device), data.y_trainval.to(device)
    #_, val_acc = evaluate_model(model, X, y, torch.as_tensor(data.split.idx_val, device=device), cfg, device)
    info = {
        "action": action, 
        "reason": reason,
        "y_pred": y_pred,
        "test_accuracy": test_accuracy, 
        "history": history, "path": path,
        "val_acc_history": history[-1]["val_acc"] if history else None,
        "device": str(device),
    }
    return model, info

    