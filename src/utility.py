"""
    Commont utility for Normalization , Model Cnn and Logistic Regression
"""

from pathlib import Path
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENET_MEAN_TRANSFORM  =  [0.485, 0.456, 0.406]
IMAGENET_STD_TRANSFORM =   [0.229, 0.224, 0.225]

# Statistiche standard ImageNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_saliency_map(saliency_map: np.ndarray) -> np.ndarray:
    """
        Normalize a 2d array in the range [0, 1].
    """
    min_val = saliency_map.min()
    max_val = saliency_map.max()
    if max_val - min_val == 0:
        return np.zeros_like(saliency_map)
    return (saliency_map - min_val) / (max_val - min_val)

def denormalize_imagenet(img_hwc_np):
    """
        Rollback from image net Normalize value
        This function Converts the image from ImageNet scale to a pixels view range [0, 1].
    """
    img_denorm = (img_hwc_np * IMAGENET_STD) + IMAGENET_MEAN
    # Clip to be sure the range will be strictly [0, 1] for LIME
    return np.clip(img_denorm, 0.0, 1.0)

def normalize_imagenet(input_img):
    """
        Normalize for ImageNet from pixels view range
        
        this normalize is mandatory  
        before send the image to the model DenseNet. 
        This net expect pixels has distribuited  
        with this specific mean and std to get correctly the logits
        and the gradients.
       
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=input_img.dtype, device=input_img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=input_img.dtype, device=input_img.device).view(1, 3, 1, 1)

    return (input_img - mean) / std

def black_normalized(size: int, device):
    """
        The black of the ORIGINAL image, expressed in the normalized space.
    """
    return normalize_imagenet(torch.zeros(1, 3, size, size, device=device))

def denormalize_imagenet_np(x_data: np.ndarray) -> np.ndarray:
    """
        numpy version of denormalize image net, for the contract checks (no torch needed).
    """
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float64).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float64).reshape(1, 3, 1, 1)
    return np.asarray(x_data, dtype=np.float64) * std + mean

def ls_dir(str_dir):

    print(f"Comando ls {str_dir} ")

    target_dir = Path(str_dir) 

    if target_dir.exists() and target_dir.is_dir():
        print(f"Contenuto di {target_dir.resolve()} ")
        for item in target_dir.iterdir():
            # Stampa se è una cartella o un file
            tipo = "📁 [DIR] " if item.is_dir() else "📄 [FILE]"
            print(f"{tipo} {item.name}")
    else:
        print(f"La cartella {target_dir} non esiste o non è una directory.")

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


def run_create_list_images( logger, 
                            X_test_ , 
                            y_test_,
                            y_pred_ , 
                            list_class_, 
                            n_images_=50,
                            # n_images_=100,
                            flag_before_ko=True):
    """
        Run for create list images for both lr and cnn
    """
    if not list_class_:
        raise ValueError("list_class non può essere vuota")
     
    n_per_class = max(1, n_images_ // len(list_class_))

    # Maschere su tutte le classi
    correct_mask = y_pred_ == y_test_
    wrong_mask = y_pred_ != y_test_
    logger.debug
    (f" correct_mask {correct_mask} wrong_mask {wrong_mask}")

    list_images_ok = []
    list_images_ko = []

    logger.debug(f" y_test_ {y_test_}")
    logger.debug(f" y_test_ {type(y_test_)}")

    for id_class in list_class_:
        logger.debug(f" start classe {id_class}")

        class_mask = y_test_ == id_class
        logger.debug(f" total_class_mask {class_mask}")

        class_ok_mask = class_mask & correct_mask
        class_ko_mask = class_mask & wrong_mask

        # indici delle maschere in modo che il valore
        # sia utilizzabile
        class_ok_mask = torch.as_tensor(class_ok_mask).cpu().numpy()
        class_ko_mask = torch.as_tensor(class_ko_mask).cpu().numpy()

        idx_ko = np.where((np.asarray(y_test_) == id_class) &
                          (np.asarray(y_test_) != np.asarray(y_pred_)))[0]

        idx_ok = np.where((np.asarray(y_test_) == id_class) &
                          (np.asarray(y_test_) == np.asarray(y_pred_)))[0]

        logger.debug(f"Indici KO di controllo: {idx_ko[:10]}")
        logger.debug(f"Indici OK di controllo: {idx_ok[:10]}")

        # stabilisco quanti errori e quante
        # immagini corrette possiamo selezionare
        if flag_before_ko is True:
            # n_ko non può superare né il numero di errori 
            # disponibili né il limite per classe.
            n_ko = min(len(idx_ko), n_per_class)

            # n_ok utilizza soltanto i posti rimasti dopo aver selezionato gli errori.
            n_ok = min(len(idx_ok), n_per_class - n_ko)

            logger.debug(f"BEFORE KO n_ko {n_ko}")
            logger.debug(f"BEFORE KO  n_ok {n_ok}")
        else:
            # n_ok non può superare né il numero di errori 
            # disponibili né il limite per classe.
            n_ok = min(len(idx_ok), n_per_class)

            # n_ko utilizza soltanto i posti rimasti dopo aver selezionato gli ok.
            n_ko = min(len(idx_ko), n_per_class - n_ok)

            logger.debug(f"BEFORE OK n_ok {n_ok}")
            logger.debug(f"BEFORE OK n_ko {n_ko}")
            
        idx_ko_selected = idx_ko[:n_ko]
        idx_ok_selected = idx_ok[:n_ok]

        for i in (idx_ok_selected):

            # CNN:In questo modo se è un immagine che deriva dalla cnn
            # con 3 dimensioni ma la prima è uguale a 1 relativa a 1x28x28
            # toglie la prima dimensione ininfluente
            # LR: il codice rimane quasi identico al precedente
            img_to_save = X_test_[i]
            if img_to_save.ndim == 3 and img_to_save.shape[0] == 1:
                img_to_save = img_to_save.squeeze(0) # Rimuove la dimensione del canale se è 1

            # cast to int else json.write not work
            dict_curr_ok = {  "class": int(id_class),
                              "id_dataset": int(i),
                              "image": img_to_save.tolist() ,
                              "y_true": int(y_test_[i]) if not hasattr(y_test_[i], 'item') else y_test_[i].item(),
                              "y_pred": int(y_pred_[i]) if not hasattr(y_pred_[i], 'item') else y_pred_[i].item()
                            }

            list_images_ok.append(dict_curr_ok)

        for i in (idx_ko_selected):
            
            # CNN:In questo modo se è un immagine che deriva dalla cnn
            # con 3 dimensioni ma la prima è uguale a 1 relativa a 1x28x28
            # toglie la prima dimensione ininfluente
            # LR: il codice rimane quasi identico al precedente
            img_to_save = X_test_[i]
            if img_to_save.ndim == 3 and img_to_save.shape[0] == 1:
                img_to_save = img_to_save.squeeze(0) # Rimuove la dimensione del canale se è 1

            # cast to int else json.write not work
            dict_curr_ko = {  "class": int(id_class),
                              "id_dataset": int(i),
                              "image": img_to_save.tolist() ,
                              "y_true": int(y_test_[i]) if not hasattr(y_test_[i], 'item') else y_test_[i].item(),
                              "y_pred": int(y_pred_[i]) if not hasattr(y_pred_[i], 'item') else y_pred_[i].item()
                            }

            list_images_ko.append(dict_curr_ko)

        logger.debug(f" end classe {id_class}")

    return list_images_ok , list_images_ko


def run_check_class_ok_ko( logger, 
                            y_test_,
                            y_pred_ , 
                            list_class_,
                            dom_name_):
    """
        Run for check list images ko for both lr and cnn
        dom_name ('LR' , 'CNN') 
    """
    for id_class in list_class_:
        idx_ko_check = np.where((y_test_ == id_class) &
                             (y_test_ != y_pred_)
                             )[0]

        idx_ok_check = np.where((y_test_ == id_class) &
                                     (y_test_ == y_pred_)
                                     )[0]
        
        logger.debug(f"Classe {id_class}: {len(idx_ko_check)} errate {dom_name_}")
        logger.debug(f"Classe {id_class}: {len(idx_ok_check)} corrette {dom_name_}")
