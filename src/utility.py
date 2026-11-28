"""
    Commont utility for Normalization , Model Cnn and Logistic Regression
"""
import random
import math
import html
import json
import inspect
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from torchvision.transforms import GaussianBlur


from constants import IMAGENET_MEAN_TRANSFORM , IMAGENET_STD_TRANSFORM
from constants import IMAGENET_MEAN_NP , IMAGENET_STD_NP
from config import CONFIG


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def read_text_for_html(path):
    """
    It reads file .txt and return a text html,
    preserves (\n) and space.
    """
    with open(path, "r", encoding="utf-8") as f:
        txt_raw = f.read()
    
    # Run escape (<, >, &) to avoid failure on HTML
    txt_fine = html.escape(txt_raw)
    
    # Substitutr newline with <br> and use white-space: 
    # pre-wrap to maint newline and table align 
    content_html = f'<div style="white-space: pre-wrap; font-family: inherit;">{txt_fine}</div>'
    
    return content_html

def require(condition: bool, message: str) -> None:
    """
        Wrap raise condition
    """
    if not condition:
        raise ValueError(message)
            
def to_numpy(x):
    return torch.as_tensor(x).detach().cpu().numpy()

def save_quadrant_to_json( cfg,
                           logger, 
                           y_true, 
                           y_pred_lr, 
                           y_pred_cnn, 
                           idx,
                           classes, 
                           out_path ):
    """
        Save quadrants on json
        Quadrants: both_ok, lr_ko_cnn_ok, lr_ok_cnn_ko, both_ko (same split)

        return dict to plot the graph.
    """
    
    n_per_class = cfg.XAI_IMAGES_FOR_CLASS
    y_true, y_pred_lr, y_pred_cnn, idx = map(np.asarray, (y_true, y_pred_lr, y_pred_cnn, idx))
    
    lr_ok, cnn_ok = y_pred_lr == y_true, y_pred_cnn == y_true

    groups = { "both_ok": lr_ok & cnn_ok,
               "lr_ko_cnn_ok": ~lr_ok & cnn_ok,
               "lr_ok_cnn_ko": lr_ok & ~cnn_ok,
               "both_ko": ~lr_ok & ~cnn_ok }

    # random but reproducible, not the "first" ones
    rng = np.random.default_rng(cfg.SEED)  
    counts_info = {}     
    for name, mask in groups.items():
        items = []
        for c in classes:
            cand = np.where(mask & (y_true == c))[0]
            pick = np.sort(rng.choice(cand, 
                                      size=min(len(cand), n_per_class), 
                                      replace=False)) if len(cand) else []
            
            items += [{"class": int(c), 
                       "id_dataset": int(idx[p]), 
                       "y_true": int(y_true[p]),
                       "y_pred": int(y_pred_cnn[p]),          
                       "y_pred_lr": int(y_pred_lr[p])
                       } for p in pick]
        save_dict_to_json(out_path / f"images_{name}.json", items)
        logger.debug("%s: %d candidati, %d salvati", name, int(mask.sum()), len(items))

        counts_info[name] = { "candidati": int(mask.sum()),
                         "selezionati": len(items)
                       }
    return counts_info
        
def save_dict_to_json(file_images, data_dict):
    """
        to save dict on json file
    """
    file_images = Path(file_images)
    file_images.parent.mkdir(parents=True, exist_ok=True)
    with open(file_images, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)


def load_dict_from_json(logger,path: Path):
    """
       to load dict from json file
    """
    with open(path, "r", encoding="utf-8") as f:
        list_load_dict = json.load(f)

    logger.debug("Caricati %d elementi con successo!",len(list_load_dict))
    return list_load_dict


def load_one_image_by_class(logger,
                            json_path: str | Path) -> dict:
    """
        to load an image for every input class
    """

    with open(json_path, "r", encoding="utf-8") as f:
        list_load_dict = json.load(f)

    images_by_class = {}

    for item in list_load_dict:
        id_class = item["class"]

        if id_class not in images_by_class:
            # Warninh this format generate a dict
            # so in the callback loop we write
            # item = dict_images[id_class]
            images_by_class[id_class] = { "id_dataset": item["id_dataset"],
                                            "y_true": item["y_true"],
                                            "y_pred": item["y_pred"]
                                        }
    return images_by_class

def load_images_by_class( logger , 
                         path: Path, 
                         n_images_for_class: int = 1 ) -> dict:
    """
         to load N images for every input class    
    """

    with open(path, "r", encoding="utf-8") as f:
        list_load_dict = json.load(f)

    images_by_class = {}

    for item in list_load_dict:

        id_class = item["class"]

        if id_class not in images_by_class:
            images_by_class[id_class] = []

        # Avoid to select more images for the same classes
        if len(images_by_class[id_class]) >= n_images_for_class:
            continue

        # Warninh this format generate a list
        # so in the callback loop we write
        # item = list_images[id_class][0]
        images_by_class[id_class].append({ "id_dataset": item["id_dataset"],
                                            "y_true": item["y_true"],
                                            "y_pred": item["y_pred"]
                                        })

    return images_by_class

def save_quadrant_balance_to_json(cfg, 
                                  logger, 
                                  y_true, 
                                  y_pred_lr, 
                                  y_pred_cnn, 
                                  idx, 
                                  classes, 
                                  out_path ):
    """
        Save candidates for all quadrants with casual order (reproducibile)
        It maintain a large pool so it can select 
        n_images_for_class (5, 6, 7...) during load phase without reload json.
    """
    max_pool_per_class = max(cfg.XAI_IMAGES_FOR_CLASS, 20)
    y_true, y_pred_lr, y_pred_cnn, idx = map(np.asarray, (y_true, y_pred_lr, y_pred_cnn, idx))

    lr_ok, cnn_ok = (y_pred_lr == y_true), (y_pred_cnn == y_true)

    groups = {
        "both_ok": lr_ok & cnn_ok,
        "lr_ko_cnn_ok": ~lr_ok & cnn_ok,
        "lr_ok_cnn_ko": lr_ok & ~cnn_ok,
        "both_ko": ~lr_ok & ~cnn_ok,
    }

    rng = np.random.default_rng(cfg.SEED)
    counts_info = {}

    for name, mask in groups.items():
        items = []
        for c in classes:
            cand = np.where(mask & (y_true == c))[0]
            if len(cand):
                # casual order riproducibile instead of take alwai the first one
                shuffled = rng.permutation(cand)[:max_pool_per_class]
                items.extend([
                    {
                        "class": int(c),
                        "id_dataset": int(idx[p]),
                        "y_true": int(y_true[p]),
                        "y_pred": int(y_pred_cnn[p]),
                        "y_pred_lr": int(y_pred_lr[p]),
                    }
                    for p in shuffled
                ])

        save_dict_to_json(out_path / f"images_{name}.json", items)
        logger.debug("%s: %d candidati totali, %d salvati nel pool", name, int(mask.sum()), len(items))

        counts_info[name] = { "selected": len(items),       
                                "candidates": int(mask.sum()),
                                "save_pool": len(items),
                                "items": items,               # <-- lista vera e propria
                            }

    return counts_info


def load_images_balance_by_class(logger, 
                                 path: Path, 
                                 n_images_for_class: int = 1, 
                                 balance_total: bool = True) -> dict:
    """
        Load N images per class. If balance_total=True and some classes in a quadrant
        have fewer than N examples available (typical in error quadrants), the missing
        slots are redistributed among the other available classes to keep the total
        number of images balanced across the different quadrants.
    """
    with open(path, "r", encoding="utf-8") as f:
        list_load_dict = json.load(f)

    # Group by all candidates available for class
    pool_by_class = {}
    for item in list_load_dict:
        c = item["class"]
        pool_by_class.setdefault(c, []).append({
            "id_dataset": item["id_dataset"],
            "y_true": item["y_true"],
            "y_pred": item["y_pred"],
        })

    all_classes = sorted(pool_by_class.keys())
    target_total = n_images_for_class * len(all_classes)

    # Step 1: take until a n_images_for_class for every class
    images_by_class = {c: pool_by_class[c][:n_images_for_class] for c in all_classes}
    used_counts = {c: len(images_by_class[c]) for c in all_classes}

    # Passo 2: if loss image compare to target_total, get around to the class that has already errors
    if balance_total:
        current_total = sum(used_counts.values())
        while current_total < target_total:
            added_in_round = False
            for c in all_classes:
                if current_total >= target_total:
                    break
                if used_counts[c] < len(pool_by_class[c]):
                    images_by_class[c].append(pool_by_class[c][used_counts[c]])
                    used_counts[c] += 1
                    current_total += 1
                    added_in_round = True
            if not added_in_round:
                # There is not more candidates in this quadrants
                break

    tot_loaded = sum(len(v) for v in images_by_class.values())
    logger.info("Caricate %d/%d immagini da %s", tot_loaded, target_total, path.name)
    return images_by_class

def normalize_saliency_map(saliency_map: np.ndarray) -> np.ndarray:
    """
        Normalize a 2d array in the range [0, 1].
    """
    min_val = saliency_map.min()
    max_val = saliency_map.max()
    if max_val - min_val == 0:
        return np.zeros_like(saliency_map)
    return (saliency_map - min_val) / (max_val - min_val)

def denormalize_imagenet(img):
    """
    Reverse ImageNet normalization.

    Input:
        img: PyTorch tensor with shape (N, 3, H, W)

    Output:
        Denormalized tensor with values clipped to [0, 1].
    """

    mean = torch.as_tensor( IMAGENET_MEAN_TRANSFORM,
                            dtype=img.dtype,
                            device=img.device
                        ).view(1, 3, 1, 1)

    std = torch.as_tensor(IMAGENET_STD_TRANSFORM,
                        dtype=img.dtype,
                        device=img.device
                    ).view(1, 3, 1, 1)

    img_denorm = img * std + mean

    return img_denorm.clamp(0.0, 1.0)

def denormalize_imagenet_old(img_hwc_np):
    """
        Rollback from image net Normalize value
        This function Converts the image from ImageNet scale to a pixels view range [0, 1].
    """
    img_denorm = (img_hwc_np * IMAGENET_STD_TRANSFORM) + IMAGENET_MEAN_TRANSFORM
    # Clip to be sure the range will be strictly [0, 1] for LIME
    return np.clip(img_denorm, 0.0, 1.0)

def black_baseline(device):
    """
        black pixel after ImageNet normalization: (0 - mean) / std, shape (1,3,1)
        Substitute by black_normalized
        baseline with a shape compatible with (1, 3, k)
        It used by deletion curve
    """
    mean = torch.tensor(IMAGENET_MEAN_TRANSFORM, device=device)
    std = torch.tensor(IMAGENET_STD_TRANSFORM, device=device)
    return ((0 - mean) / std).view(1, 3, 1)

def make_blurred_baseline(input_img, sigma):
    """
          Create a blurred baseline with the same shape as input_img.
        input_img: tensor [1,C,H,W] is the shape used by the model.
        
        Sigma does not act on the importance ,map but rather the image used as a baseline.
        With a small sigma , the image remains close to the original;
        as sigma increased , details are attenuated and the images bevome more uniform.
        During deletion , the selected pixels are replaced by the valued from the blurred
        baseline at the same coordinates.
    """
    if sigma <= 0:
        raise ValueError("sigma deve essere maggiore di 0")

    h, w = input_img.shape[-2:]

    # Kernel even, near=6*sigma, limit to image size 
    max_kernel = min(h, w)
    if max_kernel % 2 == 0:
        max_kernel -= 1

    kernel_size = min(2 * math.ceil(3 * sigma) + 1, max_kernel)

    blur = GaussianBlur( kernel_size=kernel_size,
                        sigma=[sigma, sigma],
                        )

    return blur(input_img)

def normalize_imagenet(input_img):
    """
        Normalize for ImageNet from pixels view range
        
        this normalize is mandatory  
        before send the image to the model DenseNet. 
        This net expect pixels has distribuited  
        with this specific mean and std to get correctly the logits
        and the gradients.
       
    """
    mean = torch.tensor(IMAGENET_MEAN_TRANSFORM, dtype=input_img.dtype, device=input_img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD_TRANSFORM, dtype=input_img.dtype, device=input_img.device).view(1, 3, 1, 1)

    return (input_img - mean) / std

def normalize_imagenet_np(input_img):
    return (input_img - IMAGENET_MEAN_NP) / IMAGENET_STD_NP

def black_normalized(size: int, device):
    """
        The black of the ORIGINAL image, expressed in the normalized space.
    """
    return normalize_imagenet(torch.zeros(1, 3, size, size, device=device))

def denormalize_imagenet_np(x_data: np.ndarray) -> np.ndarray:
    """
        numpy version of denormalize image net, for the contract checks (no torch needed).
    """
    mean = np.asarray(IMAGENET_MEAN_TRANSFORM, dtype=np.float64).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD_TRANSFORM, dtype=np.float64).reshape(1, 3, 1, 1)
    return np.asarray(x_data, dtype=np.float64) * std + mean

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

        Limits:THRESHOLD must be consistent 
        with the pixel values in X_train. 
        If the images are normalized or standardized, 
        the threshold must be chosen accordingly.




        Threshold: cfg.MASK_THRESHOLD selecting a value such that the mask provides 
        a reasonable representation of the digit’s pixels. 
        Visually inspect a few masks using different values before 
        settling on one; there is no universal value.
        
        Dilation: I would use 0 as the primary setting, provided the function allows it.
        Dilation expands the mask, making it easier to achieve overlaps; 
        this can increase the IoU, even if the explanation extends into the background. 
        I would treat your current dilation setting as a sensitivity analysis rather 
        than a primary result.
        For instance, we could take the top 10% IoU results obtained with 
        a dilation of 0 and recalculate them using a dilation of 1 
        to see how much the outcome depends on the mask boundaries. 
        
        If you are uncertain about the threshold, you could also compare 
        a second threshold—determined via visual inspection—while keeping 
        the dilation constant; changing both parameters simultaneously 
        makes it difficult to determine which one is responsible for any observed differences.

        PAY ATTENTION:To ensures that comparisons between techniques remain valid: every comparison 
        must use the same threshold and dilation for the same image.

       We will Use the top 10% IoU—with a fixed threshold and no dilation—as 
       our primary metric, followed by a brief sensitivity analysis using 
       our current dilation setting. The IoU function itself can remain 
       independent of these parameters, as it receives the mask only after it has already 
       been constructed. 
       The digit_mask MASK_DILATION=0 have to interpret as no dilation.
    """
    mask = np.asarray(input_img28) > threshold

    # expands the mask, slightly enlarging the identified area.
    # The dilation parameter determines how many times 
    # this expansion is performed. If it is set to zero, 
    # the mask remains unchanged.
    return binary_dilation(mask, 
                           structure=np.ones((3, 3), bool), 
                           iterations=dilation
                           ) if dilation else mask

def img28(pos,X_data,size ):
    """
        flat vector (784,) -> image (28, 28) for mask and imshow
    """
    return X_data[pos].reshape(size, size)

def map_to_28_old(m, size=CONFIG.IMAGE_SIZE):
    """
        any map (4x4, 8x8, 128x128) -> size x size, to compare with the digit mask
        Substitude by to_grid28
    """
    t = torch.as_tensor(np.asarray(m), dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(size, size), mode="area")[0, 0].numpy()

def to_grid28(map2d, size, side=CONFIG.IMAGE_SIZE):
    """
        This Function convert XAI 2dmap  (4x4, 8x8, 128x128) -> size x size, 
        to compare with the digit mask
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
                            # X_test_ , 
                            y_test_,
                            y_pred_ , 
                            idx_dataset_,
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
   
    list_images_ok = []
    list_images_ko = []

    for id_class in list_class_:
        logger.debug(f" start classe {id_class}")

        class_mask = y_test_ == id_class
        # logger.debug(f" total_class_mask {class_mask}")

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

        # stabilisco quanti errori e quante
        # immagini corrette possiamo selezionare
        if flag_before_ko is True:
            # n_ko non può superare né il numero di errori 
            # disponibili né il limite per classe.
            n_ko = min(len(idx_ko), n_per_class)

            # n_ok utilizza soltanto i posti rimasti dopo aver selezionato gli errori.
            n_ok = min(len(idx_ok), n_per_class - n_ko)

        else:
            # n_ok non può superare né il numero di errori 
            # disponibili né il limite per classe.
            n_ok = min(len(idx_ok), n_per_class)

            # n_ko utilizza soltanto i posti rimasti dopo aver selezionato gli ok.
            n_ko = min(len(idx_ko), n_per_class - n_ok)

        idx_ko_selected = idx_ko[:n_ko]
        idx_ok_selected = idx_ok[:n_ok]

        for i in (idx_ok_selected):

            # CNN:In questo modo se è un immagine che deriva dalla cnn
            # con 3 dimensioni ma la prima è uguale a 1 relativa a 1x28x28
            # toglie la prima dimensione ininfluente
            # LR: il codice rimane quasi identico al precedente
            # img_to_save = X_test_[i]
            # if img_to_save.ndim == 3 and img_to_save.shape[0] == 1:
            #     img_to_save = img_to_save.squeeze(0) # Rimuove la dimensione del canale se è 1

            # cast to int else json.write not work
            # i=Posizione nella sequenza delle predizioni
            # idx_dataset_[i] Indice dell'immagine nel Dataset
            dict_curr_ok = {  "class": int(id_class),
                              # "id_dataset": int(i), 
                              "id_dataset": int(idx_dataset_[i]),
                              # "image": img_to_save.tolist() ,
                              "y_true": int(y_test_[i]) if not hasattr(y_test_[i], 'item') else y_test_[i].item(),
                              "y_pred": int(y_pred_[i]) if not hasattr(y_pred_[i], 'item') else y_pred_[i].item()
                            }

            list_images_ok.append(dict_curr_ok)

        for i in (idx_ko_selected):
            
            # CNN:In questo modo se è un immagine che deriva dalla cnn
            # con 3 dimensioni ma la prima è uguale a 1 relativa a 1x28x28
            # toglie la prima dimensione ininfluente
            # LR: il codice rimane quasi identico al precedente
            # img_to_save = X_test_[i]
            # if img_to_save.ndim == 3 and img_to_save.shape[0] == 1:
            #     img_to_save = img_to_save.squeeze(0) # Rimuove la dimensione del canale se è 1

            # cast to int else json.write not work
            dict_curr_ko = {  "class": int(id_class),
                              # "id_dataset": int(i),
                             "id_dataset": int(idx_dataset_[i]),
                              # "image": img_to_save.tolist() ,
                              "y_true": int(y_test_[i]) if not hasattr(y_test_[i], 'item') else y_test_[i].item(),
                              "y_pred": int(y_pred_[i]) if not hasattr(y_pred_[i], 'item') else y_pred_[i].item()
                            }

            list_images_ko.append(dict_curr_ko)

        logger.debug(f" end classe {id_class}")

    return list_images_ok , list_images_ko


def create_and_save_images_lists(logger, 
                                 # X_test,
                                 y_true, 
                                 y_pred, 
                                 idx_dataset,
                                 list_class, 
                                 out_path,
                                 file_ok, 
                                 file_ko,
                                 n_images=50, 
                                 flag_before_ko=True):
    """
        Create list_images_ok / list_images_ko and save them on 2 json files
        (used for both LR and CNN)
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    
    logger.info(f"idx_dataset type before conversion: {type(idx_dataset)}")
    logger.info(f"idx_dataset first element: {idx_dataset[0]}")

    if y_true.ndim != 1 or y_pred.shape != y_true.shape or idx_dataset.shape != y_true.shape:
        raise ValueError(f"Attesi array 1D della stessa lunghezza, ottenuti "
                         f"{y_true.shape}, {y_pred.shape}, {idx_dataset.shape}")

    
    idx_dataset = np.asarray(idx_dataset)

    list_ok, list_ko = run_create_list_images(logger,
                                              # X_test,          
                                              y_true,
                                              y_pred,
                                              idx_dataset,
                                              list_class,
                                              n_images_=n_images,
                                              flag_before_ko=flag_before_ko)

    save_dict_to_json(out_path / file_ok, list_ok)
    save_dict_to_json(out_path / file_ko, list_ko)
    
    logger.debug(f"Salvati {len(list_ok)} OK in {file_ok} e {len(list_ko)} KO in {file_ko}")
    return list_ok, list_ko

def run_check_class_ok_ko( logger, 
                            y_test_,
                            y_pred_ , 
                            list_class_,
                            dom_name_):
    """
        Run for check list images ko for both lr and cnn
        dom_name ('LR' , 'CNN') 
    """
    idx_ko_check , idx_ok_check = 0,0
    for id_class in list_class_:
        idx_ko_check = np.where((y_test_ == id_class) &
                             (y_test_ != y_pred_)
                             )[0]

        idx_ok_check = np.where((y_test_ == id_class) &
                                     (y_test_ == y_pred_)
                                     )[0]
        
        logger.debug(f"Classe {id_class}: {len(idx_ko_check)} errate {dom_name_}")
        logger.debug(f"Classe {id_class}: {len(idx_ok_check)} corrette {dom_name_}")

    return idx_ok_check , idx_ko_check

def verify_output_files_exist(out_path: Path, cfg) -> bool:
    """
        This Function verifies that all need output files real exist
    """
    files_to_check = [
        out_path / cfg.FILE_IMG_LR_OK,
        out_path / cfg.FILE_IMG_LR_KO,
        out_path / cfg.FILE_IMG_CNN_OK,
        out_path / cfg.FILE_IMG_CNN_KO
    ]
    
    # Restituisce True solo se .exists() è True per TUTTI i file nella lista
    return all(f.exists() for f in files_to_check)

def print_inspect_function(logger,f_imported_defined):
    """
        logger.info("Funzione: %s", make_gradcam_callback)
        logger.info("Modulo: %s", make_gradcam_callback.__module__)logger.info("File: %s", inspect.getfile(make_gradcam_callback))
        logger.info("Firma: %s", inspect.signature(make_gradcam_callback))
        logger.info("Sorgente make_gradcam_callback:\n%s",inspect.getsource(make_gradcam_callback))    
    """
    logger.info("Funzione: %s", f_imported_defined)
    logger.info("Modulo: %s", f_imported_defined.__module__)
    logger.info("File: %s", inspect.getfile(f_imported_defined))
    logger.info("Firma: %s", inspect.signature(f_imported_defined))
    logger.info("Sorgente f_imported_defined:\n%s",inspect.getsource(f_imported_defined))
    
    
def stroke_thickness_score(position, X_data, cfg):
    """
         detect stroke_thickness_score
         for select thin thick images
    """
    image = img28(position, X_data, size=cfg.IMAGE_SIZE)
    mask = digit_mask(image, cfg.MASK_THRESHOLD, dilation=0).astype(bool)

    skeleton = skeletonize(mask)
    if not mask.any() or not skeleton.any():
        return None

    distances = distance_transform_edt(mask)
    # thickness estimate: twice mean distance
    # of skeleton pixels from border of mask 
    return float(2 * distances[skeleton].mean())

def validate_maps_json_positions(json_path, dataset,idx_to_pos_lr=None):
    """
        Check Alignment between dataset and json maps
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    idx_to_pos_cnn = dataset.idx_dataset_position()

    for row in data.get("images", []):
        index = int(row["index"])
        pos_json = int(row["pos"])

        if index not in idx_to_pos_cnn:
            raise KeyError(f"index={index} del JSON non presente nel dataset")

        pos_expected = int(idx_to_pos_cnn[index])
        if pos_json != pos_expected:
            raise ValueError(f"Posizione non coerente per index={index}: "
                            f"JSON pos={pos_json}, dataset pos={pos_expected}"
                            )

        _, _, dataset_index = dataset[pos_expected]
        if int(dataset_index) != index:
            raise ValueError(f"Il dataset alla posizione {pos_expected} restituisce "
                            f"index={dataset_index}, atteso {index}"
                            )
                          
        # if idx_to_pos_lr is None:
        #    if pos_json not in idx_to_pos_lr:
        #        raise KeyError(f"index={pos_json} non trovato nei dati LR")
        if idx_to_pos_lr is not None:
            if index not in idx_to_pos_lr:
                raise KeyError(f"index={index} del JSON non presente nei dati LR")

            pos_lr = int(idx_to_pos_lr[index])
            # The comparison pos_json / pos_lr is valid 
            # because the CNN and LR datasets are constructed using the same val_idx, 
            # and the validation loaders have `shuffle=False`. 
            # It is a useful check nonetheless: should the split or ordering change in the future, 
            # it will alert , rather than silently associating the wrong image.
            if pos_json != pos_lr:
                raise ValueError(f"Posizione non coerente per index={index}: "
                                f"JSON/dataset CNN pos={pos_json}, dati LR pos={pos_lr}"
                                )
                

    return True
    

def select_thin_thick_from_maps_json( json_path,
                                      X_data,
                                      cfg,
                                      quantiles=(0.2, 0.8)):
   
    """
        select_thin_thick_examples from maps json file
        select images in which the lines are particularly thin or thick.
        
    """
    
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images = data.get("images", [])

    if not images:
        raise ValueError(f"Nessuna immagine trovata in {json_path}")

    selected = {}

    # group by record for true label, using only one from json
    for digit in sorted({int(row["true"]) for row in images}):
        candidates = [ row for row in images
                            if int(row["true"]) == digit
                    ]

        scored = []
        for row in candidates:
            position = int(row["pos"])
            score = stroke_thickness_score(position, X_data, cfg)

            if score is not None:
                scored.append((row, float(score)))

        if len(scored) < 2:
            continue

        scores = np.array([score for _, score in scored])
        thin_target, thick_target = np.quantile(scores, quantiles)

        thin_i = int(np.argmin(np.abs(scores - thin_target)))
        thick_i = int(np.argmin(np.abs(scores - thick_target)))

        thin_row, thin_score = scored[thin_i]
        thick_row, thick_score = scored[thick_i]

        selected[digit] = { "thin": {
                            "index": int(thin_row["index"]),
                            "pos": int(thin_row["pos"]),
                            "score": thin_score,
                          },
                          "thick": {
                              "index": int(thick_row["index"]),
                              "pos": int(thick_row["pos"]),
                              "score": thick_score,
                          },
                        }

    return selected


def select_thin_thick_examples(X_data, y_data, cfg, quantiles=(0.2, 0.8)):
    """
        select_thin_thick_examples from X_Data
        select images in which the lines are particularly thin or thick.
        
    """
    labels = np.asarray(y_data).reshape(-1)
    selected = {}
    
    for digit in np.unique(labels):
        candidates = np.flatnonzero(labels == digit)
        scored = []

        for position in candidates:
            score = stroke_thickness_score(position, X_data, cfg)
            if score is not None:
                scored.append((int(position), score))

        if len(scored) < 2:
            continue

        positions = np.array([item[0] for item in scored])
        scores = np.array([item[1] for item in scored])

        thin_target, thick_target = np.quantile(scores, quantiles)

        thin_idx = np.argmin(np.abs(scores - thin_target))
        thick_idx = np.argmin(np.abs(scores - thick_target))

        selected[int(digit)] = { "thin": int(positions[thin_idx]),
                                "thick": int(positions[thick_idx]),
                                "thin_score": float(scores[thin_idx]),
                                "thick_score": float(scores[thick_idx]),
                                }

    return selected
    
    

