
"""
    Commont utility for Model Cnn and Logistic Regression
"""

import numpy as np
import torch

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
    else:
        # n_per_class = n_images // len(list_class)
        n_per_class = max(1, n_images_ // len(list_class_))

    # Maschere su tutte le classi
    correct_mask = y_pred_ == y_test_
    wrong_mask = y_pred_ != y_test_
    logger.info(f" correct_mask {correct_mask} wrong_mask {wrong_mask}")

    list_images_lr_ok = []
    list_images_lr_ko = []

    logger.info(f" y_test_ {y_test_}")
    logger.info(f" y_test_ {type(y_test_)}")

    for id_class in list_class_:
        logger.info(f" start classe {id_class}")

        class_mask = y_test_ == id_class
        logger.info(f" total_class_mask {class_mask}")

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

        logger.info(f"Indici KO di controllo: {idx_ko[:10]}")
        logger.info(f"Indici OK di controllo: {idx_ok[:10]}")

        # stabilisco quanti errori e quante
        # immagini corrette possiamo selezionare
        if flag_before_ko is True:
            # n_ko non può superare né il numero di errori 
            # disponibili né il limite per classe.
            n_ko = min(len(idx_ko), n_per_class)

            # n_ok utilizza soltanto i posti rimasti dopo aver selezionato gli errori.
            n_ok = min(len(idx_ok), n_per_class - n_ko)

            logger.info(f"BEFORE KO n_ko {n_ko}")
            logger.info(f"BEFORE KO  n_ok {n_ok}")
        else:
            # n_ok non può superare né il numero di errori 
            # disponibili né il limite per classe.
            n_ok = min(len(idx_ok), n_per_class)

            # n_ko utilizza soltanto i posti rimasti dopo aver selezionato gli ok.
            n_ko = min(len(idx_ko), n_per_class - n_ok)

            logger.info(f"BEFORE OK n_ok {n_ok}")
            logger.info(f"BEFORE OK n_ko {n_ko}")
            
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

            # Adatto il dizionario al formato json
            dict_curr_ok = { "image": img_to_save.tolist() ,
                              "y_true": int(y_test_[i]) if not hasattr(y_test_[i], 'item') else y_test_[i].item(),
                              "y_pred": int(y_pred_[i]) if not hasattr(y_pred_[i], 'item') else y_pred_[i].item()
                            }

            list_images_lr_ok.append(dict_curr_ok)

        for i in (idx_ko_selected):
            
            # CNN:In questo modo se è un immagine che deriva dalla cnn
            # con 3 dimensioni ma la prima è uguale a 1 relativa a 1x28x28
            # toglie la prima dimensione ininfluente
            # LR: il codice rimane quasi identico al precedente
            img_to_save = X_test_[i]
            if img_to_save.ndim == 3 and img_to_save.shape[0] == 1:
                img_to_save = img_to_save.squeeze(0) # Rimuove la dimensione del canale se è 1

            dict_curr_ko = { "image": img_to_save.tolist() ,
                              "y_true": int(y_test_[i]) if not hasattr(y_test_[i], 'item') else y_test_[i].item(),
                              "y_pred": int(y_pred_[i]) if not hasattr(y_pred_[i], 'item') else y_pred_[i].item()
                            }

            list_images_lr_ko.append(dict_curr_ko)

        logger.info(f" end classe {id_class}")

    return list_images_lr_ok , list_images_lr_ko


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
        
        logger.info(f"Classe {id_class}: {len(idx_ko_check)} errate {dom_name_}")
        logger.info(f"Classe {id_class}: {len(idx_ok_check)} corrette {dom_name_}")
