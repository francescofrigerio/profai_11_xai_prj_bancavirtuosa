"""
     Here we load dataset and dataloader
"""
import json
from pathlib import Path

# pytorch
import torch

# preprocessd dataset with pytorch
from torchvision import datasets , transforms 
# computer vision with pytorch
from torch.utils.data import Subset , Dataset , DataLoader

from sklearn.model_selection import train_test_split
from logger import LOGGER
from config import CONFIG , IMAGENET_MEAN_TRANSFORM , IMAGENET_STD_TRANSFORM

class TransformedDataset(Dataset):
    """
        Here we Transform Data for DataLoader
        and wrap Dataset to save index of the images
    """
    def init_err_idx_lr(self, dataset, transform=None):
        """
            This version not store he correct idx
            parameter for lr model
        """
        self.dataset = dataset
        self.transform = transform
    def __init__(self, dataset, transform, original_indices=None):
        self.dataset = dataset
        self.transform = transform
        self.original_indices = original_indices

    def __len__(self):
        """
        """
        return len(self.dataset)

    def getitem_err_idx_lr(self, idx):
        """
            This version not return the correct idx
            parameter for lr model
        """
        image, label = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label , idx
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        if self.original_indices is not None:
            # get original idx from idx of subset
            original_idx = int(self.original_indices[idx])
        else:
            original_idx = idx

        return image, label, original_idx

def debug_iter_dataloader(dataloader_):
    """
        Number elements batch: 3
        Type batch: <class 'list'>
        Element 0: shape=torch.Size([64, 3, 128, 128]), dtype=torch.float32
        Element 1: shape=torch.Size([64]), dtype=torch.int64
        Element 2: shape=torch.Size([64]), dtype=torch.int64
    """
    batch = next(iter(dataloader_))
            
    print("Numero elementi nel batch:", len(batch))
    print("Tipo batch:", type(batch))
            
    for i, item in enumerate(batch):
        print(f"Elemento {i}: shape={item.shape}, dtype={item.dtype}")

def load_datasets(cfg=CONFIG,logger=LOGGER):
    """
       The transform (x - mean) / std (mean and std of ImageNet) is mandatory 
       to pass images at DenseNet model . 
       DenseNet expect pixels are passed  
       with this specify media/std to calc correctly logit and gradients.
       
       return idx on DataLoader
       For test_lr and test_cnn, idx corresponds to the original index from `test_base`, 
       because both use `test_base` directly, without a `Subset`.

       For train_lr, val_lr, `train_cnn`, and `val_cnn`, however, `idx` 
       is relative to the `Subset`. If you wish to save the original indices 
       for the training and validation sets as well, we will need 
       to handle this case separately.

       """
    transform_lr = transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                        # Transform to tensor
                                        transforms.ToTensor(),
                                      ])

    transform_cnn = transforms.Compose([ # Increase the size image
                                         transforms.Resize((128, 128)),
                                         # Convert to tensors
                                         transforms.ToTensor(),
                                        # expand channels 
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # Espande a 3 canali
                                        # normalize img
                                        transforms.Normalize(
                                        # mean=   [0.485, 0.456, 0.406],
                                        # std=[0.229, 0.224, 0.225]
                                        mean  =  IMAGENET_MEAN_TRANSFORM,
                                        std  =IMAGENET_STD_TRANSFORM
                                        )
                                        ])
    
    trainval_base = datasets.MNIST(root="./data",
                                    train=True,
                                    download=True,
                                    transform=None
                                )

    test_base = datasets.MNIST(root="./data",
                                train=False,
                                download=True,
                                transform=None
                            )

    # (dataset Torch standard function as MNIST)
    targets = trainval_base.targets
    
    # split size (ex. 6000 su 54000 = ~11.1%)
    val_fraction = cfg.VAL_SIZE / len(trainval_base)
    
    # get stratify indexes for train and validation
    # stratify=targets: scikit-learn set the % for every class
    #  (from '0' to '9') as same as train_indices and val_indices.
    train_idx, val_idx = train_test_split( range(len(trainval_base)),
                                                test_size=val_fraction,
                                                stratify=targets,
                                                random_state=cfg.SEED
                                         )
    
    # Regressione Logistic View
    # When we use Subset(trainval_base, train_idx), 
    # the idx parameter on the SubSet builder
    # refers to the position 
    # within the Subset, not the original index 
    # from trainval_base.
    train_lr = TransformedDataset(Subset(trainval_base, 
                                         train_idx), 
                                         transform_lr,
                                         train_idx)
    
    val_lr = TransformedDataset(Subset(trainval_base, 
                                       val_idx), 
                                       transform_lr,
                                       val_idx)
    
    test_lr = TransformedDataset(test_base, 
                                 transform_lr)

    # DenseNet View
    train_cnn = TransformedDataset(Subset(trainval_base, 
                                          train_idx), 
                                          transform_cnn,
                                          train_idx)
    
    val_cnn = TransformedDataset(Subset(trainval_base, 
                                        val_idx), 
                                        transform_cnn,
                                        val_idx )
    
    test_cnn = TransformedDataset(test_base, 
                                  transform_cnn)

    # DataLoader distinti
    datasets_dict = { "trainval": trainval_base,
                      "test": test_base,
                      "train_idx": train_idx,
                      "val_idx": val_idx,
                    }


    transform_cnn_dict = { "train": train_cnn,
                           "val": val_cnn,
                           "test": test_cnn  
                        }
    
    loaders_dict = { "lr": {
                "train": DataLoader(train_lr, batch_size=cfg.BATCH_SIZE, shuffle=True),
                "val": DataLoader(val_lr, batch_size=cfg.BATCH_SIZE, shuffle=False),
                "test": DataLoader(test_lr, batch_size=cfg.BATCH_SIZE, shuffle=False
                 )
            },
        "cnn": {
            "train": DataLoader(
                train_cnn, batch_size=cfg.BATCH_SIZE, shuffle=True
            ),
            "val": DataLoader(val_cnn, batch_size=cfg.BATCH_SIZE, shuffle=False),
            "test": DataLoader(test_cnn, batch_size=cfg.BATCH_SIZE, shuffle=False)
        }
    }

    return datasets_dict , loaders_dict , transform_cnn_dict

def data_to_input_lr( dataloader_ , 
                      channels , 
                      w , 
                      h ):
    """
        load data from dataset 
        to input struct for lr
    """
    # Alloc tensors 1, 28, 28
    X_input = torch.empty((len(dataloader_.dataset),  channels , w , h),
                dtype=torch.float32)
    y_input = torch.empty((len(dataloader_.dataset),),
                dtype=torch.int64)
    idx_list = []

    start = 0
    # for images, labels in dataloader_:
    for images, labels, idx in dataloader_:
        # images shape (64, 1, 28, 28) -> X
        # labels shape (64,)           -> y
        n = images.size(0)

        if tuple(images.shape[1:]) != (channels, w, h):
            raise ValueError(
                f"Forma immagini inattesa: {tuple(images.shape[1:])}; "
                f"attesa {(channels, w, h)}"
            )
        
        X_input[start:start+n].copy_(images)
        y_input[start:start+n].copy_(labels)
        idx_list.append(idx)

        start += n

    if start != len(dataloader_.dataset):
        raise ValueError(
            f"Sample Number: {start}; "
            f"Sample Expected: {len(dataloader_.dataset)}"
        )

    # flat the tensor only for baseline
    X_input = X_input.reshape(X_input
                              .shape[0], -1).numpy()
    y_input = y_input.numpy()
    idx = torch.cat(idx_list, dim=0)
    
    return X_input , y_input , idx

def get_images_labels(dataset_,flag_data_loader=True,idx_dataset=0,
                      cfg=CONFIG,logger=LOGGER):
    """
         load image from idx
    """
    if flag_data_loader:
        # once
        dataloader = DataLoader(dataset_, batch_size=32, shuffle=True)
        logger.debug(len(dataset_), "Images,", len(dataset_.class_to_idx), "identity")
        dataiter = iter(dataloader)
        # images and labels are pytorch tensors 
        images_, labels_ = next(dataiter)
    else:
        image, label = dataset_[idx_dataset]
        # images_ = [image]
        # labels_ = [label]
        # torch.stack() takes an input list of tensor and stack along new dimension.
        images_ = torch.stack([image])
        # builds a new tensor starting by list
        labels_ = torch.tensor([label])
    return images_, labels_


def save_dict_to_json(file_images , data_dict):
    """
        to save dict on json file
    """
    with open(file_images, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)

def load_dict_from_json(logger,path: Path):
    """
       to load dict from json file
    """
    with open(path, "r", encoding="utf-8") as f:
        list_load_dict = json.load(f)

    logger.info(f"Caricati {len(list_load_dict)} elementi con successo!")
    logger.info(f"Primo elemento:{list_load_dict[0]}")
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





