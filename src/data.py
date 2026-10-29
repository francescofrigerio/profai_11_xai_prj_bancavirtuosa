

import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# computer vision with pytorch
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
from torchvision.models import DenseNet121_Weights, densenet121

# preprocessd dataset with pytorch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset , Dataset

from sklearn.model_selection import train_test_split

from config import CONFIG
from logger import LOGGER

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENET_MEAN_TRANSFORM  =  [0.485, 0.456, 0.406]
IMAGENET_STD_TRANSFORM =   [0.229, 0.224, 0.225]



# ----------------------------------------------------------------------------------------
# DenseNet view
# ----------------------------------------------------------------------------------------
def image_net_stats(like):
    mean = torch.tensor(IMAGENET_MEAN, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    return mean, std


def normalize(x):
    """
        Image Net Normalize
    """
    mean, std = image_net_stats(x)
    return (x - mean) / std


def denormalize(x):
    """
        Rollback from image net Normalize value
    """
    mean, std = image_net_stats(x)
    return x * std + mean


def denormalize_np(x: np.ndarray) -> np.ndarray:
    """
        numpy version of denormalize, for the contract checks (no torch needed).
    """
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float64).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float64).reshape(1, 3, 1, 1)
    return np.asarray(x, dtype=np.float64) * std + mean

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        """
        image, label = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
def load_datasets(cfg=CONFIG,logger=LOGGER):
    """
       The transform (x - mean) / std (mean and std of ImageNet) is mandatory 
       to pass images at DenseNet model . 
       DenseNet expect pixels are passed  
       with this specify media/std to calc correctly logit and gradients.
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
    train_lr = TransformedDataset(Subset(trainval_base, train_idx), transform_lr)
    val_lr = TransformedDataset(Subset(trainval_base, val_idx), transform_lr)
    test_lr = TransformedDataset(test_base, transform_lr)

    # DenseNet View
    train_cnn = TransformedDataset(Subset(trainval_base, train_idx), transform_cnn)
    val_cnn = TransformedDataset(Subset(trainval_base, val_idx), transform_cnn)
    test_cnn = TransformedDataset(test_base, transform_cnn)

    # DataLoader distinti
    datasets_dict = { "trainval": trainval_base,
                      "test": test_base,
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

    return datasets_dict , loaders_dict

def data_to_input_lr( dataloader_ , 
                      channels , 
                      w , h,
                      cfg=CONFIG,
                      logger=LOGGER ):
    """
    """
    # Alloc tensors 1, 28, 28
    X = torch.empty((len(dataloader_.dataset),  channels , w , h),
                dtype=torch.float32)
    y = torch.empty((len(dataloader_.dataset),),
                dtype=torch.int64)
    start = 0
    for images, labels in dataloader_:
        # images shape (64, 1, 28, 28) -> X
        # labels shape (64,)           -> y
        n = images.size(0)

        if tuple(images.shape[1:]) != (channels, w, h):
            raise ValueError(
                f"Forma immagini inattesa: {tuple(images.shape[1:])}; "
                f"attesa {(channels, w, h)}"
            )
        
        X[start:start+n].copy_(images)
        y[start:start+n].copy_(labels)

        start += n

    if start != len(dataloader_.dataset):
        raise ValueError(
            f"Sample Number: {start}; "
            f"Sample Expected: {len(dataloader_.dataset)}"
        )

    # flat the tensor only for baseline
    X = X.reshape(X.shape[0], -1).numpy()
    y = y.numpy()
    
    return X , y

def get_images_labels(dataset_,flag_data_loader=True,idx_dataset=0,
                      cfg=CONFIG,logger=LOGGER):
    """
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

