from config import CONFIG
from logger import LOGGER

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
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

def load_dataset_lr(cfg=CONFIG,logger=LOGGER):

    transform_lr = transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                    # Transform to tensor
                                    transforms.ToTensor(),
                                  ])

    mnist_trainvalset_lr = datasets.MNIST(root='./data', train=True, download=True, transform=transform_lr)
    mnist_testset_lr = datasets.MNIST(root='./data', train=False, download=False, transform=transform_lr)

    return mnist_trainvalset_lr,mnist_testset_lr

def load_dataset_cnn(cfg=CONFIG,logger=LOGGER):
    # Takes and views random images
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

def split_dataset(trainvalset ,train_size = 54000,val_size = 6000,
                  cfg=CONFIG,logger=LOGGER):


    # (dataset Torch standard function as MNIST)
    targets = trainvalset.targets

    # split size (ex. 6000 su 54000 = ~11.1%)
    val_fraction = val_size / (train_size + val_size)

    # get stratify indexes for train and validation
    # stratify=targets: scikit-learn set the % for every class
    #  (from '0' to '9') as same as train_indices and val_indices.
    train_indices, val_indices = train_test_split( range(len(trainvalset)),
                                                    test_size=val_fraction,
                                                    stratify=targets,
                                                    random_state=cfg.SEED
                                                   )

    # 3. Create PyTorch Subset  based on stratify indexes
    train_dataset = Subset(trainvalset, train_indices)
    val_dataset = Subset(trainvalset, val_indices)

    return train_dataset,val_dataset

def dataloader_to_tensor( dataloader_ , channels , w , h,
                         cfg=CONFIG,logger=LOGGER ):

    # Alloc tensors 1, 28, 28
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

def get_images_labels(dataset_,flag_data_loader=True,idx_dataset=0,
                      cfg=CONFIG,logger=LOGGER):

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


def load_data_baseline(cfg=CONFIG,logger=LOGGER):
    """
        load and process data for the baseline (Logistic Regression)
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

    # Load tensor for train and test
    X_train , y_train = dataloader_to_tensor(train_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_val , y_val = dataloader_to_tensor(val_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)
    X_test , y_test = dataloader_to_tensor(test_loader_lr,1, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)

    # flat the tensor
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    logger.debug(f" X_train flat {X_train.shape}")
    logger.debug(f" X_val flat {X_val.shape}")
    logger.debug(f" X_test flat {X_test.shape}")

    # Transform the tensors as numpy array for LogisticRegression
    X_train = X_train.numpy()
    X_val = X_val.numpy()
    X_test = X_test.numpy()
    return  X_train,X_val, X_test,  y_train ,y_val, y_test
