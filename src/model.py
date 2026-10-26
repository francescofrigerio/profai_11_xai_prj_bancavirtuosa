"""
    DenseNet121 for MNIST: ImageNet backbone (frozen by default) + a new Linear head.

    Two flags decide the behaviour (nothing switches by itself):
    1. FLAG_BATCHNORM_EVAL: the BatchNorm of the backbone stay in eval mode during training 
    (only meaningful when FREEZE_BACKBONE=True). 
    Test : validation accuracy 0.9700 with FLAG_BATCHNORM_EVAL=True  
    Test : validation accuracy 0.9400 with FLAG_BATCHNORM_EVAL=False  
           (the running statistics drift away from the ImageNet ones).

    2. FLAG_RELU_INPLACE: True keeps torchvision's in-place ReLU (less VRAM), False 
    replaces them with a patched model. 
    Test: Grad-CAM, IG, Saliency and SHAP give the same maps in both cases.
"""
from torchvision.models import DenseNet121_Weights, densenet121
# import torchvision.models as models

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def check_model_params(cfg,logger,model):
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

# Building
def build_densenet(cfg,logger):
    """
         Build Model , optimizer and define loss Function
    """
    # standard code
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    # to preserve memory VRAM
    set_batchnorm_eval(model)

    for param in model.parameters():
        param.requires_grad = False

    # Modifica dell'ultimo layer per il numero corretto di classi
    # num_classes = len(trainval_dataset_cnn.class_to_idx)
    model.classifier = nn.Linear(model.classifier.in_features, cfg.NUM_CLASSES)
    check_model_params(cfg,logger,model)

    return model

def count_inplace_relu(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.ReLU) and m.inplace)

def validate_layers(model: nn.Module, cfg) -> None:
    """
        The Grad-CAM layers named in the config must exist in the model.
        This is the layer where GRADCAM run 
    """
    names = dict(model.named_modules())
    for param in ("GRADCAM_LAYER", "GRADCAM_ALT_LAYER"):
        if getattr(cfg, param) not in names:
            raise ValueError(f"{param}={getattr(cfg, param)!r} is not a module of the model")


# Training mode
def freeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad_(False)


def set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

def enter_train_mode(model: nn.Module, cfg) -> None:
    """
        Single entry point for training mode, called at the start of EVERY epoch (a validation
        pass in between would otherwise put the BatchNorm back in train mode).
    """
    model.train()
    if cfg.FREEZE_BACKBONE and cfg.FLAG_BATCHNORM_EVAL:
        model.features.apply(set_batchnorm_eval)

def trainable_parameters(model: nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]

def define_loss_function():
    """
        return the correct function
        for multiclass classifier
    """
    return nn.CrossEntropyLoss()

def make_optimizer(model: nn.Module, cfg):
    """
        Head only (frozen backbone) or the whole network, each with its own learning rate.
    """
    if cfg.FREEZE_BACKBONE:
        return torch.optim.Adam(trainable_parameters(model), lr=cfg.LEARNING_RATE_HEAD)
    return torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE_FULL)


