"""
    Global configuration of the Banca Virtuosa XAI project.

    Modules receive the config as an argument; only the notebook / run_main scripts import CONFIG.

    Changes with respect to the design document v1 (see docs/design_errata.md):
    - USE_SURROGATE removed (D03-bis): no surrogate model in this project.
    - MIN_VAL_ACCURACY = 0.95 (E01).
    - IMAGE_SIZE is now the MNIST side (28); the DenseNet input side is DENSENET_SIZE (E19).
    - Early stopping removed (E20): the number of epochs is fixed, final weights are kept.
    - FLAG_BATCHNORM_EVAL and FLAG_RELU_INPLACE added (E21).
    - SHAP_NUM_SAMPLES added (E18): the gradient samples of Gradient SHAP were an unnamed magic number;
    fixed at 100 after phase 0 (seed-to-seed Spearman 0.78 against 0.80 at 200 samples, half the time).
    - LIME_BACKEND / SHAP_BACKEND removed: LIME and SHAP use the `lime` and `shap` packages (E24).
    - Validation extended to backend names, ranges and grid/window consistency (E16).

    Not validated here: GRADCAM_LAYER / GRADCAM_ALT_LAYER must exist in the model; that check
    needs the model and lives in model.py.
"""

import numpy as np
from dataclasses import dataclass

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class ConfigConstantsDefault:
    """
        Global config. Replaces magic numbers with descriptive names.
    """
    RUN_MODE: str = "train"             # "train" | "load"
    RUN_PHASE: int = 9                  # 1 | 2 | 3 ..... | 9 
    RUN_PHASE_1: int = 1                  
    RUN_PHASE_2: int = 2                  
    RUN_PHASE_3: int = 3                                    
    LR_EXPECTED_VAL_ACC = (0.90, 0.94)    # design: about 0.92 (phase 0: 0.9222)
    GAP_EXPECTED = (0.03, 0.07)           # design: 4-6 points (phase 0: 4.78 on validation)
    
    # Reproducibility and environment
    SEED: int = 42
    DEVICE: str = "auto"                 # "auto" -> cuda if available, else cpu (warn on cpu)

    # Data
    IMAGE_SIZE: int = 28                 # MNIST side (1x28x28): input of the LR baseline
    DENSENET_SIZE: int = 128             # DenseNet input side (3x128x128)
    NUM_CLASSES: int = 10
    VAL_SIZE: int = 6000                 # stratified split of the 60k train set
    DATA_ROOT: str = "mnist_raw"         # where torchvision downloads MNIST (not a module name)
    NUM_DEBUG_IMAGES: int = 10           # fixed validation images (one per class)

    # Model and training
    BACKBONE_WEIGHTS: str = "IMAGENET1K_V1"
    FREEZE_BACKBONE: bool = True         # True -> head only, BatchNorm of backbone in eval
    MIN_VAL_ACCURACY: float = 0.95       # warning (not stop) if final validation accuracy is below
    NUM_EPOCHS: int = 5
    BATCH_SIZE: int = 64                 # to confirm in phase zero
    OPTIMIZER: str = "adam"
    LEARNING_RATE_HEAD: float = 1e-3     # used when FREEZE_BACKBONE=True
    LEARNING_RATE_FULL: float = 5e-5     # used when FREEZE_BACKBONE=False

    CHECKPOINT_DIR: str = ROOT_DIR / "checkpoints/"
    CHECKPOINT_PATH: str = ROOT_DIR / "checkpoints/densenet121_mnist.pt"
    # BatchNorm of the backbone in eval during training. Effective only when FREEZE_BACKBONE=True
    # (otherwise the backbone trains normally). False reproduces plain model.train().
    FLAG_BATCHNORM_EVAL: bool = True
    # True: keep torchvision's in-place ReLU (less VRAM). False: patch them (non in-place).
    # Nothing switches automatically: if an explanation technique needs False, say so via config.
    FLAG_RELU_INPLACE: bool = True

    # Phase 2 Parameters
    # NUM_IMAGES_FOR_CLASS: int = 100   # to confirm in phase two
    NUM_IMAGES_FOR_CLASS: int = 50   # to confirm in phase two
    NUM_IMAGES_MIN_OK: int = 10       # to confirm in phase two
    
    # Grad-CAM
    GRADCAM_LAYER: str = "features.norm5"
    GRADCAM_ALT_LAYER: str = "features.denseblock3"   # for the comparison figure
    GRADCAM_DEBUG_EPOCHS: str = "first_last"

    # Logistic regression baseline
    LR_PENALTY: str = "l2"
    LR_SOLVER: str = "lbfgs"
    LR_C: float = 1.0
    LR_MAX_ITER: int = 1000              # check ConvergenceWarning in phase zero

    # XAI techniques
    # LIME and SHAP always use the reference packages `lime` and `shap` (E24): no backend switch.
    IG_STEPS: int = 200 
    IG_BATCH_SIZE : int = 50                # phase 0, check 4: 3.0 % completeness error (50 steps: 20 %)
    OCCLUSION_WINDOW: int = 16
    OCCLUSION_STRIDE: int = 8
    LIME_NUM_SAMPLES: int = 100 # or 1000
    LIME_BATCH_SIZE : int = 64 
    # LIME_METHOD: str = "quickshift"
    LIME_METHOD: str = "grid"
    # 8x8 grid segmentation (to check visually)
    LIME_GRID_CELLS: int = 8             
    SHAP_NUM_BACKGROUND: int = 50
    # gradient samples of Gradient SHAP;
    SHAP_NUM_SAMPLES: int = 100     
    SHAP_BATCH_SIZE : int = 50           

    # Sampling
    GALLERY_SIZE: int = 24
    N_QUANT: int = 100                   # quantitative sample; fixed after timing in phase zero

    # Comparison
    # Spearman and IoU
    COMPUTE_AGREEMENT: bool = True  
    # could be "absolute" or "positive"     
    SIGN_MODE: str = "absolute"   
    # could be "digit" or "full"      
    MASK_MODE: str = "digit"   
    # intensity above which a pixel belongs to the digit          
    MASK_THRESHOLD: float = 0.1          
    MASK_DILATION: int = 1
    TOPK_FRACTION: float = 0.2
    COMPUTE_FAITHFULNESS: bool = False   # deletion curves
    DELETION_STEPS: int = 10

    # Execution and outputs ONLY on Notebook
    OUTPUT_DIR: str = ROOT_DIR / "outputs"

    # found and confirmed by visual control
    CHECK_CLASSES: tuple[int, ...] = (2, 3, 5, 8)
    FILE_IMG_LR_OK: str = "images_lr_ok.json" 
    FILE_IMG_LR_KO: str = "images_lr_ko.json" 
    FILE_IMG_CNN_OK: str = "images_cnn_ok.json" 
    FILE_IMG_CNN_OK: str = "images_cnn_ko.json" 

    LIST_IMAGES_SEL_LR_OK: str = "LR_OK" 
    LIST_IMAGES_SEL_LR_KO: str = "LR_KO"
    LIST_IMAGES_SEL_CNN_OK: str = "CNN_OK"
    LIST_IMAGES_SEL_CNN_OK: str = "CNN_KO"
    LIST_IMAGES_SEL: str = "LR_KO"

    IG_COMPLETENESS_MAX = 0.05
    RELU_MAX_REL_DIFF = 1e-3
    RELU_MAX_REF_DIFF = 1e-3

    IN_NOTEBOOK = "ipykernel" in sys.modules

    THRESHOLDS = { 
                   "backend_max_seconds_per_image": 15.0,    # LIME / SHAP backend accepted if <= (T4)
                   "quant_stage_max_minutes": 20.0,          # time budget of the quantitative stage
                   "run_all_load_max_minutes": 30.0,         # "Esegui tutto" with RUN_MODE=load
                   "run_all_train_max_minutes": 45.0,        # "Esegui tutto" with RUN_MODE=train
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

    def __post_init__(self):
        # Validate combinations and ranges that would otherwise fail late or silently.
        # An invalid value always raises: there is no automatic fallback (principle 3).
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        def one_of(name: str, allowed: tuple) -> None:
            require(getattr(self, name) in allowed,
                    f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

        # Choices
        one_of("DEVICE", ("auto", "cuda", "cpu"))
        one_of("RUN_MODE", ("auto", "train", "load"))
        one_of("RUN_PHASE", (1,2,3,9))
        one_of("SIGN_MODE", ("absolute", "positive"))
        one_of("MASK_MODE", ("digit", "full"))
        one_of("OPTIMIZER", ("adam",))
        one_of("GRADCAM_DEBUG_EPOCHS", ("first_last",))
        one_of("LR_PENALTY", ("l1", "l2"))
        one_of("LR_SOLVER", ("lbfgs", "saga"))

        # Logistic regression: penalty/solver combination
        require(not (self.LR_PENALTY != "l2" and self.LR_SOLVER == "lbfgs"),
                "lbfgs supports only the l2 penalty: use saga for l1")

        # Positive integers
        for name in ("NUM_CLASSES", "NUM_DEBUG_IMAGES", "NUM_EPOCHS", "BATCH_SIZE",
                     "LR_MAX_ITER", "IG_STEPS", "OCCLUSION_WINDOW", "OCCLUSION_STRIDE",
                     "LIME_NUM_SAMPLES", "LIME_GRID_CELLS", "SHAP_NUM_BACKGROUND",
                     "SHAP_NUM_SAMPLES", "GALLERY_SIZE", "N_QUANT", "DELETION_STEPS"):
            require(getattr(self, name) > 0, f"{name} must be > 0")
        require(self.MASK_DILATION >= 0, "MASK_DILATION must be >= 0")

        # Positive floats and ranges
        for name in ("LEARNING_RATE_HEAD", "LEARNING_RATE_FULL", "LR_C"):
            require(getattr(self, name) > 0, f"{name} must be > 0")
        require(0.0 < self.MIN_VAL_ACCURACY <= 1.0, "MIN_VAL_ACCURACY must be in (0, 1]")
        require(0.0 < self.TOPK_FRACTION <= 1.0, "TOPK_FRACTION must be in (0, 1]")
        require(0.0 <= self.MASK_THRESHOLD < 1.0, "MASK_THRESHOLD must be in [0, 1)")

        # Geometry: DenseNet downsamples by 32, so smaller images collapse the last layer
        require(self.IMAGE_SIZE == 28, "IMAGE_SIZE is the MNIST side and must be 28")
        require(self.DENSENET_SIZE >= 32, "DENSENET_SIZE must be >= 32 (DenseNet stride is 32)")
        require(self.OCCLUSION_WINDOW <= self.DENSENET_SIZE,
                "OCCLUSION_WINDOW cannot exceed DENSENET_SIZE")
        require(self.DENSENET_SIZE % self.LIME_GRID_CELLS == 0,
                "DENSENET_SIZE must be a multiple of LIME_GRID_CELLS (equal grid cells)")

        # Flags
        for name in ("FREEZE_BACKBONE", "FLAG_BATCHNORM_EVAL", "FLAG_RELU_INPLACE"):
            require(isinstance(getattr(self, name), bool), f"{name} must be a bool")

        # Data split and sampling
        require(0 < self.VAL_SIZE < 60000, "VAL_SIZE must be in (0, 60000)")
        require(self.NUM_DEBUG_IMAGES == self.NUM_CLASSES,
                "NUM_DEBUG_IMAGES must equal NUM_CLASSES (one fixed image per class)")
        require(self.N_QUANT % self.NUM_CLASSES == 0,
                "N_QUANT must be a multiple of NUM_CLASSES (stratified by predicted class)")
    
CONFIG = ConfigConstantsDefault()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Statistiche standard ImageNet
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_MEAN_TRANSFORM  =  [0.485, 0.456, 0.406]
IMAGENET_STD_TRANSFORM =   [0.229, 0.224, 0.225]

TITLES = {"gradcam": "Grad-CAM", "integrated_gradients": "Integrated\nGradients", "occlusion": "Occlusion",
          "saliency": "Saliency", "lime": "LIME", "shap": "SHAP"}
