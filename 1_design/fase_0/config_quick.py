"""Global configuration of the Banca Virtuosa XAI project.

Single immutable source of truth for every parameter (design principle 4: no magic numbers).
Modules receive the config as an argument; only the notebook / scripts import CONFIG.

Changes with respect to the design document v1 (see docs/design_errata.md):
  - USE_SURROGATE removed (D03-bis): no surrogate model in this project.
  - MIN_VAL_ACCURACY = 0.94 (E01).
  - SHAP_NUM_SAMPLES added: the number of gradient samples of Gradient SHAP was an
    unnamed magic number in the design (E18).
  - Validation extended to backend names, ranges and grid/window consistency (E16).

Not validated here: GRADCAM_LAYER / GRADCAM_ALT_LAYER must exist in the model; that check
needs the model and lives in model.py (stage 1).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigConstantsDefault:
    """Global config. Replaces magic numbers with descriptive names."""

    # --- Reproducibility and environment
    SEED: int = 42
    DEVICE: str = "auto"                 # "auto" -> cuda if available, else cpu (warn on cpu)

    # --- Data
    IMAGE_SIZE: int = 128                # MNIST 1x28x28 -> 3x128x128 for DenseNet
    NUM_CLASSES: int = 10
    VAL_SIZE: int = 6000                 # stratified split of the 60k train set
    NUM_DEBUG_IMAGES: int = 10           # fixed validation images (one per class)

    # --- Model and training
    BACKBONE_WEIGHTS: str = "IMAGENET1K_V1"
    FREEZE_BACKBONE: bool = True         # True -> head only, BatchNorm of backbone in eval
    MIN_VAL_ACCURACY: float = 0.94       # warning (not stop) if validation accuracy is below
    NUM_EPOCHS: int = 5
    BATCH_SIZE: int = 64                 # to confirm in phase zero
    OPTIMIZER: str = "adam"
    LEARNING_RATE_HEAD: float = 1e-3     # used when FREEZE_BACKBONE=True
    LEARNING_RATE_FULL: float = 5e-5     # used when FREEZE_BACKBONE=False
    EARLY_STOPPING_PATIENCE: int = 2
    MODEL_MODE: str = "auto"             # "auto" | "train" | "load"
    CHECKPOINT_PATH: str = "checkpoints/densenet121_mnist.pt"

    # --- Grad-CAM
    GRADCAM_LAYER: str = "features.norm5"
    GRADCAM_ALT_LAYER: str = "features.denseblock3"   # for the comparison figure
    GRADCAM_DEBUG_EPOCHS: str = "first_last"

    # --- Logistic regression baseline
    LR_PENALTY: str = "l2"
    LR_SOLVER: str = "lbfgs"
    LR_C: float = 1.0
    LR_MAX_ITER: int = 1000              # check ConvergenceWarning in phase zero

    # --- XAI techniques
    LIME_BACKEND: str = "lime"           # "lime" | "captum"  (provisional, phase zero)
    SHAP_BACKEND: str = "shap"           # "shap" | "captum"  (provisional, phase zero)
    IG_STEPS: int = 50
    OCCLUSION_WINDOW: int = 16
    OCCLUSION_STRIDE: int = 8
    LIME_NUM_SAMPLES: int = 1000
    LIME_GRID_CELLS: int = 8             # 8x8 grid segmentation (to check visually)
    SHAP_NUM_BACKGROUND: int = 50
    SHAP_NUM_SAMPLES: int = 200          # gradient samples of Gradient SHAP (phase zero)

    # --- Sampling
    GALLERY_SIZE: int = 24
    N_QUANT: int = 100                   # quantitative sample; fixed after timing in phase zero

    # --- Comparison
    COMPUTE_AGREEMENT: bool = True       # Spearman and IoU
    SIGN_MODE: str = "absolute"          # "absolute" | "positive"
    MASK_MODE: str = "digit"             # "digit" | "full"
    MASK_THRESHOLD: float = 0.1          # intensity above which a pixel belongs to the digit
    MASK_DILATION: int = 1
    TOPK_FRACTION: float = 0.2
    COMPUTE_FAITHFULNESS: bool = False   # deletion curves
    DELETION_STEPS: int = 10

    # --- Execution and outputs
    RUN_TESTS: bool = False
    OUTPUT_DIR: str = "outputs"

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
        one_of("MODEL_MODE", ("auto", "train", "load"))
        one_of("SIGN_MODE", ("absolute", "positive"))
        one_of("MASK_MODE", ("digit", "full"))
        one_of("LIME_BACKEND", ("lime", "captum"))
        one_of("SHAP_BACKEND", ("shap", "captum"))
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
        require(self.EARLY_STOPPING_PATIENCE >= 0, "EARLY_STOPPING_PATIENCE must be >= 0")
        require(self.MASK_DILATION >= 0, "MASK_DILATION must be >= 0")

        # Positive floats and ranges
        for name in ("LEARNING_RATE_HEAD", "LEARNING_RATE_FULL", "LR_C"):
            require(getattr(self, name) > 0, f"{name} must be > 0")
        require(0.0 < self.MIN_VAL_ACCURACY <= 1.0, "MIN_VAL_ACCURACY must be in (0, 1]")
        require(0.0 < self.TOPK_FRACTION <= 1.0, "TOPK_FRACTION must be in (0, 1]")
        require(0.0 <= self.MASK_THRESHOLD < 1.0, "MASK_THRESHOLD must be in [0, 1)")

        # Geometry: DenseNet downsamples by 32, so smaller images collapse the last layer
        require(self.IMAGE_SIZE >= 32, "IMAGE_SIZE must be >= 32 (DenseNet stride is 32)")
        require(self.OCCLUSION_WINDOW <= self.IMAGE_SIZE,
                "OCCLUSION_WINDOW cannot exceed IMAGE_SIZE")
        require(self.IMAGE_SIZE % self.LIME_GRID_CELLS == 0,
                "IMAGE_SIZE must be a multiple of LIME_GRID_CELLS (equal grid cells)")

        # Data split and sampling
        require(0 < self.VAL_SIZE < 60000, "VAL_SIZE must be in (0, 60000)")
        require(self.NUM_DEBUG_IMAGES == self.NUM_CLASSES,
                "NUM_DEBUG_IMAGES must equal NUM_CLASSES (one fixed image per class)")
        require(self.N_QUANT % self.NUM_CLASSES == 0,
                "N_QUANT must be a multiple of NUM_CLASSES (stratified by predicted class)")


CONFIG = ConfigConstantsDefault()
