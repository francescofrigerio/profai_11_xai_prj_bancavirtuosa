"""Phase zero - technical checks for the Banca Virtuosa XAI project (design section 12).

Exploratory and self-contained: after stage 0 this file is frozen (kept for reproducibility,
never imported by the project). What works here is re-implemented, with tests, in stage 1-2.

HOW TO RUN (Google Colab, GPU runtime)
--------------------------------------
    %cd /content                       # avoid cloning inside an already-cloned folder
    !git clone https://github.com/francescofrigerio/profai_11_xai_prj_bancavirtuosa
    %cd profai_11_xai_prj_bancavirtuosa/1_design/fase_0   # wherever config.py and this file live
    !pip -q install captum lime shap scikit-image
    %run phase0_checks.py --quick      # ~minutes, random weights, tiny data: catches script bugs
    %run phase0_checks.py              # the real run (all checks, in the recommended order)

Use %run (same kernel as the pip install), not "!python": check 1 detects a package that
pip changed after the kernel had already loaded it (=> runtime restart needed), and that
is only visible inside the kernel.

Other options:
    --checks 1,2,5          run only some checks (ids 1-12, see CHECK_NAMES)
    --epochs N              epochs for check 6 (default: CONFIG.NUM_EPOCHS)
    --batch-size N          override CONFIG.BATCH_SIZE for check 6
    --lr-c-sweep 0.1,1.0    extra values of LR_C to evaluate in check 5
    --no-bn-comparison      skip the opposite-FLAG_BATCHNORM_EVAL variant of check 6
    --out-dir PATH          default: <OUTPUT_DIR>/phase0

NOTHING CHANGES BY ITSELF: the script never edits config.py. It prints "proposed values"
and the operator decides (design principle 3). Statuses: PASS / WARN / FAIL / INFO, and
REVIEW = a figure to be judged by eye.
"""
from __future__ import annotations

import sys

_WATCHED = ("numpy", "scipy", "sklearn", "torch", "torchvision", "matplotlib",
            "pandas", "captum", "shap", "lime", "skimage")
_PRELOADED = {m for m in _WATCHED if m in sys.modules}   # loaded before this script ran

import argparse
import copy
import dataclasses
import importlib
import importlib.metadata as importlib_metadata
import json
import platform
import random
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_ERROR = None
except Exception as exc:                     # reported by check 1, not a crash
    torch = nn = F = None
    _TORCH_ERROR = exc

if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG  # noqa: E402

IN_NOTEBOOK = "ipykernel" in sys.modules

# ----------------------------------------------------------------------------------------
# Decision thresholds (proposed in the approved plan; the operator can change them here)
# ----------------------------------------------------------------------------------------
THRESHOLDS = {
    "backend_max_seconds_per_image": 15.0,    # LIME / SHAP backend accepted if <= (T4)
    "quant_stage_max_minutes": 20.0,          # time budget of the quantitative stage
    "run_all_load_max_minutes": 30.0,         # "Esegui tutto" with MODEL_MODE=load
    "run_all_train_max_minutes": 45.0,        # "Esegui tutto" with MODEL_MODE=train
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
N_QUANT_CANDIDATES = (50, 100, 200)
ERROR_EXPLAIN_EXTRA_FRACTION = 0.5   # gallery: errors are explained twice (predicted + true)
FIXED_OVERHEAD_SECONDS = 120.0       # data download, setup, plotting (rough)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MNIST_SIDE = CONFIG.IMAGE_SIZE

CHECK_NAMES = {
    1: "Environment, versions, GPU, no-restart",
    2: "In-place ReLU patch and Grad-CAM",
    3: "LIME and SHAP backends",
    4: "Integrated Gradients (completeness, black baseline)",
    5: "Logistic regression (convergence, accuracy, exactness)",
    6: "Frozen DenseNet training (time, accuracy, BatchNorm)",
    7: "LIME grid segmentation (visual)",
    8: "Digit mask (visual)",
    9: "Timings and N_QUANT",
    10: "Backend decision and table of fixed values",
    11: "Occlusion behaviour (overlap, channels, forwards)",
    12: "Resampling rule to the 28x28 grid",
    13: "Saliency (candidate sixth technique)",
    14: "Checkpoint round trip",
}
ORDER = [1, 2, 5, 6, 14, 3, 4, 13, 7, 8, 11, 12, 9, 10]
SEVERITY = {"FAIL": 4, "WARN": 3, "REVIEW": 2, "PASS": 1, "INFO": 0}


# ----------------------------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


class CheckScope:
    """Context manager for one check: collects lines, swallows (and reports) exceptions."""

    def __init__(self, rec: "Recorder", cid: int, name: str):
        self.rec, self.cid, self.name = rec, cid, name
        self.lines: list[tuple[str, str]] = []
        self.values: dict = {}
        self.traceback = ""
        self.seconds = 0.0

    def __enter__(self):
        self.t0 = time.time()
        print(f"\n=== CHECK {self.cid}: {self.name} " + "=" * 20)
        return self

    def _line(self, status: str, msg: str):
        self.lines.append((status, msg))
        print(f"  [{status}] {msg}")

    def ok(self, msg): self._line("PASS", msg)
    def warn(self, msg): self._line("WARN", msg)
    def fail(self, msg): self._line("FAIL", msg)
    def info(self, msg): self._line("INFO", msg)
    def review(self, msg): self._line("REVIEW", msg)

    def __exit__(self, exc_type, exc, tb):
        self.seconds = time.time() - self.t0
        if exc_type is not None:
            if not issubclass(exc_type, Exception):
                return False                      # let KeyboardInterrupt through
            self.traceback = "".join(traceback.format_exception(exc_type, exc, tb))[-3000:]
            self._line("FAIL", f"exception {exc_type.__name__}: {exc}")
            print(self.traceback)
        self.rec.finish(self)
        return True


class Recorder:
    def __init__(self, path: Path, quick: bool):
        self.path, self.quick = Path(path), quick
        self.data = {"quick_run": quick, "checks": {}, "decisions": {}, "meta": {}}
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text())
                if old.get("quick_run") == quick:
                    self.data = old                   # accumulate across partial runs
            except Exception:
                pass

    def check(self, cid: int) -> CheckScope:
        return CheckScope(self, cid, CHECK_NAMES[cid])

    def finish(self, scope: CheckScope):
        status = max((s for s, _ in scope.lines), key=lambda s: SEVERITY[s], default="INFO")
        self.data["checks"][str(scope.cid)] = {
            "name": scope.name, "status": status, "seconds": round(scope.seconds, 2),
            "lines": scope.lines, "values": scope.values, "traceback": scope.traceback,
        }
        print(f"  -> check {scope.cid} status: {status} ({scope.seconds:.1f}s)")
        self.save()

    def values(self, cid: int) -> dict:
        return self.data["checks"].get(str(cid), {}).get("values", {})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=_json_default))


# ----------------------------------------------------------------------------------------
# Pure helpers (no torch): testable anywhere
# ----------------------------------------------------------------------------------------
def grid_segments(size: int, cells: int) -> np.ndarray:
    """Segment labels of a cells x cells grid over a size x size image."""
    step = size // cells
    rows = np.arange(size) // step
    return (rows[:, None] * cells + rows[None, :]).astype(np.int64)


def digit_mask(img28: np.ndarray, threshold: float, dilation: int) -> np.ndarray:
    """Pixels above `threshold`, dilated by `dilation` pixels (square 3x3 element, 8-connected)."""
    from scipy.ndimage import binary_dilation
    mask = np.asarray(img28) > threshold
    if dilation > 0:
        mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=dilation)
    return mask


def lr_kwargs(cfg) -> dict:
    """Map the config to LogisticRegression kwargs. The l2 penalty is the default in every
    sklearn version, so it is never passed (sklearn >= 1.8 deprecates `penalty`)."""
    import sklearn
    kw = dict(C=cfg.LR_C, solver=cfg.LR_SOLVER, max_iter=cfg.LR_MAX_ITER, random_state=cfg.SEED)
    if cfg.LR_PENALTY != "l2":
        major_minor = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
        if major_minor >= (1, 8):
            kw["l1_ratio"] = 1.0
        else:
            kw["penalty"] = cfg.LR_PENALTY
    return kw


def lr_fit_and_diagnose(Xtr, ytr, Xva, yva, cfg) -> tuple:
    """Fit the LR and measure everything check 5 needs. Returns (model, diagnostics)."""
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model = LogisticRegression(**lr_kwargs(cfg)).fit(Xtr, ytr)
        fit_seconds = time.time() - t0
    categories = sorted({w.category.__name__ for w in caught})
    manual = Xva @ model.coef_.T + model.intercept_
    diag = {
        "fit_seconds": fit_seconds,
        "n_iter": int(np.max(model.n_iter_)),
        "convergence_warning": "ConvergenceWarning" in categories,
        "other_warnings": sorted({f"{w.category.__name__}: {str(w.message)[:140]}"
                                  for w in caught if w.category.__name__ != "ConvergenceWarning"}),
        "val_accuracy": float(model.score(Xva, yva)),
        "coef_colsum_max": float(np.abs(model.coef_.sum(axis=0)).max()),
        "decision_exact_max_err": float(np.abs(manual - model.decision_function(Xva)).max()),
        "classes_ok": bool(np.array_equal(model.classes_, np.arange(cfg.NUM_CLASSES))),
    }
    return model, diag


def worst_status(statuses) -> str:
    return max(statuses, key=lambda s: SEVERITY[s], default="INFO")


def backend_ok(entry: dict | None) -> bool:
    return bool(entry) and bool(entry.get("ok"))


def choose_backend(check3_values: dict, technique: str, preferred: str, alternative: str):
    """Preferred backend if it passed check 3, else the alternative, else None."""
    table = check3_values.get("backends", {})
    for name in (preferred, alternative):
        if backend_ok(table.get(f"{technique}/{name}")):
            return name
    return None


def estimate_run_all(t: dict, cfg, n_quant: int) -> dict:
    """Rough 'Esegui tutto' time (seconds). `t` holds measured per-image / per-epoch seconds."""
    per_image = t["gradcam"] + t["ig"] + t["occlusion"] + t["lime"] + t["shap"] + t.get("saliency", 0.0)
    gallery = cfg.GALLERY_SIZE * (1.0 + ERROR_EXPLAIN_EXTRA_FRACTION) * per_image
    quant = n_quant * per_image
    test_inference = t["val_seconds_per_image"] * 10000
    base = FIXED_OVERHEAD_SECONDS + t["lr_fit"] + test_inference + gallery + quant
    train = cfg.NUM_EPOCHS * (t["epoch_train"] + t["epoch_val"])
    return {"per_image_all_techniques": per_image, "gallery": gallery, "quant": quant,
            "test_inference": test_inference, "load_total": base, "train_total": base + train}


def pick_n_quant(per_image_seconds: float, budget_minutes: float):
    for n in sorted(N_QUANT_CANDIDATES, reverse=True):
        if n * per_image_seconds <= budget_minutes * 60:
            return n, True
    return min(N_QUANT_CANDIDATES), False


# ----------------------------------------------------------------------------------------
# Torch helpers
# ----------------------------------------------------------------------------------------
def sync(device) -> None:
    if torch is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return out, time.perf_counter() - t0


def _imagenet_stats(like):
    mean = torch.tensor(IMAGENET_MEAN, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=like.dtype, device=like.device).view(1, 3, 1, 1)
    return mean, std


def normalize(x):
    mean, std = _imagenet_stats(x)
    return (x - mean) / std


def denormalize(x):
    mean, std = _imagenet_stats(x)
    return x * std + mean


def to_densenet_input(x28, size: int):
    """(N,1,28,28) in [0,1] -> (N,3,size,size) normalized. Order matters (D06): resize,
    replicate the channel, then normalize."""
    x = F.interpolate(x28, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return normalize(x)


def black_normalized(size: int, device):
    """The black of the ORIGINAL image, expressed in the normalized space (D06)."""
    return normalize(torch.zeros(1, 3, size, size, device=device))


def set_batchnorm_eval(module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def enter_train_mode(model, bn_eval: bool) -> None:
    """Single entry point for training mode, called at the start of EVERY epoch (design 7)."""
    model.train()
    if bn_eval:
        model.features.apply(set_batchnorm_eval)


def bn_buffers(features) -> dict:
    return {k: v.detach().clone() for k, v in features.named_buffers()
            if k.endswith("running_mean") or k.endswith("running_var")}


def bn_drift(before: dict, after: dict) -> float:
    return max((after[k] - before[k]).abs().max().item() for k in before) if before else 0.0


def count_inplace_relu(model) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.ReLU) and m.inplace)


def build_densenet(cfg, pretrained: bool):
    from torchvision.models import DenseNet121_Weights, densenet121
    weights = getattr(DenseNet121_Weights, cfg.BACKBONE_WEIGHTS) if pretrained else None
    base = densenet121(weights=weights)
    base.classifier = nn.Linear(base.classifier.in_features, cfg.NUM_CLASSES)
    return base


if torch is not None:
    class SafeDenseNet(nn.Module):
        """DenseNet121 with no in-place ReLU (module flags AND the functional one in forward).
        Same parameter names as torchvision's model, so state dicts are interchangeable."""

        def __init__(self, base):
            super().__init__()
            for m in base.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False
            self.features = base.features
            self.classifier = base.classifier

        def forward(self, x):
            out = F.relu(self.features(x), inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            return self.classifier(torch.flatten(out, 1))
else:
    SafeDenseNet = None


def wrap_for_flag(base, cfg):
    """FLAG_RELU_INPLACE=True keeps torchvision's in-place ReLUs, False patches them.
    Never switched automatically: only the config decides."""
    return base if cfg.FLAG_RELU_INPLACE else SafeDenseNet(base)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_fig(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"  figure saved: {path}")
    plt.show() if IN_NOTEBOOK else plt.close(fig)


# ----------------------------------------------------------------------------------------
# Context: data, model, inputs (lazy, cached)
# ----------------------------------------------------------------------------------------
def make_ctx(args) -> SimpleNamespace:
    overrides = {}
    if args.batch_size:
        overrides["BATCH_SIZE"] = args.batch_size
    cfg = dataclasses.replace(CONFIG, **overrides)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.OUTPUT_DIR) / "phase0"
    suffix = "_quick" if args.quick else ""
    device = None
    if torch is not None:
        want_cuda = cfg.DEVICE == "cuda" or (cfg.DEVICE == "auto" and torch.cuda.is_available())
        device = torch.device("cuda" if want_cuda else "cpu")
    return SimpleNamespace(
        cfg=cfg, args=args, quick=args.quick, device=device, out_dir=out_dir, cache=SimpleNamespace(),
        rec=Recorder(out_dir / f"phase0_results{suffix}.json", args.quick),
        ckpt_path=Path("checkpoints") / f"phase0_densenet121_head{suffix}.pt",
    )


def load_data(ctx):
    """MNIST + stratified split (54k/6k as in D-data), kept as float tensors in [0,1]."""
    if hasattr(ctx.cache, "data"):
        return ctx.cache.data
    from sklearn.model_selection import train_test_split
    from torchvision import datasets
    tr = datasets.MNIST(root="./data", train=True, download=True)
    te = datasets.MNIST(root="./data", train=False, download=True)
    X_tv = tr.data.unsqueeze(1).float() / 255.0
    y_tv = tr.targets.clone()
    idx_train, idx_val = train_test_split(
        np.arange(len(y_tv)), test_size=ctx.cfg.VAL_SIZE, stratify=y_tv.numpy(),
        random_state=ctx.cfg.SEED)
    if ctx.quick:
        idx_train, idx_val = idx_train[:2000], idx_val[:500]
    ctx.cache.data = SimpleNamespace(
        X_trainval=X_tv, y_trainval=y_tv, idx_train=idx_train, idx_val=idx_val,
        X_test=te.data.unsqueeze(1).float() / 255.0, y_test=te.targets.clone())
    return ctx.cache.data


def get_model(ctx):
    """Eval-mode model (in-place or patched ReLU, per FLAG_RELU_INPLACE). Uses the check-6 head if it exists, else a random head
    (then maps are meaningless: only shapes, finiteness and timings are valid)."""
    if hasattr(ctx.cache, "model"):
        return ctx.cache.model
    base = build_densenet(ctx.cfg, pretrained=not ctx.quick)
    trained = ctx.ckpt_path.exists()
    if trained:
        base.load_state_dict(torch.load(ctx.ckpt_path, map_location="cpu")["state_dict"])
    else:
        print("  NOTE: no phase-0 checkpoint found (run check 6 first): random head, "
              "attribution maps are not meaningful.")
    model = wrap_for_flag(base, ctx.cfg).to(ctx.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ctx.cache.model, ctx.cache.model_trained = model, trained
    return model


def sample_inputs(ctx, classes=(7, 4, 1)):
    """First test image of each requested class: model, normalized batch, predictions, raw."""
    D, model, S = load_data(ctx), get_model(ctx), ctx.cfg.DENSENET_SIZE
    y = D.y_test.numpy()
    idx = [int(np.where(y == c)[0][0]) for c in classes]
    x28 = D.X_test[idx]
    x = to_densenet_input(x28, S).to(ctx.device)
    with torch.no_grad():
        pred = model(x).argmax(1).cpu().numpy()
    return model, x, pred, x28


def get_background(ctx):
    D, S = load_data(ctx), ctx.cfg.DENSENET_SIZE
    rng = np.random.RandomState(ctx.cfg.SEED)
    pick = rng.choice(D.idx_train, ctx.cfg.SHAP_NUM_BACKGROUND, replace=False)
    return to_densenet_input(D.X_trainval[torch.as_tensor(pick)], S).to(ctx.device)


# ----------------------------------------------------------------------------------------
# Explainers (prototypes of stage 2; each returns a native-resolution 2D numpy map)
# ----------------------------------------------------------------------------------------
def explain_gradcam(model, x1, target, layer_name, relu=True):
    from captum.attr import LayerGradCam
    layer = dict(model.named_modules())[layer_name]
    att = LayerGradCam(model, layer).attribute(x1, target=int(target), relu_attributions=relu)
    return att[0, 0].detach().cpu().numpy()


def explain_ig(model, x1, target, cfg, steps=None):
    from captum.attr import IntegratedGradients
    att = IntegratedGradients(model).attribute(
        x1, baselines=black_normalized(x1.shape[-1], x1.device), target=int(target),
        n_steps=steps or cfg.IG_STEPS, internal_batch_size=50)
    return att


def explain_saliency(model, x1, target):
    from captum.attr import Saliency
    att = Saliency(model).attribute(x1, target=int(target), abs=True)
    return att[0].sum(0).detach().cpu().numpy()       # sum over channels, >= 0


def explain_lime_pkg(model, x1, target, cfg, device):
    from lime import lime_image
    S = x1.shape[-1]
    img = denormalize(x1)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float64)
    segments = grid_segments(S, cfg.LIME_GRID_CELLS)

    def classifier_fn(batch):
        t = normalize(torch.from_numpy(batch).float().permute(0, 3, 1, 2)).to(device)
        probs = []
        with torch.no_grad():
            for i in range(0, len(t), 64):
                probs.append(F.softmax(model(t[i:i + 64]), dim=1).cpu())
        return torch.cat(probs).numpy()

    import inspect
    explainer = lime_image.LimeImageExplainer(random_state=cfg.SEED)
    kwargs = dict(labels=(int(target),), top_labels=None, hide_color=0,
                  num_samples=cfg.LIME_NUM_SAMPLES, batch_size=64,
                  segmentation_fn=lambda im: segments, random_seed=cfg.SEED)
    if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
        kwargs["progress_bar"] = False            # absent in lime 0.2.0.1
    expl = explainer.explain_instance(img, classifier_fn, **kwargs)
    out = np.zeros((S, S))
    for seg, weight in dict(expl.local_exp[int(target)]).items():
        out[segments == seg] = weight
    return out


def explain_lime_captum(model, x1, target, cfg, device):
    from captum.attr import Lime
    S = x1.shape[-1]
    seg = torch.from_numpy(grid_segments(S, cfg.LIME_GRID_CELLS)).long().to(device)
    mask = seg.view(1, 1, S, S).expand(1, 3, S, S).contiguous()
    torch.manual_seed(cfg.SEED)
    att = Lime(lambda inp: F.softmax(model(inp), dim=1)).attribute(
        x1, baselines=black_normalized(S, device), target=int(target), feature_mask=mask,
        n_samples=cfg.LIME_NUM_SAMPLES, perturbations_per_eval=64, show_progress=False)
    return att[0].mean(0).detach().cpu().numpy()


def _pick_shap_target(sv, target, num_classes):
    """shap returns a list per class (older) or an array with classes last (newer)."""
    if isinstance(sv, list):
        arr = np.asarray(sv[target])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 5 and arr.shape[-1] == num_classes:
            arr = arr[..., target]
        elif arr.ndim == 5 and arr.shape[0] == num_classes:
            arr = arr[target]
        else:
            raise ValueError(f"unexpected shap output shape {arr.shape}")
    return arr[0].sum(axis=0)


def explain_shap_pkg(model, x1, target, background, cfg):
    import shap
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    sv = shap.GradientExplainer(model, background).shap_values(x1, nsamples=cfg.SHAP_NUM_SAMPLES)
    return _pick_shap_target(sv, int(target), cfg.NUM_CLASSES)


def explain_shap_captum(model, x1, target, background, cfg, chunk=50):
    from captum.attr import GradientShap
    gs, total, done, k = GradientShap(model), None, 0, 0
    while done < cfg.SHAP_NUM_SAMPLES:          # chunked: bounded GPU memory
        m = min(chunk, cfg.SHAP_NUM_SAMPLES - done)
        set_seed(cfg.SEED + k)                      # Captum draws alphas with the NUMPY RNG too
        a = gs.attribute(x1, baselines=background, n_samples=m, stdevs=0.0, target=int(target))
        total = a * m if total is None else total + a * m
        done, k = done + m, k + 1
    return (total / cfg.SHAP_NUM_SAMPLES)[0].sum(0).detach().cpu().numpy()


def to_grid28(map2d, size: int):
    """Resampling rule under test (E03): native -> 128 (bilinear if smaller) -> 28 (area mean)."""
    t = torch.as_tensor(np.asarray(map2d), dtype=torch.float32)[None, None]
    if t.shape[-1] < size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return F.interpolate(t, size=(MNIST_SIDE, MNIST_SIDE), mode="area")[0, 0].numpy()


# ----------------------------------------------------------------------------------------
# CHECK 1 - environment
# ----------------------------------------------------------------------------------------
DIST_NAMES = {"sklearn": "scikit-learn", "skimage": "scikit-image"}
PIN_CANDIDATES = ["captum", "lime", "shap", "scikit-learn", "scikit-image", "scipy", "numpy",
                  "matplotlib", "pandas", "seaborn", "pytest"]


def check_1(ctx):
    with ctx.rec.check(1) as r:
        r.values["python"] = platform.python_version()
        r.values["platform"] = platform.platform()
        r.values["colab"] = "google.colab" in sys.modules
        required = ["torch", "torchvision", "numpy", "scipy", "sklearn", "captum", "lime",
                    "shap", "skimage", "matplotlib"]
        versions, failed, restart = {}, [], []
        for name in required + ["pandas", "seaborn"]:
            dist = DIST_NAMES.get(name, name)
            try:
                mod = importlib.import_module(name)
                loaded = getattr(mod, "__version__", "?")
                disk = importlib_metadata.version(dist)
                versions[dist] = disk
                if name in _PRELOADED and loaded.split("+")[0] != disk.split("+")[0]:
                    restart.append(f"{dist}: loaded {loaded}, installed {disk}")
            except Exception as exc:
                if name in required:
                    failed.append(f"{name}: {type(exc).__name__}: {exc}")
        r.values["versions"] = versions
        if failed:
            r.fail("import failed -> " + "; ".join(failed))
        else:
            r.ok("all required libraries import: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
        if restart:
            r.fail("pip changed already-loaded packages -> RESTART THE RUNTIME and rerun: "
                   + "; ".join(restart))
        elif IN_NOTEBOOK:
            r.ok("no package changed after being loaded (no restart needed)")
        else:
            r.info("not running inside a notebook kernel: the no-restart test is vacuous "
                   "(use %run in Colab)")
        if torch is not None:
            try:
                np.zeros(3, dtype=np.float32)
                torch.from_numpy(np.zeros(3, dtype=np.float32)).numpy()
                r.ok("torch <-> numpy round trip works (ABI compatible)")
            except Exception as exc:
                r.fail(f"torch/numpy incompatibility: {exc}")
            cuda = torch.cuda.is_available()
            r.values["cuda"] = cuda
            if cuda:
                r.values["gpu"] = torch.cuda.get_device_name(0)
                r.ok(f"GPU available: {r.values['gpu']}")
            else:
                r.warn("NO GPU: the full run would be very slow (design: visible warning on CPU)")
        pins = [f"{d}=={versions.get(d) or _safe_version(d)}" for d in PIN_CANDIDATES
                if versions.get(d) or _safe_version(d)]
        text = ("# candidate pins measured in phase 0 (torch/torchvision deliberately excluded)\n"
                f"# torch {versions.get('torch')} / torchvision {versions.get('torchvision')}\n"
                + "\n".join(pins) + "\n")
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        (ctx.out_dir / "requirements_candidate.txt").write_text(text)
        r.info(f"candidate pins written to {ctx.out_dir / 'requirements_candidate.txt'}")


def _safe_version(dist: str):
    try:
        return importlib_metadata.version(dist)
    except Exception:
        return None


# ----------------------------------------------------------------------------------------
# CHECK 2 - in-place ReLU and Grad-CAM
# ----------------------------------------------------------------------------------------
def _rel_diff(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def _relu_probes(x, bg, cfg):
    """Technique -> function(model) -> 2D numpy map. Same seeds for both models, so any
    difference between the in-place and the patched model comes from the ReLU, not the dice."""
    from captum.attr import GradientShap

    def gradshap(m):
        set_seed(cfg.SEED)                      # numpy + torch: Captum uses both
        att = GradientShap(m).attribute(x, baselines=bg, n_samples=4, stdevs=0.0, target=3)
        return att[0].sum(0).detach().cpu().numpy()

    def shap_pkg(m):
        import shap
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        sv = shap.GradientExplainer(m, bg).shap_values(x, nsamples=4)
        return _pick_shap_target(sv, 3, cfg.NUM_CLASSES)

    return {
        "gradcam on norm5 (signed)": lambda m: explain_gradcam(m, x, 3, cfg.GRADCAM_LAYER, relu=False),
        "integrated gradients": lambda m: explain_ig(m, x, 3, cfg, steps=10)[0].sum(0).detach().cpu().numpy(),
        "saliency": lambda m: explain_saliency(m, x, 3),
        "gradient shap (captum)": gradshap,
        "gradient shap (shap)": shap_pkg,
    }


def check_2(ctx):
    with ctx.rec.check(2) as r:
        from captum.attr import LayerGradCam
        cfg, S, dev = ctx.cfg, ctx.cfg.DENSENET_SIZE, ctx.device
        set_seed(cfg.SEED)
        x = torch.randn(1, 3, S, S, device=dev)
        bg = torch.randn(4, 3, S, S, device=dev)
        base_a = build_densenet(cfg, pretrained=not ctx.quick)
        base_b = copy.deepcopy(base_a)                 # identical weights, separate modules
        inplace = base_a.to(dev).eval()                 # torchvision as shipped
        safe = SafeDenseNet(base_b).to(dev).eval()      # patched
        r.values["inplace_relu_modules"] = count_inplace_relu(inplace)
        r.values["inplace_relu_after_patch"] = count_inplace_relu(safe)
        r.info(f"in-place ReLU modules in torchvision DenseNet121: {r.values['inplace_relu_modules']} "
               "(plus the functional F.relu(inplace=True) in forward)")
        r.info(f"FLAG_RELU_INPLACE={cfg.FLAG_RELU_INPLACE}: the models of the other checks use "
               + ("torchvision's in-place ReLUs" if cfg.FLAG_RELU_INPLACE else "the patched model"))
        with torch.no_grad():
            diff = (inplace(x) - safe(x)).abs().max().item()
        r.values["logits_max_diff"] = diff
        if r.values["inplace_relu_after_patch"] == 0 and diff <= THRESHOLDS["logits_patch_atol"]:
            r.ok(f"patch removes every in-place ReLU and leaves logits unchanged (max diff {diff:.2e})")
        else:
            r.fail(f"patch problem: in-place left={r.values['inplace_relu_after_patch']}, "
                   f"logits diff={diff:.2e}")
        # Does the in-place model give the SAME explanations? "Runs without error" proves nothing:
        # an in-place ReLU can silently change what a hook sees.
        probe_results, trouble = {}, []
        for name, fn in _relu_probes(x, bg, cfg).items():
            out = {}
            for tag, model in (("inplace", inplace), ("patched", safe)):
                try:
                    out[tag] = fn(model)
                except Exception as exc:
                    out[tag] = f"{type(exc).__name__}: {str(exc)[:160]}"
            entry = {}
            if isinstance(out["patched"], str):
                entry["patched_error"] = out["patched"]
                r.fail(f"{name}: fails even on the PATCHED model: {out['patched']}")
            elif isinstance(out["inplace"], str):
                entry["inplace_error"] = out["inplace"]
                trouble.append(name)
                (r.fail if cfg.FLAG_RELU_INPLACE else r.info)(
                    f"{name}: FAILS on the in-place model ({out['inplace']}) -> the patch is needed")
            else:
                rel = _rel_diff(out["inplace"], out["patched"])
                entry["rel_diff_vs_patched"] = rel
                if rel <= 1e-3:
                    r.ok(f"{name}: in-place model gives the same map (relative diff {rel:.1e})")
                else:
                    trouble.append(name)
                    (r.fail if cfg.FLAG_RELU_INPLACE else r.warn)(
                        f"{name}: in-place model gives a DIFFERENT map (relative diff {rel:.2f}) "
                        "-> silently wrong explanations")
            probe_results[name] = entry
        r.values["relu_probes"] = probe_results
        r.values["inplace_safe_for_all"] = not trouble
        if trouble and cfg.FLAG_RELU_INPLACE:
            r.fail("FLAG_RELU_INPLACE=True is NOT safe for: " + ", ".join(trouble)
                   + ". Set FLAG_RELU_INPLACE=False in config (nothing switches by itself).")
        # Grad-CAM shapes and timing on the patched model (design D10)
        for name, divisor in ((cfg.GRADCAM_LAYER, 32), (cfg.GRADCAM_ALT_LAYER, 16)):
            expected = S // divisor
            times = []
            for _ in range(3):
                att, sec = timed(lambda: LayerGradCam(
                    safe, dict(safe.named_modules())[name]).attribute(
                        x, target=3, relu_attributions=True), dev)
                times.append(sec)
            a = att.detach().cpu().numpy()
            good = (a.shape == (1, 1, expected, expected) and np.isfinite(a).all() and (a >= 0).all())
            r.values[f"gradcam_seconds_{name}"] = float(np.median(times))
            (r.ok if good else r.fail)(
                f"Grad-CAM on {name}: shape {a.shape} (expected (1,1,{expected},{expected})), "
                f"finite, non-negative: {good}; {np.median(times) * 1000:.0f} ms")
        r.values["gradcam_seconds"] = r.values[f"gradcam_seconds_{cfg.GRADCAM_LAYER}"]


# ----------------------------------------------------------------------------------------
# CHECK 3 - LIME and SHAP backends
# ----------------------------------------------------------------------------------------
def check_3(ctx):
    with ctx.rec.check(3) as r:
        cfg, dev, S = ctx.cfg, ctx.device, ctx.cfg.DENSENET_SIZE
        model, x, pred, x28 = sample_inputs(ctx)
        bg = get_background(ctx)
        trained = getattr(ctx.cache, "model_trained", False)
        runners = {
            "lime/lime": lambda i: explain_lime_pkg(model, x[i:i + 1], pred[i], cfg, dev),
            "lime/captum": lambda i: explain_lime_captum(model, x[i:i + 1], pred[i], cfg, dev),
            "shap/shap": lambda i: explain_shap_pkg(model, x[i:i + 1], pred[i], bg, cfg),
            "shap/captum": lambda i: explain_shap_captum(model, x[i:i + 1], pred[i], bg, cfg),
        }
        r.values["backends"], first_maps = {}, {}
        for key, fn in runners.items():
            entry = {"ok": False}
            r.values["backends"][key] = entry
            try:
                maps, secs = [], []
                for i in range(len(x)):
                    m, sec = timed(lambda: fn(i), dev)
                    maps.append(np.asarray(m, dtype=np.float64))
                    secs.append(sec)
                first_maps[key] = maps[0]
                shapes_ok = all(m.shape == (S, S) for m in maps)
                finite = all(np.isfinite(m).all() for m in maps)
                median = float(np.median(secs))
                again = np.asarray(fn(0), dtype=np.float64)
                scale = max(np.abs(maps[0]).max(), 1e-12)
                max_rel = float(np.abs(maps[0] - again).max() / scale)
                deterministic = bool(np.allclose(maps[0], again, rtol=THRESHOLDS["determinism_rtol"],
                                                 atol=1e-4 * scale))
                constant = bool(all(np.ptp(m) == 0 for m in maps))
                empty_maps = int(sum(np.abs(m).max() <= 1e-12 for m in maps))
                entry.update(shape=list(maps[0].shape), finite=finite, seconds_median=median,
                             deterministic=deterministic, repeat_max_rel_diff=max_rel,
                             constant_map=constant, empty_maps=empty_maps)
                fast = median <= THRESHOLDS["backend_max_seconds_per_image"]
                collapsed = bool(empty_maps and trained and not ctx.quick)
                entry["ok"] = bool(shapes_ok and finite and fast and not collapsed)
                line = (f"{key}: shape {maps[0].shape}, finite={finite}, "
                        f"median {median:.1f}s/img, deterministic={deterministic}")
                (r.ok if entry["ok"] else r.fail)(line + ("" if fast else "  [TOO SLOW]"))
                if collapsed:
                    r.warn(f"{key}: the map is all zeros for {empty_maps} of {len(maps)} images (the "
                           "explanation collapsed, e.g. a Lasso that over-regularizes): backend NOT "
                           "selectable as it is")
                if not deterministic:
                    r.warn(f"{key}: same seed gave different maps (max relative diff {max_rel:.1e}; "
                           "GPU atomics or unseeded randomness)")
                if constant:
                    (r.info if (ctx.quick or not trained) else r.warn)(
                        f"{key}: constant map (expected with a barely trained quick-run model; "
                        "suspicious on the real checkpoint)")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                r.fail(f"{key}: {entry['error']}")
        if first_maps:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 1 + len(first_maps), figsize=(3 * (1 + len(first_maps)), 3.2))
            axes[0].imshow(x28[0, 0], cmap="gray")
            axes[0].set_title(f"true 7 / predicted {pred[0]}")
            for ax, (k, m) in zip(axes[1:], first_maps.items()):
                a = max(np.abs(m).max(), 1e-12)
                ax.imshow(m, cmap="seismic", vmin=-a, vmax=a)
                ax.set_title(k)
            for ax in axes:
                ax.axis("off")
            save_fig(fig, ctx.out_dir / "phase0_03_backend_maps.png")
            r.review("look at phase0_03_backend_maps.png: do the maps of both backends of each "
                     "technique roughly highlight the digit? (meaningful only if check 6 ran first)")


# ----------------------------------------------------------------------------------------
# CHECK 4 - Integrated Gradients
# ----------------------------------------------------------------------------------------
def check_4(ctx):
    with ctx.rec.check(4) as r:
        cfg, dev, S = ctx.cfg, ctx.device, ctx.cfg.DENSENET_SIZE
        model, x, pred, _ = sample_inputs(ctx)
        baseline = black_normalized(S, dev)
        back_to_pixels = denormalize(baseline).abs().max().item()
        r.values["normalized_black_per_channel"] = baseline[0, :, 0, 0].tolist()
        (r.ok if back_to_pixels < 1e-6 else r.fail)(
            f"baseline is black in the original image (max |denormalized| = {back_to_pixels:.1e}); "
            f"in normalized space it is {np.round(r.values['normalized_black_per_channel'], 3).tolist()}, "
            "NOT zero")

        def measure(steps):
            rels, secs, zero_ok = [], [], []
            for i in range(len(x)):
                xi = x[i:i + 1]
                att, sec = timed(lambda: explain_ig(model, xi, pred[i], cfg, steps=steps), dev)
                secs.append(sec)
                with torch.no_grad():
                    f_x, f_b = model(xi)[0, pred[i]].item(), model(baseline)[0, pred[i]].item()
                gap = abs(att.sum().item() - (f_x - f_b))
                rels.append(gap / max(abs(f_x - f_b), 1e-6))
                same = ((xi - baseline).abs().amax(dim=1, keepdim=True) < 1e-9).expand_as(att)
                zero_ok.append(bool((att[same].abs() <= 1e-12).all()) if same.any() else True)
            return rels, secs, zero_ok

        # completeness as a function of the number of integration steps
        sweep = {}
        for steps in sorted({cfg.IG_STEPS, 100, 200, 400}):
            rels, secs, zero_ok = measure(steps)
            sweep[str(steps)] = {"worst_rel_error": float(max(rels)), "per_image_rel_error": rels,
                                 "seconds_median": float(np.median(secs)), "zero_on_background": all(zero_ok)}
            r.info(f"IG_STEPS={steps}: worst completeness error {max(rels):.2%}, "
                   f"{np.median(secs):.2f}s/img")
        r.values["ig_sweep"] = sweep
        entry = sweep[str(cfg.IG_STEPS)]
        worst = entry["worst_rel_error"]
        r.values.update(ig_rel_error=[worst], ig_seconds_per_image=entry["seconds_median"])
        if worst <= THRESHOLDS["ig_completeness_pass_rel"]:
            r.ok(f"completeness holds at the configured IG_STEPS={cfg.IG_STEPS}: worst error {worst:.2%}")
        elif worst <= THRESHOLDS["ig_completeness_warn_rel"]:
            r.warn(f"completeness only approximate at IG_STEPS={cfg.IG_STEPS}: worst error {worst:.2%}")
        else:
            r.fail(f"completeness violated at IG_STEPS={cfg.IG_STEPS}: worst error {worst:.2%}")
        passing = [int(k) for k, v in sweep.items()
                   if v["worst_rel_error"] <= THRESHOLDS["ig_completeness_pass_rel"]]
        r.values["ig_steps_proposed"] = min(passing) if passing else None
        if passing:
            r.info(f"smallest tested IG_STEPS meeting {THRESHOLDS['ig_completeness_pass_rel']:.0%}: {min(passing)}")
        else:
            r.warn("no tested number of steps reaches the completeness threshold: look at the per-image errors")
        (r.ok if all(v["zero_on_background"] for v in sweep.values()) else r.fail)(
            "attribution is exactly zero wherever the image equals the baseline (background)")


# ----------------------------------------------------------------------------------------
# CHECK 13 - Saliency (candidate sixth technique)
# ----------------------------------------------------------------------------------------
def check_13(ctx):
    with ctx.rec.check(13) as r:
        cfg, dev, S = ctx.cfg, ctx.device, ctx.cfg.DENSENET_SIZE
        model, x, pred, x28 = sample_inputs(ctx)
        secs, sal_share, ig_share, area = [], [], [], []
        valid, deterministic = True, True
        for i in range(len(x)):
            xi = x[i:i + 1]
            m, sec = timed(lambda: explain_saliency(model, xi, pred[i]), dev)
            secs.append(sec)
            again = explain_saliency(model, xi, pred[i])
            valid &= bool(m.shape == (S, S) and np.isfinite(m).all() and (m >= 0).all() and m.max() > 0)
            deterministic &= bool(np.allclose(m, again, rtol=THRESHOLDS["determinism_rtol"],
                                              atol=1e-4 * max(m.max(), 1e-12)))
            ig = explain_ig(model, xi, pred[i], cfg)[0].sum(0).detach().cpu().numpy()
            mask = digit_mask(x28[i, 0].numpy(), cfg.MASK_THRESHOLD, cfg.MASK_DILATION)

            def share(a):
                g = to_grid28(np.abs(a), S)
                return float(g[mask].sum() / max(g.sum(), 1e-12))
            sal_share.append(share(m))
            ig_share.append(share(ig))
            area.append(float(mask.mean()))
        r.values.update(saliency_seconds_per_image=float(np.median(secs)), mass_in_mask_saliency=sal_share,
                        mass_in_mask_ig=ig_share, mask_area_fraction=area)
        (r.ok if valid else r.fail)("saliency map: shape (S,S), finite, non-negative (abs=True), not all zero")
        (r.ok if deterministic else r.warn)(f"saliency is deterministic: {deterministic}")
        r.info(f"saliency {np.median(secs):.3f}s/img (one backward pass)")
        r.info("share of the attribution falling inside the digit mask (a random map would give the "
               f"mask area, {np.mean(area):.1%}): saliency {np.mean(sal_share):.1%}, "
               f"integrated gradients {np.mean(ig_share):.1%}")
        if np.mean(sal_share) <= np.mean(area) + 0.05:
            r.warn("saliency concentrates on the digit no more than a random map would "
                   "(expected to be noisy: a result to explain, not a bug)")
        r.review("saliency is the noisiest gradient technique: the 7x7 agreement matrices and the gallery "
                 "would gain a column/row. Operator decision (errata E22).")


# ----------------------------------------------------------------------------------------
# CHECK 5 - logistic regression
# ----------------------------------------------------------------------------------------
def check_5(ctx):
    with ctx.rec.check(5) as r:
        import sklearn
        cfg, D = ctx.cfg, load_data(ctx)
        X = D.X_trainval.numpy().reshape(len(D.y_trainval), -1).astype(np.float64)
        y = D.y_trainval.numpy()
        Xtr, ytr, Xva, yva = X[D.idx_train], y[D.idx_train], X[D.idx_val], y[D.idx_val]
        r.values.update(sklearn=sklearn.__version__, input_range=[float(X.min()), float(X.max())])
        r.info(f"sklearn {sklearn.__version__}; LR input range [{X.min():.1f}, {X.max():.1f}], "
               f"{X.shape[1]} features, not standardized")
        sweep = [cfg.LR_C] + [c for c in (ctx.args.lr_c_sweep or []) if c != cfg.LR_C]
        r.values["runs"] = {}
        models = {}
        for c in sweep:
            run_cfg = dataclasses.replace(cfg, LR_C=c)
            model, d = lr_fit_and_diagnose(Xtr, ytr, Xva, yva, run_cfg)
            models[c] = model
            r.values["runs"][str(c)] = d
            tag = f"C={c}"
            r.info(f"{tag}: fit {d['fit_seconds']:.1f}s, n_iter {d['n_iter']}, val acc {d['val_accuracy']:.4f}")
            (r.warn if d["convergence_warning"] else r.ok)(
                f"{tag}: ConvergenceWarning={d['convergence_warning']}"
                + (f" -> consider raising LR_MAX_ITER (now {cfg.LR_MAX_ITER})" if d["convergence_warning"] else ""))
            lo, hi = THRESHOLDS["lr_expected_acc_range"]
            (r.ok if lo <= d["val_accuracy"] <= hi else r.warn)(
                f"{tag}: validation accuracy {d['val_accuracy']:.4f} (design expects ~0.92, range {lo}-{hi})")
            (r.ok if d["decision_exact_max_err"] < 1e-8 else r.fail)(
                f"{tag}: logit = sum(w*x)+b exactly (max error {d['decision_exact_max_err']:.1e})")
            (r.ok if d["classes_ok"] else r.fail)(f"{tag}: classes_ == 0..9 (coefficient rows aligned)")
            (r.ok if d["coef_colsum_max"] <= THRESHOLDS["lr_colsum_warn"] else r.warn)(
                f"{tag}: per-pixel sum of coefficients over classes max {d['coef_colsum_max']:.1e} "
                "(~0 with l2: relative and absolute coefficients coincide)")
            if d["other_warnings"]:
                for w in d["other_warnings"]:
                    r.info(f"{tag}: warning raised -> {w}")
        import matplotlib.pyplot as plt
        for c, model in models.items():
            rel = model.coef_ - model.coef_.mean(axis=0, keepdims=True)
            fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
            for k, ax in enumerate(axes.ravel()):
                a = max(np.abs(rel[k]).max(), 1e-12)
                ax.imshow(rel[k].reshape(MNIST_SIDE, MNIST_SIDE), cmap="seismic", vmin=-a, vmax=a)
                ax.set_title(f"class {k}")
                ax.axis("off")
            fig.suptitle(f"LR global maps (relative coefficients), C={c}")
            save_fig(fig, ctx.out_dir / f"phase0_05_lr_global_maps_C{c}.png")
        r.review("look at the phase0_05_lr_global_maps_*.png: are the maps readable "
                 "(smooth, digit-shaped) at the configured LR_C?")
        r.values["fit_seconds"] = r.values["runs"][str(cfg.LR_C)]["fit_seconds"]


# ----------------------------------------------------------------------------------------
# CHECK 6 - frozen DenseNet training
# ----------------------------------------------------------------------------------------
def evaluate(model, X, y, idx, size, bs=256):
    model.eval()
    tot_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            out = model(to_densenet_input(X[b], size))
            tot_loss += F.cross_entropy(out, y[b], reduction="sum").item()
            correct += (out.argmax(1) == y[b]).sum().item()
    return tot_loss / len(idx), correct / len(idx)


def train_variant(ctx, bn_eval: bool, epochs: int):
    """Head-only training. Fixed number of epochs, FINAL weights are kept (no early stopping)."""
    cfg, D, dev, S = ctx.cfg, load_data(ctx), ctx.device, ctx.cfg.DENSENET_SIZE
    set_seed(cfg.SEED)
    model = wrap_for_flag(build_densenet(cfg, pretrained=not ctx.quick), cfg).to(dev)
    for p in model.features.parameters():
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.LEARNING_RATE_HEAD)
    X, y = D.X_trainval.to(dev), D.y_trainval.to(dev)
    idx_tr = torch.as_tensor(D.idx_train, device=dev)
    idx_va = torch.as_tensor(D.idx_val, device=dev)
    bn0 = bn_buffers(model.features)
    gen = torch.Generator().manual_seed(cfg.SEED)
    history, drift1 = [], None
    for epoch in range(1, epochs + 1):
        enter_train_mode(model, bn_eval)              # re-applied at EVERY epoch start
        sync(dev)
        t0 = time.time()
        perm = torch.randperm(len(idx_tr), generator=gen).to(dev)
        run_loss, correct = torch.zeros((), device=dev), torch.zeros((), device=dev)
        for i in range(0, len(perm), cfg.BATCH_SIZE):
            b = idx_tr[perm[i:i + cfg.BATCH_SIZE]]
            out = model(to_densenet_input(X[b], S))
            loss = F.cross_entropy(out, y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.detach() * len(b)
            correct += (out.argmax(1) == y[b]).sum()
        sync(dev)
        train_seconds = time.time() - t0
        t1 = time.time()
        val_loss, val_acc = evaluate(model, X, y, idx_va, S)
        sync(dev)
        val_seconds = time.time() - t1
        if epoch == 1:
            drift1 = bn_drift(bn0, bn_buffers(model.features))
        history.append(dict(epoch=epoch, train_loss=run_loss.item() / len(idx_tr),
                            train_acc=correct.item() / len(idx_tr), val_loss=val_loss,
                            val_acc=val_acc, train_seconds=train_seconds, val_seconds=val_seconds))
        print(f"    epoch {epoch}: train loss {history[-1]['train_loss']:.4f} acc "
              f"{history[-1]['train_acc']:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{train_seconds:.0f}s + {val_seconds:.0f}s")
    return dict(history=history, n_trainable=sum(p.numel() for p in trainable),
                final_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                bn_drift_epoch1=drift1, bn_drift_final=bn_drift(bn0, bn_buffers(model.features)))


def check_6(ctx):
    with ctx.rec.check(6) as r:
        cfg, D, S = ctx.cfg, load_data(ctx), ctx.cfg.DENSENET_SIZE
        epochs = 1 if ctx.quick else (ctx.args.epochs or cfg.NUM_EPOCHS)
        if not cfg.FREEZE_BACKBONE:
            r.warn("FREEZE_BACKBONE=False: this check measures the frozen-head setup only")
        # 6a: the tensor (GPU-friendly) preprocessing equals the torchvision PIL pipeline
        from PIL import Image
        from torchvision import transforms
        tf = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(),
                                 transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        raw = (D.X_test[:16, 0] * 255).round().byte().numpy()
        ref = torch.stack([tf(Image.fromarray(a)) for a in raw])
        diff = (ref - to_densenet_input(D.X_test[:16], S)).abs()
        r.values.update(preproc_max_abs_diff=diff.max().item(), preproc_mean_abs_diff=diff.mean().item())
        (r.ok if diff.max().item() <= THRESHOLDS["preproc_max_abs_diff"] else r.warn)(
            f"tensor preprocessing vs torchvision PIL pipeline: max diff {diff.max():.4f}, "
            f"mean {diff.mean():.5f} (normalized units; uint8 rounding alone is ~0.009)")
        # 6b: the configured setting
        flag = cfg.FLAG_BATCHNORM_EVAL
        r.info(f"training CONFIGURED setting: {epochs} epoch(s), batch {cfg.BATCH_SIZE}, "
               f"lr {cfg.LEARNING_RATE_HEAD}, backbone frozen, FLAG_BATCHNORM_EVAL={flag}, "
               f"FLAG_RELU_INPLACE={cfg.FLAG_RELU_INPLACE}")
        A = train_variant(ctx, bn_eval=flag, epochs=epochs)
        hist = A["history"]
        r.values["configured"] = {k: v for k, v in A.items() if k != "final_state"}
        expected_params = 1024 * cfg.NUM_CLASSES + cfg.NUM_CLASSES
        (r.ok if A["n_trainable"] == expected_params else r.fail)(
            f"trainable parameters: {A['n_trainable']} (expected {expected_params}: the Linear head only)")
        if flag:
            (r.ok if A["bn_drift_final"] == 0.0 else r.fail)(
                f"BatchNorm running statistics of the backbone unchanged after training "
                f"(max drift {A['bn_drift_final']:.2e})")
        else:
            r.info(f"BatchNorm statistics drifted by {A['bn_drift_final']:.3f} (FLAG_BATCHNORM_EVAL=False)")
        final = hist[-1]
        accs = [h["val_acc"] for h in hist]
        r.values.update(final_val_acc=final["val_acc"], final_val_loss=final["val_loss"],
                        val_acc_per_epoch=accs,
                        epoch_train_seconds=float(np.mean([h["train_seconds"] for h in hist[1:] or hist])),
                        epoch_val_seconds=float(np.mean([h["val_seconds"] for h in hist[1:] or hist])),
                        val_seconds_per_image=float(np.mean([h["val_seconds"] for h in hist]) / len(D.idx_val)))
        (r.ok if final["val_acc"] >= cfg.MIN_VAL_ACCURACY else r.warn)(
            f"validation accuracy of the FINAL epoch: {final['val_acc']:.4f} vs MIN_VAL_ACCURACY "
            f"{cfg.MIN_VAL_ACCURACY}" + ("" if final["val_acc"] >= cfg.MIN_VAL_ACCURACY
                                          else " -> below threshold (warning only)"))
        if len(accs) > 1:
            below = [i + 1 for i, a in enumerate(accs) if i >= 1 and a < cfg.MIN_VAL_ACCURACY]
            (r.ok if not below else r.info)(
                "validation accuracy from epoch 2 on: " + ", ".join(f"{a:.4f}" for a in accs[1:])
                + ("" if not below else f"  (below threshold at epochs {below})"))
        r.info(f"time per epoch: train {r.values['epoch_train_seconds']:.0f}s + validation "
               f"{r.values['epoch_val_seconds']:.0f}s (tensor preprocessing on the GPU)")
        if len(hist) >= 2:
            drop = (hist[-2]["val_loss"] - hist[-1]["val_loss"]) / hist[-2]["val_loss"]
            r.values["last_epoch_val_loss_drop"] = drop
            if drop > THRESHOLDS["loss_still_dropping_rel"]:
                r.info(f"validation loss still dropping clearly ({drop:.1%} last epoch): "
                       "NUM_EPOCHS could be raised (operator decision)")
            else:
                r.info(f"validation loss has flattened ({drop:.1%} last epoch): "
                       f"NUM_EPOCHS={cfg.NUM_EPOCHS} is enough")
        ctx.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": A["final_state"],
                    "snapshot": {k: getattr(cfg, k) for k in (
                        "FREEZE_BACKBONE", "DENSENET_SIZE", "NUM_CLASSES", "BACKBONE_WEIGHTS", "SEED",
                        "VAL_SIZE", "BATCH_SIZE", "NUM_EPOCHS", "OPTIMIZER", "LEARNING_RATE_HEAD",
                        "FLAG_BATCHNORM_EVAL")}},
                   ctx.ckpt_path)
        ctx.cache.__dict__.pop("model", None)         # later checks reload the trained head
        r.info(f"phase-0 head (final epoch) saved to {ctx.ckpt_path} "
               "(do not commit; used by checks 3,4,11,12)")
        # 6c: the opposite BatchNorm setting, for comparison
        if not ctx.args.no_bn_comparison:
            r.info(f"training OPPOSITE setting for comparison: FLAG_BATCHNORM_EVAL={not flag}")
            B = train_variant(ctx, bn_eval=not flag, epochs=epochs)
            bf = B["history"][-1]
            r.values["opposite_bn_setting"] = {"flag": not flag, "final_val_acc": bf["val_acc"],
                                               "bn_drift_final": B["bn_drift_final"]}
            r.info(f"opposite setting: final val acc {bf['val_acc']:.4f} (configured: "
                   f"{final['val_acc']:.4f}), BatchNorm drift {B['bn_drift_final']:.3f}")
            if bf["val_acc"] - final["val_acc"] > 0.005:
                r.warn(f"the OPPOSITE setting (FLAG_BATCHNORM_EVAL={not flag}) is clearly more "
                       "accurate: whether to change the config is an operator decision")
            else:
                r.ok(f"the configured setting (FLAG_BATCHNORM_EVAL={flag}) is at least as accurate")


# ----------------------------------------------------------------------------------------
# CHECK 14 - checkpoint round trip (does the reloaded model behave like the trained one?)
# ----------------------------------------------------------------------------------------
def check_14(ctx):
    with ctx.rec.check(14) as r:
        cfg, D, S, dev = ctx.cfg, load_data(ctx), ctx.cfg.DENSENET_SIZE, ctx.device
        model = get_model(ctx)
        if not getattr(ctx.cache, "model_trained", False):
            r.warn("no phase-0 checkpoint found: run check 6 first")
            return
        X, y = D.X_trainval.to(dev), D.y_trainval.to(dev)
        _, acc = evaluate(model, X, y, torch.as_tensor(D.idx_val, device=dev), S)
        ref = ctx.rec.values(6).get("final_val_acc")
        r.values.update(reloaded_val_acc=acc, training_run_val_acc=ref)
        if ref is None:
            r.info(f"reloaded checkpoint: validation accuracy {acc:.4f} (no check-6 reference to compare)")
        else:
            (r.ok if abs(acc - ref) <= 1e-3 else r.fail)(
                f"reloaded checkpoint reproduces the validation accuracy of the training run: "
                f"{acc:.4f} vs {ref:.4f}")
        _, x, pred, _ = sample_inputs(ctx)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()
        samples = []
        for c, p, pr in zip((7, 4, 1), pred, probs):
            samples.append({"true": c, "predicted": int(p), "confidence": float(pr[p])})
            (r.ok if int(p) == c else r.warn)(
                f"sample test image of class {c}: predicted {int(p)} with confidence {pr[p]:.1%}"
                + ("" if int(p) == c else "  (WRONG: a ~3% event at 97% accuracy; the explanation "
                   "will be about the predicted class)"))
        r.values["samples"] = samples


# ----------------------------------------------------------------------------------------
# CHECK 7 - LIME grid (visual)
# ----------------------------------------------------------------------------------------
def check_7(ctx):
    with ctx.rec.check(7) as r:
        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries, quickshift
        cfg, D, S = ctx.cfg, load_data(ctx), ctx.cfg.DENSENET_SIZE
        y = D.y_test.numpy()
        picks = [int(np.where(y == c)[0][0]) for c in (1, 4, 7, 9)]
        seg = grid_segments(S, cfg.LIME_GRID_CELLS)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
        stats = []
        for col, idx in enumerate(picks):
            img = F.interpolate(D.X_test[idx:idx + 1], size=(S, S), mode="bilinear",
                                align_corners=False)[0, 0].numpy()
            rgb = np.stack([img] * 3, axis=-1)
            ink = img > cfg.MASK_THRESHOLD
            cells_with_ink = len({int(s) for s in np.unique(seg[ink])})
            stats.append(cells_with_ink)
            axes[0, col].imshow(mark_boundaries(rgb, seg, color=(1, 0.2, 0.2)))
            axes[0, col].set_title(f"grid {cfg.LIME_GRID_CELLS}x{cfg.LIME_GRID_CELLS}\n"
                                   f"{cells_with_ink}/{seg.max() + 1} cells touch ink", fontsize=9)
            qs = quickshift(rgb, kernel_size=3, max_dist=6, ratio=0.5, convert2lab=False)
            axes[1, col].imshow(mark_boundaries(rgb, qs, color=(0.2, 0.6, 1)))
            axes[1, col].set_title(f"quickshift\n{qs.max() + 1} segments", fontsize=9)
        for ax in axes.ravel():
            ax.axis("off")
        save_fig(fig, ctx.out_dir / "phase0_07_lime_grid.png")
        r.values["cells_touching_ink"] = stats
        r.info(f"grid cells touching the digit (digits 1,4,7,9): {stats}")
        r.review("look at phase0_07_lime_grid.png: does the 8x8 grid separate the digit into a "
                 "useful number of pieces (not 1-2 cells, not the whole image)? and is the quickshift "
                 "row clearly worse (design D11 says so)? Confirm or change LIME_GRID_CELLS.")


# ----------------------------------------------------------------------------------------
# CHECK 8 - digit mask (visual)
# ----------------------------------------------------------------------------------------
def check_8(ctx):
    with ctx.rec.check(8) as r:
        import matplotlib.pyplot as plt
        cfg, D = ctx.cfg, load_data(ctx)
        thresholds, dilations = (0.05, 0.1, 0.2, 0.3), (0, 1, 2)
        sample = D.X_trainval[torch.as_tensor(D.idx_train[:200])][:, 0].numpy()
        table = {}
        for t in thresholds:
            for d in dilations:
                fr = np.array([digit_mask(im, t, d).mean() for im in sample])
                empty = int(sum(digit_mask(im, t, d).sum() == 0 for im in sample))
                table[f"thr={t},dil={d}"] = dict(mean=float(fr.mean()), std=float(fr.std()), empty=empty)
        r.values["mask_fraction_table"] = table
        for k, v in table.items():
            r.info(f"{k}: mask covers {v['mean']:.1%} +/- {v['std']:.1%} of the 784 pixels, empty masks: {v['empty']}")
        fr = np.array([digit_mask(im, cfg.MASK_THRESHOLD, cfg.MASK_DILATION).mean() for im in sample])
        lo, hi = THRESHOLDS["mask_fraction_range"]
        n_empty = int(sum(digit_mask(im, cfg.MASK_THRESHOLD, cfg.MASK_DILATION).sum() == 0 for im in sample))
        (r.ok if n_empty == 0 else r.fail)(f"configured mask is never empty over 200 images (empty: {n_empty})")
        out_of_range = int(((fr < lo) | (fr > hi)).sum())
        (r.ok if out_of_range == 0 else r.warn)(
            f"configured mask covers between {fr.min():.1%} and {fr.max():.1%} of the pixels "
            f"({out_of_range} images outside {lo:.0%}-{hi:.0%})")
        y = D.y_trainval[torch.as_tensor(D.idx_train[:200])].numpy()
        picks = [int(np.where(y == c)[0][0]) for c in range(10)]
        fig, axes = plt.subplots(3, 10, figsize=(14, 4.6))
        for col, i in enumerate(picks):
            im = sample[i]
            axes[0, col].imshow(im, cmap="gray")
            axes[1, col].imshow(digit_mask(im, cfg.MASK_THRESHOLD, 0), cmap="gray")
            axes[2, col].imshow(digit_mask(im, cfg.MASK_THRESHOLD, cfg.MASK_DILATION), cmap="gray")
        for ax, label in zip(axes[:, 0], ("image", f"thr={cfg.MASK_THRESHOLD}, dil=0",
                                          f"thr={cfg.MASK_THRESHOLD}, dil={cfg.MASK_DILATION}")):
            ax.set_ylabel(label, fontsize=8)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        save_fig(fig, ctx.out_dir / "phase0_08_mask.png")
        r.review("look at phase0_08_mask.png: does the dilated mask cover the whole stroke and its "
                 "immediate halo without swallowing the background? Confirm MASK_THRESHOLD and "
                 "MASK_DILATION. NOTE: dilation uses a 3x3 square element (8-connected), a detail "
                 "the design did not specify (errata E17).")


# ----------------------------------------------------------------------------------------
# CHECK 11 - Occlusion
# ----------------------------------------------------------------------------------------
def check_11(ctx):
    with ctx.rec.check(11) as r:
        from captum.attr import Occlusion
        cfg, dev, S = ctx.cfg, ctx.device, ctx.cfg.DENSENET_SIZE
        w, s = cfg.OCCLUSION_WINDOW, cfg.OCCLUSION_STRIDE
        # a) overlap semantics on a model whose answer is known: f(x) = sum of pixels
        ones, zeros = torch.ones(1, 3, S, S, device=dev), torch.zeros(1, 3, S, S, device=dev)
        att = Occlusion(lambda inp: inp.sum(dim=(1, 2, 3))).attribute(
            ones, sliding_window_shapes=(3, w, w), strides=(3, s, s), baselines=zeros)
        uniq = np.unique(np.round(att.cpu().numpy(), 3))
        mode = "averaged" if len(uniq) == 1 else "summed (or other)"
        r.values["overlap_mode"] = mode
        r.info(f"overlapping windows are {mode} (distinct values on the toy model: {uniq[:5].tolist()})")
        # b) real model
        model, x, pred, _ = sample_inputs(ctx)
        calls = {"n": 0}

        def counted(inp):
            calls["n"] += inp.shape[0]
            return model(inp)

        xi = x[:1]
        att, sec = timed(lambda: Occlusion(counted).attribute(
            xi, sliding_window_shapes=(3, w, w), strides=(3, s, s),
            baselines=black_normalized(S, dev), target=int(pred[0]),
            perturbations_per_eval=32, show_progress=False), dev)
        expected = ((S - w) // s + 1) ** 2
        r.values.update(occlusion_seconds_per_image=sec, forward_images=calls["n"], expected_windows=expected)
        (r.ok if calls["n"] in (expected, expected + 1) else r.warn)(
            f"forward images used: {calls['n']} (windows expected: {expected}, +1 for the original image)")
        a = att[0].detach()
        chan_diff = max((a[0] - a[1]).abs().max().item(), (a[0] - a[2]).abs().max().item())
        (r.ok if chan_diff < 1e-6 else r.warn)(
            f"windows cover all 3 channels together: channel maps identical (max diff {chan_diff:.1e}) "
            "-> take the mean over channels")
        (r.ok if att.shape == (1, 3, S, S) and torch.isfinite(att).all() else r.fail)(
            f"output shape {tuple(att.shape)}, finite")
        r.info(f"Occlusion {sec:.2f}s/img")


# ----------------------------------------------------------------------------------------
# CHECK 12 - resampling to 28x28
# ----------------------------------------------------------------------------------------
def check_12(ctx):
    with ctx.rec.check(12) as r:
        import matplotlib.pyplot as plt
        cfg, S = ctx.cfg, ctx.cfg.DENSENET_SIZE
        rng = np.random.RandomState(cfg.SEED)
        # a) 28 -> S (bilinear) -> 28 (area): a single bright pixel must come back in place
        positions = [(0, 0), (0, 27), (27, 0), (27, 27)] + [tuple(rng.randint(0, 28, 2)) for _ in range(96)]
        hits = 0
        for (i, j) in positions:
            delta = np.zeros((MNIST_SIDE, MNIST_SIDE), dtype=np.float32)
            delta[i, j] = 1.0
            up = F.interpolate(torch.from_numpy(delta)[None, None], size=(S, S), mode="bilinear",
                               align_corners=False)
            back = to_grid28(up[0, 0].numpy(), S)
            hits += int(np.unravel_index(back.argmax(), back.shape) == (i, j))
        r.values["alignment_28_hits"] = f"{hits}/{len(positions)}"
        (r.ok if hits == len(positions) else r.fail)(
            f"28 -> {S} -> 28 round trip keeps every test pixel in place: {hits}/{len(positions)}")
        # b) Grad-CAM native cells land inside their own block after the rule
        n = S // 32
        block = MNIST_SIDE / n
        inside = 0
        for i in range(n):
            for j in range(n):
                cell = np.zeros((n, n), dtype=np.float32)
                cell[i, j] = 1.0
                g = to_grid28(cell, S)
                pi, pj = np.unravel_index(g.argmax(), g.shape)
                inside += int(int(i * block) <= pi < int((i + 1) * block) + 1
                              and int(j * block) <= pj < int((j + 1) * block) + 1)
        r.values["gradcam_cells_inside"] = f"{inside}/{n * n}"
        (r.ok if inside == n * n else r.fail)(
            f"each {n}x{n} Grad-CAM cell peaks inside its own block of the 28x28 grid: {inside}/{n * n}")
        # c) ties: bilinear (rule) vs nearest, on a real Grad-CAM map
        model, x, pred, x28 = sample_inputs(ctx)
        g = explain_gradcam(model, x[:1], pred[0], cfg.GRADCAM_LAYER)
        rule = to_grid28(g, S)
        nearest = F.interpolate(torch.from_numpy(g)[None, None].float(), size=(MNIST_SIDE, MNIST_SIDE),
                                mode="nearest")[0, 0].numpy()
        mask = digit_mask(x28[0, 0].numpy(), cfg.MASK_THRESHOLD, cfg.MASK_DILATION)
        u_rule, u_near = len(np.unique(np.round(rule[mask], 7))), len(np.unique(np.round(nearest[mask], 7)))
        r.values.update(unique_values_in_mask_rule=u_rule, unique_values_in_mask_nearest=u_near,
                        mask_pixels=int(mask.sum()))
        r.info(f"distinct Grad-CAM values inside the digit mask ({int(mask.sum())} pixels): "
               f"bilinear rule {u_rule}, nearest {u_near} (fewer values = more ties in Spearman/IoU)")
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
        for ax, im, title in zip(axes, (x28[0, 0].numpy(), g, rule, nearest),
                                 ("image", f"Grad-CAM native {g.shape[0]}x{g.shape[1]}",
                                  "rule: bilinear->128->area 28", "nearest 28")):
            ax.imshow(im, cmap="gray" if title == "image" else "viridis")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        save_fig(fig, ctx.out_dir / "phase0_12_resampling.png")
        r.review("look at phase0_12_resampling.png: is the rule's 28x28 map a faithful, smooth "
                 "version of the native Grad-CAM (and clearly better than blocky nearest)? "
                 "Confirm the rule (errata E03).")


# ----------------------------------------------------------------------------------------
# CHECK 9 / 10 - timings, N_QUANT, decisions
# ----------------------------------------------------------------------------------------
def collect_timings(ctx):
    rec, cfg = ctx.rec, ctx.cfg
    c3 = rec.values(3)
    lime_b = choose_backend(c3, "lime", cfg.LIME_BACKEND, "captum" if cfg.LIME_BACKEND == "lime" else "lime")
    shap_b = choose_backend(c3, "shap", cfg.SHAP_BACKEND, "captum" if cfg.SHAP_BACKEND == "shap" else "shap")
    missing = []

    def get(value, label):
        if value is None:
            missing.append(label)
            return 0.0
        return float(value)

    lime_t = (c3.get("backends", {}).get(f"lime/{lime_b}", {}) or {}).get("seconds_median") if lime_b else None
    shap_t = (c3.get("backends", {}).get(f"shap/{shap_b}", {}) or {}).get("seconds_median") if shap_b else None
    timings = {
        "gradcam": get(rec.values(2).get("gradcam_seconds"), "check 2 (Grad-CAM)"),
        "ig": get(rec.values(4).get("ig_seconds_per_image"), "check 4 (IG)"),
        "occlusion": get(rec.values(11).get("occlusion_seconds_per_image"), "check 11 (Occlusion)"),
        "lime": get(lime_t, "check 3 (LIME backend)"),
        "shap": get(shap_t, "check 3 (SHAP backend)"),
        "saliency": float(rec.values(13).get("saliency_seconds_per_image") or 0.0),
        "lr_fit": get(rec.values(5).get("fit_seconds"), "check 5 (LR fit)"),
        "epoch_train": get(rec.values(6).get("epoch_train_seconds"), "check 6 (epoch time)"),
        "epoch_val": get(rec.values(6).get("epoch_val_seconds"), "check 6 (validation time)"),
        "val_seconds_per_image": get(rec.values(6).get("val_seconds_per_image"), "check 6 (inference time)"),
    }
    return timings, lime_b, shap_b, missing


def check_9(ctx):
    with ctx.rec.check(9) as r:
        cfg = ctx.cfg
        t, lime_b, shap_b, missing = collect_timings(ctx)
        if missing:
            r.warn("estimate incomplete, run first: " + ", ".join(missing))
        n_quant, fits = pick_n_quant(t["gradcam"] + t["ig"] + t["occlusion"] + t["lime"] + t["shap"]
                                     + t.get("saliency", 0.0),
                                     THRESHOLDS["quant_stage_max_minutes"])
        est = estimate_run_all(t, cfg, n_quant)
        r.values.update(timings=t, estimate_seconds=est, proposed_n_quant=n_quant,
                        lime_backend_used=lime_b, shap_backend_used=shap_b)
        r.info(f"seconds per image, all five techniques: {est['per_image_all_techniques']:.1f} "
               f"(LIME backend {lime_b}, SHAP backend {shap_b})")
        (r.ok if fits else r.warn)(
            f"N_QUANT = {n_quant}: quantitative stage ~{est['quant'] / 60:.1f} min "
            f"(budget {THRESHOLDS['quant_stage_max_minutes']:.0f} min)"
            + ("" if fits else "  [even the smallest candidate exceeds the budget]"))
        for label, key, limit in (("MODEL_MODE=load", "load_total", THRESHOLDS["run_all_load_max_minutes"]),
                                  ("MODEL_MODE=train", "train_total", THRESHOLDS["run_all_train_max_minutes"])):
            minutes = est[key] / 60
            (r.ok if minutes <= limit else r.warn)(
                f"'Esegui tutto' with {label}: ~{minutes:.0f} min (limit {limit:.0f} min)")
        r.info(f"breakdown: gallery {est['gallery'] / 60:.1f} min, test inference "
               f"{est['test_inference'] / 60:.1f} min, fixed overhead {FIXED_OVERHEAD_SECONDS / 60:.0f} min")
        if ctx.quick:
            r.warn("quick run: these timings come from random weights and tiny data, NOT valid for decisions")


def check_10(ctx):
    with ctx.rec.check(10) as r:
        cfg, rec = ctx.cfg, ctx.rec
        t, lime_b, shap_b, missing = collect_timings(ctx)
        n_quant = rec.values(9).get("proposed_n_quant")
        c6, c5 = rec.values(6), rec.values(5)
        rows = []

        def row(name, initial, proposed, basis, confirmation):
            rows.append({"parameter": name, "initial": initial, "proposed": proposed,
                         "basis": basis, "confirmation": confirmation})

        auto, manual = "criterion met (automatic)", "operator confirmation required"
        row("LIME_BACKEND", cfg.LIME_BACKEND, lime_b or "NONE WORKS",
            "check 3", auto if lime_b else "FAIL: no LIME backend works")
        row("SHAP_BACKEND", cfg.SHAP_BACKEND, shap_b or "NONE WORKS",
            "check 3", auto if shap_b else "FAIL: no SHAP backend works")
        row("N_QUANT", cfg.N_QUANT, n_quant, "check 9 (time budget)",
            auto if n_quant else "run checks 3,4,2,11,5,6 first")
        if c6:
            hist_ok = c6.get("final_val_acc") is not None
            drop = c6.get("last_epoch_val_loss_drop")
            row("NUM_EPOCHS", cfg.NUM_EPOCHS, cfg.NUM_EPOCHS,
                "check 6: last-epoch val-loss drop " + (f"{drop:.1%}" if drop is not None else "n/a (1 epoch)"),
                manual if (drop or 0) > THRESHOLDS["loss_still_dropping_rel"] else auto)
            row("MIN_VAL_ACCURACY", cfg.MIN_VAL_ACCURACY, cfg.MIN_VAL_ACCURACY,
                f"check 6: final-epoch validation accuracy {c6['final_val_acc']:.4f}" if hist_ok
                else "check 6 missing",
                auto if hist_ok and c6["final_val_acc"] >= cfg.MIN_VAL_ACCURACY else manual)
            for name, val in (("BATCH_SIZE", cfg.BATCH_SIZE), ("LEARNING_RATE_HEAD", cfg.LEARNING_RATE_HEAD)):
                row(name, val, val, f"check 6 ({c6.get('epoch_train_seconds', 0):.0f}s/epoch)", auto)
            opp = c6.get("opposite_bn_setting")
            if opp and hist_ok:
                better = opp["final_val_acc"] - c6["final_val_acc"] > 0.005
                row("FLAG_BATCHNORM_EVAL", cfg.FLAG_BATCHNORM_EVAL, cfg.FLAG_BATCHNORM_EVAL,
                    f"check 6: configured {c6['final_val_acc']:.4f} vs opposite {opp['final_val_acc']:.4f}",
                    manual if better else auto)
        c2 = rec.values(2)
        if c2.get("relu_probes") is not None:
            safe_all = c2.get("inplace_safe_for_all")
            row("FLAG_RELU_INPLACE", cfg.FLAG_RELU_INPLACE, cfg.FLAG_RELU_INPLACE,
                "check 2: in-place model gives the same maps as the patched one"
                if safe_all else "check 2: in-place model fails or gives different maps (see check 2)",
                auto if (safe_all or not cfg.FLAG_RELU_INPLACE) else "FAIL: set FLAG_RELU_INPLACE=False")
        if c5.get("runs"):
            d = c5["runs"][str(cfg.LR_C)]
            row("LR_C", cfg.LR_C, cfg.LR_C, f"check 5: val acc {d['val_accuracy']:.4f}, maps to review",
                manual)
            row("LR_MAX_ITER", cfg.LR_MAX_ITER, cfg.LR_MAX_ITER,
                f"check 5: n_iter {d['n_iter']}, ConvergenceWarning={d['convergence_warning']}",
                manual if d["convergence_warning"] else auto)
        c4 = rec.values(4)
        if c4.get("ig_rel_error"):
            worst = max(c4["ig_rel_error"])
            prop = c4.get("ig_steps_proposed") or cfg.IG_STEPS
            row("IG_STEPS", cfg.IG_STEPS, prop,
                f"check 4: completeness error {worst:.2%} at {cfg.IG_STEPS} steps (sweep in check 4)",
                auto if worst <= THRESHOLDS["ig_completeness_pass_rel"] else manual)
        c13 = rec.values(13)
        if c13.get("saliency_seconds_per_image") is not None:
            row("Saliency as 6th technique (E22)", "not in design", "operator decision",
                f"check 13: {c13['saliency_seconds_per_image']:.3f}s/img, share inside the digit "
                f"{np.mean(c13['mass_in_mask_saliency']):.0%} vs mask area {np.mean(c13['mask_area_fraction']):.0%}",
                manual)
        c11 = rec.values(11)
        if c11:
            row("OCCLUSION_WINDOW/STRIDE", f"{cfg.OCCLUSION_WINDOW}/{cfg.OCCLUSION_STRIDE}",
                f"{cfg.OCCLUSION_WINDOW}/{cfg.OCCLUSION_STRIDE}",
                f"check 11: {c11.get('forward_images')} forwards, overlap {c11.get('overlap_mode')}", auto)
        row("SHAP_NUM_SAMPLES", cfg.SHAP_NUM_SAMPLES, cfg.SHAP_NUM_SAMPLES, "check 3 timing", auto)
        row("SHAP_NUM_BACKGROUND", cfg.SHAP_NUM_BACKGROUND, cfg.SHAP_NUM_BACKGROUND, "check 3 timing", auto)
        row("LIME_NUM_SAMPLES", cfg.LIME_NUM_SAMPLES, cfg.LIME_NUM_SAMPLES, "check 3 timing", auto)
        row("LIME_GRID_CELLS", cfg.LIME_GRID_CELLS, cfg.LIME_GRID_CELLS, "check 7 (visual)", manual)
        row("MASK_THRESHOLD / MASK_DILATION", f"{cfg.MASK_THRESHOLD}/{cfg.MASK_DILATION}",
            f"{cfg.MASK_THRESHOLD}/{cfg.MASK_DILATION}", "check 8 (visual)", manual)
        row("28x28 resampling rule (E03)", "bilinear->128->area", "bilinear->128->area",
            "check 12 (numeric + visual)", manual)
        rec.data["decisions"] = rows
        rec.save()
        lines = ["| Parameter | Initial | Proposed | Basis | Confirmation |", "|---|---|---|---|---|"]
        lines += [f"| {x['parameter']} | {x['initial']} | {x['proposed']} | {x['basis']} | {x['confirmation']} |"
                  for x in rows]
        table = "\n".join(lines)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        (ctx.out_dir / "phase0_summary.md").write_text(
            "# Phase 0 - table of fixed values\n\n" + table + "\n")
        print("\n" + table)
        changed = {x["parameter"]: x["proposed"] for x in rows
                   if x["parameter"] in ("LIME_BACKEND", "SHAP_BACKEND", "N_QUANT")
                   and x["proposed"] not in (x["initial"], "NONE WORKS", None)}
        r.values["proposed_config_overrides"] = changed
        if changed:
            r.info("proposed edits to config.py (NOT applied automatically): " + json.dumps(changed))
        else:
            r.info("no change to config.py proposed by the automatic criteria")
        if missing:
            r.warn("table incomplete, missing measurements from: " + ", ".join(missing))
        if not (lime_b and shap_b):
            r.fail("a technique has no working backend: see check 3")
        if ctx.quick:
            r.warn("quick run: the table is NOT valid for decisions")
        pending = sum(1 for x in rows if x["confirmation"] == manual)
        r.review(f"{pending} rows need operator confirmation (figures of checks 5, 7, 8, 12 and flagged rows)")


CHECKS = {1: check_1, 2: check_2, 3: check_3, 4: check_4, 5: check_5, 6: check_6, 7: check_7,
          8: check_8, 9: check_9, 10: check_10, 11: check_11, 12: check_12, 13: check_13, 14: check_14}


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase zero checks")
    p.add_argument("--quick", action="store_true", help="random weights, tiny data (script smoke run)")
    p.add_argument("--checks", type=str, default="", help="comma-separated ids, e.g. 1,2,5")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--lr-c-sweep", type=lambda s: [float(v) for v in s.split(",") if v], default=[])
    p.add_argument("--no-bn-comparison", action="store_true")
    p.add_argument("--out-dir", type=str, default="")
    args, _ = p.parse_known_args(argv)               # tolerant: works under %run and in notebooks
    return args


def main(argv=None):
    args = parse_args(argv)
    ctx = make_ctx(args)
    if torch is None:
        print(f"torch could not be imported ({_TORCH_ERROR}): only check 1 can run")
    set_seed(ctx.cfg.SEED)
    wanted = [int(v) for v in args.checks.split(",") if v] or ORDER
    unknown = [c for c in wanted if c not in CHECKS]
    if unknown:
        raise SystemExit(f"unknown check ids: {unknown}")
    if args.quick:
        print("QUICK RUN: random weights and tiny data. Results are NOT valid for decisions.")
    ctx.rec.data["meta"] = {"config": dataclasses.asdict(ctx.cfg), "quick": args.quick,
                            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for cid in [c for c in ORDER if c in wanted]:
        if torch is None and cid != 1:
            continue
        CHECKS[cid](ctx)
    ctx.rec.save()
    print("\n" + "=" * 70 + "\nSUMMARY")
    for cid in ORDER:
        entry = ctx.rec.data["checks"].get(str(cid))
        if entry:
            print(f"  check {cid:2d}  {entry['status']:<7} {entry['name']}  ({entry['seconds']}s)")
    review = [c for c in ORDER if ctx.rec.data["checks"].get(str(c), {}).get("status") == "REVIEW"
              or any(s == "REVIEW" for s, _ in ctx.rec.data["checks"].get(str(c), {}).get("lines", []))]
    print(f"\nFigures to judge by eye (REVIEW): checks {review}")
    try:
        import matplotlib.pyplot as plt
        plt.close("all")                         # no stray empty figure after the run
    except Exception:
        pass
    summary = ctx.out_dir / "phase0_summary.md"
    print(f"Results: {ctx.rec.path}" + (f"  |  Table: {summary}" if summary.exists() else ""))
    return ctx.rec.data


if __name__ == "__main__":
    main()
