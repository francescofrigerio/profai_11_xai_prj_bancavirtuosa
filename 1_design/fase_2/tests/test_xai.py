"""xai/: the six techniques obey one contract and look where they should.

Most tests use a TOY model whose answer is known by construction (the planted signal): detector 0
sees only the top-left quadrant, detector 1 only the bottom-right one, and logit k = detector k.
If a technique cannot find a region we planted ourselves, it cannot be trusted on a DenseNet.
"""
import copy
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from helpers import synthetic_data_bundle, tiny_cfg
from checks import ContractError
from data import black_normalized, to_densenet_input
from xai import TECHNIQUE_NAMES, build_techniques
from xai.debug import make_gradcam_debug_callback
from xai.gradcam import GradCam
from xai.lime_explainer import grid_segments
from xai.shap_explainer import make_background

CFG = tiny_cfg(IG_STEPS=20, LIME_NUM_SAMPLES=1000, SHAP_NUM_SAMPLES=20, SHAP_NUM_BACKGROUND=8)
S = CFG.DENSENET_SIZE


class _Detectors(nn.Module):
    """(B,3,S,S) -> (B,2,8,8): channel 0 = brightness of the top-left quadrant, channel 1 = bottom-right."""

    def __init__(self):
        super().__init__()
        mask = torch.zeros(1, 2, 8, 8)
        mask[0, 0, :4, :4] = 1
        mask[0, 1, 4:, 4:] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.mean(1, keepdim=True), (8, 8)) * self.mask


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([("denseblock3", _Detectors()), ("norm5", nn.Identity())]))
        self.classifier = nn.Linear(2, 10)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.weight[0, 0] = self.classifier.weight[1, 1] = 0.05
            self.classifier.bias.zero_()
            self.classifier.bias[2:] = -10.0           # only classes 0 and 1 compete: a clear probability signal for LIME

    def forward(self, x):
        return self.classifier(self.features(x).sum(dim=(2, 3)))


def toy_world():
    """The toy model, one image with ink in BOTH quadrants, and a background for SHAP."""
    model = ToyNet().eval()
    x28 = torch.zeros(1, 1, 28, 28)
    x28[0, 0, 3:8, 3:8] = 1.0          # top-left block
    x28[0, 0, 18:23, 18:23] = 1.0      # bottom-right block
    x1 = to_densenet_input(x28, S)
    data = synthetic_data_bundle(CFG)
    # A background with ink EVERYWHERE: every baseline draw changes the SHAP estimate. With sparse
    # backgrounds (one image inking the observed quadrant) two seeds can draw the same counts and
    # the test would be unable to see the seed (first Colab run, E36).
    generator = torch.Generator().manual_seed(0)
    background = to_densenet_input(torch.rand(CFG.SHAP_NUM_BACKGROUND, 1, 28, 28, generator=generator), S)
    return model, x1, background, data


def supporting_mass(result):
    """(top-left, bottom-right) mass of the part of the map that SUPPORTS the class."""
    m = torch.from_numpy(np.clip(result.map, 0, None)).float()[None, None]
    m = F.interpolate(m, size=(S, S), mode="bilinear", align_corners=False)[0, 0].numpy()
    return float(m[:S // 2, :S // 2].sum()), float(m[S // 2:, S // 2:].sum())


def test_every_technique_returns_a_valid_contract_result():
    model, x1, background, _ = toy_world()
    techniques = build_techniques(model, CFG, background)
    assert tuple(techniques) == TECHNIQUE_NAMES and len(techniques) == 6
    for name, technique in techniques.items():
        r = technique.explain(x1, 0)
        assert r.technique == name and r.target == 0 and r.seconds > 0
        assert r.map.ndim == 2 and r.map.shape[0] == r.map.shape[1] <= S and np.isfinite(r.map).all()
        assert r.signed == technique.signed and (r.signed or r.map.min() >= 0)
        assert r.output_space == ("probability" if name == "lime" else "logit")
    # negative controls: the contract refuses what it must refuse
    t = techniques["saliency"]
    with pytest.raises(ContractError):
        t.explain(torch.rand(1, 3, S, S), 0)                        # not normalized
    with pytest.raises(ContractError):
        t.explain(torch.cat([x1, x1]), 0)                           # a batch
    with pytest.raises(ContractError):
        t.explain(x1, 10)                                           # not a class
    model.train()
    with pytest.raises(ContractError):
        t.explain(x1, 0)                                            # BatchNorm would corrupt it
    model.eval()


@pytest.mark.parametrize("name", list(TECHNIQUE_NAMES))
def test_planted_signal_is_found_and_follows_the_target_class(name):
    model, x1, background, _ = toy_world()
    technique = build_techniques(model, CFG, background, names=(name,))[name]
    tl0, br0 = supporting_mass(technique.explain(x1, 0))            # class 0 lives in the top-left quadrant
    tl1, br1 = supporting_mass(technique.explain(x1, 1))            # class 1 in the bottom-right one
    assert tl0 > 3 * br0, f"{name}: class 0 should be explained by the top-left quadrant ({tl0:.3g} vs {br0:.3g})"
    assert br1 > 3 * tl1, f"{name}: class 1 should be explained by the bottom-right quadrant ({br1:.3g} vs {tl1:.3g})"


def test_integrated_gradients_completeness_and_black_baseline():
    model, x1, background, _ = toy_world()
    r = build_techniques(model, CFG, background, names=("integrated_gradients",))["integrated_gradients"].explain(x1, 0)
    assert r.meta["completeness_error"] < 1e-3                       # exact on a linear model
    background_pixels = ((x1 - black_normalized(S, "cpu")).abs().amax(dim=1)[0] < 1e-9).numpy()
    assert background_pixels.any() and (r.map[background_pixels] == 0).all()   # nothing attributed to the black


def test_lime_and_shap_are_reproducible_and_really_use_the_seed():
    model, x1, background, _ = toy_world()
    for name in ("lime", "shap"):
        a = build_techniques(model, CFG, background, names=(name,))[name].explain(x1, 0).map
        b = build_techniques(model, CFG, background, names=(name,))[name].explain(x1, 0).map
        other_cfg = tiny_cfg(IG_STEPS=20, LIME_NUM_SAMPLES=1000, SHAP_NUM_SAMPLES=20, SHAP_NUM_BACKGROUND=8, SEED=CFG.SEED + 1)
        c = build_techniques(model, other_cfg, background, names=(name,))[name].explain(x1, 0).map
        assert np.allclose(a, b, rtol=1e-3, atol=1e-6 * np.abs(a).max()), f"{name}: same seed, different map"
        assert not np.allclose(a, c, rtol=1e-3, atol=1e-6 * np.abs(a).max()), f"{name}: the seed has no effect"


def test_shap_background_is_fixed_by_the_seed():
    data, cpu = synthetic_data_bundle(CFG), torch.device("cpu")
    a = make_background(data.X_trainval, data.split.idx_train, CFG, cpu)
    b = make_background(data.X_trainval, data.split.idx_train, CFG, cpu)
    c = make_background(data.X_trainval, data.split.idx_train, tiny_cfg(SHAP_NUM_BACKGROUND=8, SEED=CFG.SEED + 1), cpu)
    assert a.shape == (CFG.SHAP_NUM_BACKGROUND, 3, S, S)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_registry_is_strict():
    model, _, background, _ = toy_world()
    assert tuple(build_techniques(model, CFG, background, names=("shap", "gradcam"))) == ("gradcam", "shap")   # fixed order
    with pytest.raises(ValueError):
        build_techniques(model, CFG, background, names=("gradcam", "magic"))
    with pytest.raises(ValueError):
        build_techniques(model, CFG)                                  # SHAP without a background
    with pytest.raises(ValueError):
        GradCam(model, CFG, layer_name="features.nope")
    seg = grid_segments(64, 8)
    assert seg.shape == (64, 64) and (np.bincount(seg.ravel()) == 64).all()
    with pytest.raises(ValueError):
        grid_segments(64, 7)


def test_gradcam_native_resolution_on_a_real_densenet():
    from model import build_densenet
    cfg = tiny_cfg(DENSENET_SIZE=128)                                # the design size: 4x4 and 8x8 maps
    net = build_densenet(cfg, pretrained=False).eval()
    x1 = to_densenet_input(torch.rand(1, 1, 28, 28), 128)
    coarse = GradCam(net, cfg).explain(x1, 3)
    finer = GradCam(net, cfg, layer_name=cfg.GRADCAM_ALT_LAYER).explain(x1, 3)
    assert coarse.map.shape == (4, 4) and finer.map.shape == (8, 8)
    assert coarse.map.min() >= 0 and finer.map.min() >= 0


def test_relu_flag_does_not_change_the_maps_of_a_real_densenet():
    from model import SafeDenseNet, build_densenet, count_inplace_relu
    cfg = tiny_cfg(IG_STEPS=10)
    torch.manual_seed(0)
    inplace = build_densenet(cfg, pretrained=False).eval()
    patched = SafeDenseNet(copy.deepcopy(inplace)).eval()
    assert count_inplace_relu(inplace) > 0 and count_inplace_relu(patched) == 0
    x1 = to_densenet_input(torch.rand(1, 1, 28, 28), S)
    for name in ("gradcam", "integrated_gradients", "saliency"):
        a = build_techniques(inplace, cfg, names=(name,))[name].explain(x1, 3).map
        b = build_techniques(patched, cfg, names=(name,))[name].explain(x1, 3).map
        assert np.abs(a - b).max() <= 1e-3 * max(np.abs(b).max(), 1e-12), f"{name}: in-place ReLU changes the map"


def test_debug_callback_writes_the_gradcam_images():
    model, _, _, data = toy_world()
    with tempfile.TemporaryDirectory() as d:
        callback = make_gradcam_debug_callback(data, CFG, torch.device("cpu"), d, log=lambda *_: None)
        callback(model, 1)
        callback(model, 5)
        files = sorted(p.name for p in Path(d).iterdir())
        assert files == ["gradcam_debug_epoch1.png", "gradcam_debug_epoch5.png"]
        assert all((Path(d) / f).stat().st_size > 1000 for f in files)
