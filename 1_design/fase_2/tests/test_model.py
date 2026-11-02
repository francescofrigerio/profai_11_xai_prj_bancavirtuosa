"""model.py: head, ReLU flag, BatchNorm mode, optimizer, layers. CPU, random weights (no download)."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import tiny_cfg
from checks import assert_logits, assert_proba
from model import (SafeDenseNet, bn_buffers, bn_drift, build_densenet, count_inplace_relu,
                   enter_train_mode, freeze_backbone, make_optimizer, predict_logits, softmax_np,
                   trainable_parameters, validate_layers, wrap_for_flag)

CFG = tiny_cfg()


def batch(n=4, seed=0):
    return torch.from_numpy(np.random.RandomState(seed).randn(n, 3, 64, 64).astype(np.float32))


def test_head_and_logits():
    torch.manual_seed(0)
    net = build_densenet(CFG, pretrained=False).eval()
    assert isinstance(net.classifier, torch.nn.Linear) and net.classifier.out_features == 10
    with torch.no_grad():
        logits = net(batch())
    assert_logits(logits, CFG)
    assert_proba(torch.softmax(logits, 1), CFG)
    with pytest.raises(ValueError):
        build_densenet(tiny_cfg(BACKBONE_WEIGHTS="NOT_A_WEIGHT"), pretrained=True)


def test_relu_flag_changes_the_modules_not_the_numbers():
    torch.manual_seed(0)
    base = build_densenet(CFG, pretrained=False).eval()
    twin = copy.deepcopy(base)
    assert count_inplace_relu(base) > 0                              # torchvision as shipped
    assert wrap_for_flag(base, tiny_cfg(FLAG_RELU_INPLACE=True)) is base
    safe = wrap_for_flag(twin, tiny_cfg(FLAG_RELU_INPLACE=False))
    assert isinstance(safe, SafeDenseNet) and count_inplace_relu(safe) == 0
    x = batch()
    with torch.no_grad():
        assert torch.allclose(base(x), safe(x), atol=1e-5)           # same logits
    fresh = SafeDenseNet(build_densenet(CFG, pretrained=False))
    fresh.load_state_dict(base.state_dict())                         # strict: same parameter names


def test_freezing_and_optimizer():
    net = build_densenet(CFG, pretrained=False)
    freeze_backbone(net)
    assert sum(p.numel() for p in trainable_parameters(net)) == 1024 * 10 + 10   # the Linear head only
    opt = make_optimizer(net, CFG)
    assert len(opt.param_groups[0]["params"]) == 2 and opt.param_groups[0]["lr"] == CFG.LEARNING_RATE_HEAD
    unfrozen = build_densenet(CFG, pretrained=False)
    opt_all = make_optimizer(unfrozen, tiny_cfg(FREEZE_BACKBONE=False))
    assert len(opt_all.param_groups[0]["params"]) == len(list(unfrozen.parameters()))
    assert opt_all.param_groups[0]["lr"] == CFG.LEARNING_RATE_FULL


def test_batchnorm_mode_follows_the_flags_and_survives_a_validation_pass():
    net = build_densenet(CFG, pretrained=False)
    bns = [m for m in net.features.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    enter_train_mode(net, tiny_cfg(FLAG_BATCHNORM_EVAL=True))
    assert all(not m.training for m in bns) and net.classifier.training
    net.eval()                                                       # what a validation pass does
    enter_train_mode(net, tiny_cfg(FLAG_BATCHNORM_EVAL=True))        # re-applied at every epoch
    assert all(not m.training for m in bns) and net.classifier.training
    enter_train_mode(net, tiny_cfg(FLAG_BATCHNORM_EVAL=False))
    assert all(m.training for m in bns)
    enter_train_mode(net, tiny_cfg(FREEZE_BACKBONE=False, FLAG_BATCHNORM_EVAL=True))
    assert all(m.training for m in bns)                              # the flag only acts on a frozen backbone


def test_batchnorm_statistics_are_frozen_by_the_flag_and_drift_without_it():
    for flag, expect_drift in ((True, False), (False, True)):        # the False case is the negative control
        torch.manual_seed(0)
        net = build_densenet(CFG, pretrained=False)
        freeze_backbone(net)
        enter_train_mode(net, tiny_cfg(FLAG_BATCHNORM_EVAL=flag))
        before = bn_buffers(net)
        loss = torch.nn.functional.cross_entropy(net(batch(8)), torch.arange(8))
        loss.backward()
        assert (bn_drift(before, bn_buffers(net)) > 0) == expect_drift


def test_eval_is_deterministic_and_each_image_is_independent():
    torch.manual_seed(0)
    net = build_densenet(CFG, pretrained=False).eval()
    x = batch(6)
    with torch.no_grad():
        a, b = net(x), net(x)
        alone = torch.cat([net(x[i:i + 1]) for i in range(6)])
    assert torch.equal(a, b)
    assert torch.allclose(a, alone, atol=1e-4)                       # a train-mode BatchNorm would break this


def test_layers_exist_and_prediction_helper():
    net = build_densenet(CFG, pretrained=False).eval()
    validate_layers(net, CFG)
    with pytest.raises(ValueError):
        validate_layers(net, tiny_cfg(GRADCAM_LAYER="features.nope"))
    x28 = torch.rand(5, 1, 28, 28)
    logits = predict_logits(net, x28, CFG, torch.device("cpu"), batch_size=2)
    assert logits.shape == (5, 10)
    assert_proba(softmax_np(logits), CFG)
