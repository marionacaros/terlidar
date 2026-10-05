"""Smoke tests for the PointNet++ baseline and its LoRA variant.

They run on CPU with random tensors, so no dataset is needed. Run from the
repository root with:  pytest tests
"""
import os

import pytest
import torch

from conftest import REPO_ROOT
from src.LoRA.models.lora_pointnet2_params import LoraPointNet2
from src.LoRA.models.pointnet2_ss import PointNet2

N_POINTS = 2048  # the first set-abstraction layer samples 1024 points
N_LORA_TENSORS = 44  # 12 SA + 9 FP + 1 head layers, each with an A and a B matrix
CKPT_DIR = os.path.join(REPO_ROOT, 'checkpoints')
CKPT_BASELINE = os.path.join(CKPT_DIR, 'seg_02-24_15-52B29_NOclassifier.pt')
CKPT_FT = os.path.join(CKPT_DIR, 'seg_04-29_18-01lr0001RIB.pt')
CKPT_LORA = os.path.join(CKPT_DIR, 'loraPN2_07-23_12-12_32R32alph16.pt')

needs_checkpoints = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (CKPT_BASELINE, CKPT_FT, CKPT_LORA)),
    reason='shipped checkpoints not found')


def random_batch(num_feat, batch=2, n_points=N_POINTS):
    """[B, D, N] batch laid out like the datasets: x, y in [-1, 1], the rest in [0, 1]."""
    x = torch.rand(batch, num_feat, n_points)
    x[:, :2] = x[:, :2] * 2 - 1
    return x


def transfer_baseline_weights(lora_model, baseline_state, skip_head=True):
    """Positional copy used by train_lora_rib.py / train_lora_dales_pointnet2.py."""
    baseline_items = list(baseline_state.items())
    state = lora_model.state_dict()
    keys = list(state.keys())[N_LORA_TENSORS:-2] if skip_head else list(state.keys())[N_LORA_TENSORS:]
    for i, key in enumerate(keys):
        state[key] = baseline_items[i][1]
    lora_model.load_state_dict(state)
    return keys, baseline_items


@pytest.mark.parametrize('num_classes,num_feat', [(3, 8), (6, 5)])
def test_baseline_forward_shape(num_classes, num_feat):
    model = PointNet2(num_classes=num_classes, num_feat=num_feat).eval()
    with torch.no_grad():
        out, aux = model(random_batch(num_feat))
    assert out.shape == (2, N_POINTS, num_classes)
    assert aux is None
    # the model returns log-probabilities
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, N_POINTS), atol=1e-4)


@pytest.mark.parametrize('num_classes,num_feat', [(3, 8), (6, 5)])
def test_lora_forward_shape(num_classes, num_feat):
    model = LoraPointNet2(num_classes, num_feat, lora_fix_rank=True, lora_max_rank=8).eval()
    with torch.no_grad():
        out = model(random_batch(num_feat))
    assert out.shape == (2, N_POINTS, num_classes)
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, N_POINTS), atol=1e-4)


def test_lora_injection_trainable_parameters():
    rank = 8
    model = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=rank)
    params = dict(model.named_parameters())
    trainable = {n for n, p in params.items() if p.requires_grad}

    # only LoRA matrices and the classification head are trainable
    assert trainable == {n for n in params if 'lora' in n}
    lora_ab = [n for n in params if n.endswith('_A') or n.endswith('_B')]
    assert len(lora_ab) == N_LORA_TENSORS
    assert trainable == set(lora_ab) | {'lora_classifier.weight', 'lora_classifier.bias'}

    # LoRA matrices are registered first in the state dict (the weight transfer relies on it)
    assert list(model.state_dict().keys())[:N_LORA_TENSORS] == lora_ab

    for name in lora_ab:
        if name.endswith('_A'):
            assert params[name].shape[1] == rank
        else:
            assert params[name].shape[0] == rank
            assert torch.count_nonzero(params[name]) == 0  # B starts at zero


def test_lora_rank_selection():
    fixed = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=16, lora_min_rank=2)
    assert {p.shape[1] for n, p in fixed.named_parameters() if n.endswith('_A')} == {16}

    variable = LoraPointNet2(3, 8, lora_fix_rank=False, lora_max_rank=64, lora_min_rank=4)
    ranks = {n: p.shape[1] for n, p in variable.named_parameters() if n.endswith('_A')}
    assert all(4 <= r <= 64 for r in ranks.values())
    assert ranks['lora_sa1_A'] == 4      # 11 x 32 layer
    assert ranks['lora_fp1_A'] == 64     # 768 x 256 layer

    assert variable.nearest_power_of_2(0, 4, 64) == 4
    assert variable.nearest_power_of_2(5, 4, 64) == 8
    assert variable.nearest_power_of_2(1000, 4, 64) == 64


def test_lora_with_zero_B_matches_baseline():
    """With B = 0 the LoRA model must reproduce the baseline it was built from."""
    torch.manual_seed(0)
    baseline = PointNet2(num_classes=3, num_feat=8).eval()
    lora = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=8, alpha=16)
    transfer_baseline_weights(lora, baseline.state_dict(), skip_head=False)
    lora.eval()

    x = random_batch(8)
    with torch.no_grad():
        torch.manual_seed(1)  # farthest point sampling starts from a random point
        expected, _ = baseline(x)
        torch.manual_seed(1)
        got = lora(x)
    assert torch.allclose(got, expected, atol=1e-5)


def test_lora_update_changes_output():
    torch.manual_seed(0)
    lora = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=8, alpha=16).eval()
    x = random_batch(8)
    with torch.no_grad():
        torch.manual_seed(1)
        before = lora(x)
        for name, p in lora.named_parameters():
            if name.endswith('_B'):
                p.normal_(std=0.1)
        torch.manual_seed(1)
        after = lora(x)
    assert not torch.allclose(before, after, atol=1e-5)


@needs_checkpoints
def test_shipped_checkpoints_load():
    ft = torch.load(CKPT_FT, map_location='cpu')
    PointNet2(num_classes=3, num_feat=8).load_state_dict(ft['model'], strict=True)

    base = torch.load(CKPT_BASELINE, map_location='cpu')
    result = PointNet2(num_classes=3, num_feat=8).load_state_dict(base['model'], strict=False)
    assert sorted(result.missing_keys) == ['classifier.bias', 'classifier.weight']
    assert result.unexpected_keys == []

    lora = torch.load(CKPT_LORA, map_location='cpu')
    model = LoraPointNet2(num_classes=3, num_feat=8, lora_fix_rank=True, lora_max_rank=32, alpha=16)
    model.load_state_dict(lora['model'], strict=True)

    for ckpt in (ft, base, lora):
        assert {'model', 'optimizer', 'batch_size', 'lr', 'number_of_points', 'epoch'} <= set(ckpt)


@needs_checkpoints
def test_positional_transfer_from_shipped_baseline():
    """Every baseline tensor must land on the matching layer of the LoRA model."""
    base = torch.load(CKPT_BASELINE, map_location='cpu')['model']
    lora = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=32, alpha=16)
    keys, baseline_items = transfer_baseline_weights(lora, base, skip_head=True)

    assert len(keys) == len(baseline_items)
    state = lora.state_dict()
    for key, (base_key, tensor) in zip(keys, baseline_items):
        # same layer index and tensor kind, e.g. '0.weight' or '2.running_mean'
        assert key.split('.')[-2:] == base_key.split('.')[-2:] or key == base_key, (key, base_key)
        assert torch.equal(state[key], tensor)
