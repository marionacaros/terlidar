"""LoRA injection checked on random tensors: the update formula and which parameters learn."""
import pytest
import torch
from torch import nn

from src.LoRA.models.lora_pointnet2_params import LoraPointNet2

RANK = 4
ALPHA = 16
N_POINTS = 1024  # smallest cloud the first set-abstraction level accepts


@pytest.fixture(scope='module')
def lora_model():
    torch.manual_seed(0)
    return LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=RANK, alpha=ALPHA)


def random_batch(batch=2, num_feat=8, n_points=N_POINTS):
    x = torch.rand(batch, num_feat, n_points)
    x[:, :2] = x[:, :2] * 2 - 1
    return x


def test_lora_bmm3d_adds_scaled_low_rank_update(lora_model):
    torch.manual_seed(0)
    layer = nn.Conv1d(5, 7, 1)
    x = torch.randn(2, 5, 11)                       # [B, Cin, N]
    A, B = torch.randn(5, RANK), torch.randn(RANK, 7)

    out = lora_model.lora_bmm3d(x, layer, A, B, alpha=ALPHA)

    # the update is a 1x1 convolution with weight (alpha / rank) * (A @ B)^T
    delta = nn.functional.conv1d(x, (ALPHA / RANK) * (A @ B).t().unsqueeze(-1))
    assert out.shape == (2, 7, 11)
    assert torch.allclose(out, layer(x) + delta, atol=1e-5)


def test_lora_bmm4d_adds_scaled_low_rank_update(lora_model):
    torch.manual_seed(0)
    layer = nn.Conv2d(5, 7, 1)
    x = torch.randn(2, 5, 3, 11)                    # [B, Cin, nsample, npoint]
    A, B = torch.randn(5, RANK), torch.randn(RANK, 7)

    out = lora_model.lora_bmm4d(x, layer, A, B, alpha=ALPHA)

    delta = nn.functional.conv2d(x, (ALPHA / RANK) * (A @ B).t()[:, :, None, None])
    assert out.shape == (2, 7, 3, 11)
    assert torch.allclose(out, layer(x) + delta, atol=1e-5)


def test_lora_with_zero_B_leaves_layer_output_unchanged(lora_model):
    torch.manual_seed(0)
    layer3d, layer4d = nn.Conv1d(5, 7, 1), nn.Conv2d(5, 7, 1)
    x3d, x4d = torch.randn(2, 5, 11), torch.randn(2, 5, 3, 11)
    A, B = torch.randn(5, RANK), torch.zeros(RANK, 7)

    assert torch.equal(lora_model.lora_bmm3d(x3d, layer3d, A, B, alpha=ALPHA), layer3d(x3d))
    assert torch.equal(lora_model.lora_bmm4d(x4d, layer4d, A, B, alpha=ALPHA), layer4d(x4d))


def test_lora_scaling_is_alpha_over_rank(lora_model):
    A, B = torch.randn(5, RANK), torch.randn(RANK, 7)
    assert lora_model.lora_alpha == ALPHA
    assert torch.allclose(lora_model._compute_lora_weight(A, B), (ALPHA / RANK) * (A @ B))


def test_gradients_reach_only_lora_parameters_and_head():
    torch.manual_seed(0)
    model = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=RANK, alpha=ALPHA).train()
    log_probs = model(random_batch())                               # [B, N, classes]
    targets = torch.randint(0, 3, (2 * N_POINTS,))
    nn.functional.nll_loss(log_probs.reshape(-1, 3), targets).backward()

    for name, p in model.named_parameters():
        if 'lora' not in name:
            assert p.grad is None, name
        else:
            assert p.grad is not None, name
            if name.endswith('_B') or name.startswith('lora_classifier'):
                assert torch.count_nonzero(p.grad) > 0, name
            elif name.endswith('_A'):
                # B starts at zero, so the first gradient of every A is exactly zero
                assert torch.count_nonzero(p.grad) == 0, name


def test_optimizer_step_changes_only_lora_parameters_and_head():
    torch.manual_seed(0)
    model = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=RANK, alpha=ALPHA).train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    # same optimizer construction as train_lora_rib.py
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,
                                 eps=1e-08, weight_decay=1e-4)

    for _ in range(2):  # A matrices only move from the second step on (B starts at zero)
        optimizer.zero_grad()
        log_probs = model(random_batch())
        targets = torch.randint(0, 3, (2 * N_POINTS,))
        nn.functional.nll_loss(log_probs.reshape(-1, 3), targets).backward()
        optimizer.step()

    for name, p in model.named_parameters():
        if 'lora' in name:
            assert not torch.equal(p, before[name]), name
        else:
            assert torch.equal(p, before[name]), name
