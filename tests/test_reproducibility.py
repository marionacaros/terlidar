"""Seeding, file lists, the RIB train/val split and checkpoint files."""
import hashlib
import math
import os
import random

import numpy as np
import torch

from conftest import REPO_ROOT
from src.LoRA.models.lora_pointnet2_params import LoraPointNet2
from src.LoRA.models.pointnet2_ss import PointNet2
from src.LoRA.models.utils import save_checkpoint, save_checkpoint_without_classifier_layer
from utils.utils import get_sampled_sequence, read_file_list, rotate_point_cloud_z, set_seed

RIB_LISTS = os.path.join(REPO_ROOT, 'train_test_files', 'RIB_smallLoRA_80x80')


def test_set_seed_seeds_python_numpy_and_torch():
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first
    set_seed(124)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) != first


def test_seeded_model_init_and_forward_are_repeatable():
    def run():
        set_seed(5)
        model = LoraPointNet2(3, 8, lora_fix_rank=True, lora_max_rank=4, alpha=16).eval()
        x = torch.rand(1, 8, 1024)
        with torch.no_grad():
            return model.state_dict(), model(x)  # farthest point sampling draws a random start point

    state_a, out_a = run()
    state_b, out_b = run()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    assert torch.equal(out_a, out_b)


def test_read_file_list_keeps_paths_without_data_root(tmp_path):
    list_file = tmp_path / 'train_files.txt'
    list_file.write_text('/server/data/train/pc_RIB_a_w1.pt\n/server/data/train/tower_RIB_a_w2.pt\n')
    assert read_file_list(str(list_file)) == ['/server/data/train/pc_RIB_a_w1.pt',
                                              '/server/data/train/tower_RIB_a_w2.pt']


def test_read_file_list_remaps_to_data_root(tmp_path):
    list_file = tmp_path / 'train_files.txt'
    list_file.write_text('/server/data/train/pc_RIB_a_w1.pt\n/server/data/train/tower_RIB_a_w2.pt\n')
    data_root = tmp_path / 'local'
    data_root.mkdir()

    # flat layout: <data_root>/<file name>
    assert read_file_list(str(list_file), str(data_root)) == [str(data_root / 'pc_RIB_a_w1.pt'),
                                                              str(data_root / 'tower_RIB_a_w2.pt')]
    # layout with the parent folder of the original path: <data_root>/train/<file name>
    (data_root / 'train').mkdir()
    assert read_file_list(str(list_file), str(data_root)) == [str(data_root / 'train' / 'pc_RIB_a_w1.pt'),
                                                              str(data_root / 'train' / 'tower_RIB_a_w2.pt')]


def _fingerprint(files):
    return hashlib.sha256('\n'.join(os.path.basename(f) for f in files).encode()).hexdigest()[:16]


def _rib_split(seed):
    """Same steps as train_lora_rib.py (seed 5) and train_ft_rib.py (seed 4)."""
    set_seed(seed)
    files = read_file_list(os.path.join(RIB_LISTS, 'train_files.txt'))
    files.sort()
    random.shuffle(files)
    val = files[int(0.80 * len(files)):]
    train = files[:int(0.80 * len(files))]
    towers = [f for f in train if f.split('/')[-1].startswith('tower')]
    lines = [f for f in train if f.split('/')[-1].startswith('line')]
    train = train + towers + lines + towers + lines
    random.shuffle(train)
    return train, val


def test_rib_file_lists_are_unchanged():
    train = read_file_list(os.path.join(RIB_LISTS, 'train_files.txt'))
    assert len(train) == len(set(train)) == 7157
    assert _fingerprint(sorted(train)) == 'e91ef0d497a7c666'
    assert len(read_file_list(os.path.join(RIB_LISTS, 'test_files.txt'))) == 25968
    # empty on purpose: validation files are held out of train_files.txt by the training scripts
    assert read_file_list(os.path.join(RIB_LISTS, 'val_files.txt')) == []


def test_rib_train_val_split_is_pinned():
    """The 80/20 split and the oversampled training order for the default seeds must not move."""
    train, val = _rib_split(seed=5)  # train_lora_rib.py
    assert (len(train), len(val)) == (6871, 1432)
    assert (_fingerprint(train), _fingerprint(val)) == ('3ebbb442f05fa075', '6279ba61e02fe8a5')

    train, val = _rib_split(seed=4)  # train_ft_rib.py
    assert (len(train), len(val)) == (6947, 1432)
    assert (_fingerprint(train), _fingerprint(val)) == ('ff0b7fe83b006e58', '8f3d114e37afdbc1')


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = PointNet2(num_classes=3, num_feat=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_checkpoint(str(tmp_path / 'full'), model, optimizer, 100, 0.001, 8000, 7)
    ckpt = torch.load(tmp_path / 'full.pt', map_location='cpu')
    assert set(ckpt) == {'model', 'optimizer', 'batch_size', 'lr', 'number_of_points', 'epoch'}
    assert (ckpt['batch_size'], ckpt['lr'], ckpt['number_of_points'], ckpt['epoch']) == (100, 0.001, 8000, 7)
    restored = PointNet2(num_classes=3, num_feat=8)
    restored.load_state_dict(ckpt['model'], strict=True)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), restored.state_dict().values()))


def test_checkpoint_without_classifier_fits_another_number_of_classes(tmp_path):
    torch.manual_seed(0)
    model = PointNet2(num_classes=3, num_feat=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_checkpoint_without_classifier_layer(str(tmp_path / 'base_NOclassifier'), model, optimizer,
                                             100, 0.001, 8000, 7)
    state = torch.load(tmp_path / 'base_NOclassifier.pt', map_location='cpu')['model']
    assert 'classifier.weight' not in state and 'classifier.bias' not in state
    assert 'classifier.weight' in model.state_dict()  # the model itself is left intact

    other = PointNet2(num_classes=6, num_feat=8)
    result = other.load_state_dict(state, strict=False)
    assert sorted(result.missing_keys) == ['classifier.bias', 'classifier.weight']
    assert result.unexpected_keys == []


def test_rotate_point_cloud_z():
    torch.manual_seed(0)
    xyz = torch.rand(2, 50, 3)

    quarter_turn = rotate_point_cloud_z(xyz.clone(), rotation_angle=math.pi / 2)
    assert torch.allclose(quarter_turn[..., 0], -xyz[..., 1], atol=1e-6)
    assert torch.allclose(quarter_turn[..., 1], xyz[..., 0], atol=1e-6)

    rotated = rotate_point_cloud_z(xyz.clone())
    assert torch.equal(rotated[..., 2], xyz[..., 2])                                  # z untouched
    assert torch.allclose(rotated.norm(dim=-1), xyz.norm(dim=-1), atol=1e-5)          # rigid


def test_get_sampled_sequence_keeps_every_point():
    torch.manual_seed(0)
    n_points = 8
    pc = torch.arange(20, dtype=torch.float32).unsqueeze(1).repeat(1, 3)   # 20 points, value == id
    ids = torch.arange(20)

    groups, out_ids, _ = get_sampled_sequence(pc, ids, n_points)

    assert groups.shape == (3, n_points, 3)                # 20 points padded to 24
    assert set(out_ids.tolist()) == set(range(20))         # nothing is dropped
    assert torch.equal(groups.view(-1, 3)[:, 0].long(), out_ids)   # ids stay aligned with their points
