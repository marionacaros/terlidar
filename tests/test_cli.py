"""Every script must start from the repository root: imports resolve and arguments parse."""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

SCRIPTS = [
    'src/LoRA/train_cat3_pointnet2.py',
    'src/LoRA/train_dales_pointnet2.py',
    'src/LoRA/train_ft_rib.py',
    'src/LoRA/train_lora_dales_pointnet2.py',
    'src/LoRA/train_lora_rib.py',
    'src/LoRA/test_segmentation.py',
    'src/LoRA/test_ft_segmentation.py',
    'src/LoRA/test_lora_segmentation.py',
    'src/LoRA/test_dales_segmentation.py',
    'src/LoRA/test_ft_dales_segmentation.py',
    'src/LoRA/test_lora_dales_segmentation.py',
    'proc_no_ground.py',
]


@pytest.mark.parametrize('script', SCRIPTS)
def test_script_help(script):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, script, '--help'], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'usage:' in result.stdout


def test_seed_defaults_match_the_original_scripts():
    """The seeds that were hardcoded at import time are now the --seed defaults."""
    expected = {'train_lora_rib.py': 5, 'train_ft_rib.py': 4, 'train_cat3_pointnet2.py': 4,
                'train_dales_pointnet2.py': 4, 'train_lora_dales_pointnet2.py': 4}
    for name, seed in expected.items():
        with open(os.path.join(REPO_ROOT, 'src', 'LoRA', name)) as f:
            source = f.read()
        assert f"parser.add_argument('--seed', type=int, default={seed}," in source, name
