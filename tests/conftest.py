import os
import sys

import torch  # noqa: F401  (imported before numpy on purpose: avoids an MKL/OpenMP clash in some conda envs)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
