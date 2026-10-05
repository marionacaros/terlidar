# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code accompanying the paper *Efficient Task and Domain Adaptation in ALS Semantic Segmentation via LoRA for PointNet++*, plus the documentation for the TerLiDAR dataset (airborne LiDAR along the Ter River, Catalonia). The code is a published snapshot extracted from a larger private project (`3DSemanticSegmentation`); it is not a packaged library and has no build step or linter config. `tests/` holds CPU-only pytest smoke tests (`pytest tests`).

Note that the `test_*.py` files under `src/LoRA/` are **evaluation/inference scripts**, not unit tests; the unit tests are in `tests/`.

## Environment

```bash
conda create -n lora-pn2 python=3.10 -y
conda activate lora-pn2
pip install -r requirements.txt
```

`requirements.txt` is a full environment freeze (torch 2.3.1, laspy 2.5.4, numpy 1.26.4, ...), not a minimal dependency list. `requirements_conda.txt` is the conda equivalent.

## Running scripts

All scripts should be run **from the repository root**: they write to relative paths (`src/runs/lora/`, `src/LoRA/logs/`, `src/LoRA/checkpoints_lidarcat/`, `src/LoRA/metrics/`).

```bash
# Baseline PointNet++ (source domain: B29 / DALES)
python src/LoRA/train_cat3_pointnet2.py  --in_paths train_test_files/B29_80x80
python src/LoRA/train_dales_pointnet2.py --in_paths <dir of DALES .pt windows>

# Adaptation to the target domain (RIB = TerLiDAR)
python src/LoRA/train_lora_rib.py --in_paths train_test_files/RIB_smallLoRA_80x80 \
    --model_checkpoint <baseline *_NOclassifier.pt> --min_rank 32 --max_rank 32 --lora_alpha 16
python src/LoRA/train_ft_rib.py   --in_paths train_test_files/RIB_smallLoRA_80x80 \
    --model_checkpoint <baseline .pt>          # full fine-tuning comparison

# Evaluation (per-class IoU CSV + confusion matrix)
python src/LoRA/test_lora_segmentation.py --model_checkpoint <lora .pt> --rank 32 --max_rank 32 --lora_alpha 16 --dataset RIB
python src/LoRA/test_segmentation.py / test_ft_segmentation.py / test_*dales_segmentation.py

tensorboard --logdir src/runs/lora
```

### Things to know before running

- **No machine-specific paths are left in the code.** Data locations are always arguments: `--data_root` for the ICGC scripts, `--in_paths` / `--in_path` (required) for DALES, `--LAS_files_path` / `--out_path` for preprocessing.
- **`train_test_files/*/*.txt` hold paths relative to `--data_root`**: bare file names for RIB, `train/`, `val/`, `test/` sub-folders for B29 (`read_file_list` in `utils/utils.py` resolves them). `RIB_smallLoRA_80x80/val_files.txt` is empty on purpose: the RIB training scripts carve an 80/20 train/val split out of `train_files.txt` (seed 5 for LoRA, 4 for full fine-tuning).
- **Shipped weights live in `checkpoints/`** and are the argparse defaults of the RIB scripts. The DALES checkpoints are not shipped, so `--model_checkpoint` must be passed there.
- **Hyperparameters of the published runs differ from the script defaults**; the README reproduction commands pass them explicitly. See `TODO.md` for this and other known issues that were left alone because fixing them would change behaviour.
- `DATASET_NAME = 'Z31'` in `proc_no_ground.py` is still a constant in `main()` and must be edited by hand.

## Architecture

### Data pipeline

1. **Preprocessing** (repo root): `proc_no_ground.py` (TerLiDAR/ICGC LAS tiles) and `proc_split_LAS_DALES.py` (DALES) cut LAS tiles into square windows (80x80 m, default 8000 points) and store each window as a `.pt` tensor. Ground is removed when a window exceeds `n_points`; HAG and NDVI are added. Stored columns: `x, y, z, class, I, R, G, B, NIR, NDVI, HAG, point_id`.
2. **File naming carries meaning**: windows are prefixed by their dominant target class (`tower_`, `lines_`, `windturbine_`, `othertower_`, `crane_`, otherwise `pc_`). Training scripts oversample by filename prefix (`startswith('tower')` / `startswith('line')` lists appended to the train list multiple times).
3. **Datasets** ([src/datasets.py](src/datasets.py)): `CAT3Dataset` (train) and `CAT3SamplingDataset` (test, returns point ids so window predictions can be merged back per tile) for ICGC-style data; `DalesDataset` / `DalesSamplingDataset` for DALES. The other classes in the file (`CAT3DatasetViews`, `BarlowTwins*`) are leftovers from the parent project and unused here. Datasets normalize x,y to [-1,1] and z to [0,1] per window, and remap raw LAS classes to training labels in `get_labels_segmen` / `get_labels` / `get_all_labels`.

### Label spaces and feature counts

These must be kept consistent between dataset flags, `--num_classes`, `--num_features`, and the hardcoded class weights (`c_weights`) inside each training script:

- **ICGC (B29, RIB/TerLiDAR)**, default non-prod mode with `use_windturbine=False`: `0` surrounding/other, `1` transmission tower (LAS 15), `2` power lines (LAS 14). With `use_z=True` the input is 8 features: `x, y, HAG, z, I, G, B, NDVI`; otherwise 7.
- **DALES**: 5 features; `get_labels` gives 4 classes (other, ground, poles, power lines), `get_all_labels` gives 6 classes with `-1` as ignore (`CrossEntropyLoss(ignore_index=-1)`).
- `src/config.py` holds per-class sample counts and `COLOR_DROPOUT`; `N_CLASSES` there is not the source of truth, the script arguments are.

Training loops additionally subsample each 8000-point window to 4096 random points, apply a random z-rotation, and zero the last three feature channels with probability `COLOR_DROPOUT`.

### Models (`src/LoRA/models/`)

- [pointnet2_ss.py](src/LoRA/models/pointnet2_ss.py): baseline single-scale `PointNet2` built from the modular `PointNetSetAbstraction` / `PointNetFeaturePropagation` blocks in `pointnet2_utils.py`; final layer is named `classifier`.
- [lora_pointnet2_params.py](src/LoRA/models/lora_pointnet2_params.py): `LoraPointNet2` re-implements the same network **flattened** (`mlp_convs_1..4`, `mlp_convs_fp4..1`, `conv1`, `lora_classifier`) so each 1x1 conv can be wrapped by `lora_bmm4d` / `lora_bmm3d`, which add `(alpha / rank) * x @ (A @ B)` to the frozen conv output. LoRA params are registered as `lora_sa{1..12}_A/B`, `lora_fp{1..9}_A/B`, `lora_l1_A/B`.
  - Trainability is decided **by parameter name**: anything containing `lora` is trainable (A: Kaiming init, B: zeros), everything else is frozen. The classification head is named `lora_classifier` precisely so it stays trainable; renaming parameters changes what gets trained.
  - Rank is either fixed (`lora_fix_rank=True`, uses `lora_max_rank`) or per-layer, proportional to layer size and clamped to `[min_rank, max_rank]` (set both equal for a fixed rank, as the checkpoint name `32R32alph16` indicates).

### Checkpoint transfer between the two model classes

Because layer names differ between `PointNet2` and `LoraPointNet2`, the LoRA training scripts copy baseline weights **positionally**: `list(state_dict.keys())[44:-2]` skips the 44 LoRA A/B tensors (registered first) and the 2 classifier tensors, then assigns baseline tensors in order. Any change to the number or registration order of parameters in `LoraPointNet2` (or to the baseline layer order) silently breaks this loading. The baseline is expected to be a `*_NOclassifier.pt` checkpoint written by `save_checkpoint_without_classifier_layer`, so the target task can have a different number of classes.

Full fine-tuning (`train_ft_rib.py`) instead loads into `PointNet2` with `strict=False`.

Checkpoints are dicts with keys `model`, `optimizer`, `batch_size`, `lr`, `number_of_points`, `epoch`; the best-validation-loss model is saved, named from a timestamp plus the rank/alpha settings.

### Utilities

`utils/utils.py` (augmentation, sampling, kNN, LAS export, preprocessing helpers), `utils/get_metrics.py` (IoU, accuracy, class-weighting schemes), `utils/utils_plot.py`, `utils/frompt2las.py` (convert `.pt` windows back to LAS). `show_confusionmatrix_acc.ipynb` renders confusion matrices from evaluation output. `doc/point counts/` has per-class point counts for B29 and RIB.

## Dataset facts worth knowing

- "RIB" in code and filenames is the TerLiDAR dataset; "B29" is the source-domain ICGC block used to train the baseline.
- Proposed TerLiDAR benchmark split: blocks `pt438656`, `pt438652`, `pt438658` are the held-out test set; everything else is train.
- TerLiDAR class codes are the ICGC LAS codes listed in the README (2 ground, 14 power lines, 15 transmission tower, 18 other towers, ...).
- Licensing is dual: code is MIT, dataset and pretrained weights are CC BY 4.0.
