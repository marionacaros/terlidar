import glob
import numpy as np
import torch
from progressbar import progressbar
import os
import pickle
import laspy
from sklearn.neighbors import KDTree, NearestNeighbors
import random
from scipy.spatial import cKDTree
import math
from typing import List, Optional, Tuple


def read_file_list(list_path: str, data_root: Optional[str] = None) -> List[str]:
    """
    Read a text file with one point cloud path per line.

    The lists in train_test_files/ store paths relative to the folder with the preprocessed
    windows (e.g. pc_RIB_pt436658_w709.pt or train/pc_B29_ETehpt_315545_w284.pt). Pass that
    folder as data_root: each entry becomes <data_root>/<file name>, or
    <data_root>/<parent folder>/<file name> when that sub-folder (e.g. train, val, test) exists
    inside data_root. Without data_root the entries are returned unchanged.

    :param list_path: str, path to the .txt list
    :param data_root: str or None, directory containing the preprocessed .pt files
    :return: list of str
    """
    with open(list_path, 'r') as f:
        files = f.read().splitlines()

    if data_root:
        remapped = []
        for path in files:
            parent = os.path.basename(os.path.dirname(path))
            sub_dir = os.path.join(data_root, parent) if parent else data_root
            base_dir = sub_dir if os.path.isdir(sub_dir) else data_root
            remapped.append(os.path.join(base_dir, os.path.basename(path)))
        files = remapped

    return files


def set_seed(seed: int) -> None:
    """
    Seed the Python, NumPy and PyTorch random number generators.

    cuDNN is left in its default (non-deterministic) mode, so runs on GPU with the same seed
    can still differ slightly.

    :param seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Preprocessing ----------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

def preprocessing(pc: np.ndarray, max_h: float = 200.0, n_points: int = 8000,
                  max_points: int = 400000) -> np.ndarray:
    """
    Perform preprocessing on a given point cloud.

    Steps:
    1. Remove outliers (height above ground > max_h) and points with negative height above
       ground; divide the height above ground by max_h.
    2. Calculate and clip the Normalized Difference Vegetation Index (NDVI) within the range [-1, 1].
    3. If there are between 100 and n_points non-ground points, keep only as many ground
       points (class 2) as needed to reach n_points.
    4. If there are already enough non-ground points, remove the ground points.

    Parameters:
    - pc (numpy.ndarray): Input point cloud data [points, 12] with columns
      x, y, z, class, I, R, G, B, NIR, NDVI, HAG, point_id.
    - max_h (float): maximum height above ground kept, also the HAG normalisation constant.
    - n_points (int): number of points a window should have.
    - max_points (int): only used to print a warning for very large windows.

    Returns:
    - pc numpy.ndarray: Processed point cloud after the specified preprocessing steps.
    """
    # Remove outliers (points above max_z)
    pc = pc[pc[:, 10] <= max_h]
    pc[:, 10] = pc[:, 10]/max_h
    
    # Remove points z < 0
    pc = pc[pc[:, 10] >= 0]

    # add NDVI
    pc[:, 9] = get_ndvi(pc[:, 8], pc[:, 5])  # range [-1, 1]
    pc[:, 9] = np.clip(pc[:, 9], -1, 1.0)

    # Check if num. points different from ground < n_points
    len_pc = pc[pc[:, 3] != 2].shape[0]
    if 100 < len_pc < n_points:
        # if there are few points we will keep ground points
        len_needed_p = n_points - len_pc

        # Get indices of ground points
        labels = pc[:, 3]
        i_terrain = np.where(labels == 2.0)[0]

        # if we have enough points of ground to cover missed points
        if len_needed_p < len(i_terrain):
            needed_i = random.sample(list(i_terrain), k=len_needed_p)
        else:
            needed_i = i_terrain

        # store points needed
        points_needed_terrain = pc[needed_i, :]

        # remove terrain points
        pc = pc[pc[:, 3] != 2, :]

        # store only needed terrain points
        pc = np.concatenate((pc, points_needed_terrain), axis=0)

    # if enough points, remove ground
    elif len_pc >= n_points:
        pc = pc[pc[:, 3] != 2, :]

    # max number of points is MAX_N_PTS 
    if len_pc > max_points:
        print(f'PC with more than 400K pts. shape= {pc.shape}')
        # Reduce number of points
        # sampled_indices = np.random.choice(pc.shape[0], max_points, replace=False)
        # pc = pc[sampled_indices,:]

    return pc


def weights_init(m: torch.nn.Module) -> None:
    """Xavier-normal weights and zero bias for Conv2d and Linear layers; use with ``model.apply``."""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def inplace_relu(m: torch.nn.Module) -> None:
    """Make ReLU layers operate in place; use with ``model.apply``."""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def rotate_point_cloud_z(batch_data: torch.Tensor, rotation_angle: Optional[float] = None) -> torch.Tensor:
    """
    Randomly rotate the point clouds around the Z-axis to augment the dataset.
    Rotation is per shape based along up (Z) direction.
    Use input angle if given.
    One angle is drawn with torch.rand for the whole batch.
    
    Input:
      batch_data: BxNx3 tensor, original batch of point clouds
    Return:
      BxNx3 tensor, rotated batch of point clouds
    """
    batch_data = batch_data.to(torch.float32)

    if rotation_angle is None:
        rotation_angle = torch.rand(1).item() * 2 * math.pi

    cosval = math.cos(rotation_angle)
    sinval = math.sin(rotation_angle)
    rotation_matrix = torch.tensor([
        [cosval, sinval, 0],
        [-sinval, cosval, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=batch_data.device)

     # Right-multiply: (B, N, 3) @ (3, 3) -> (B, N, 3)
    rotated_data = batch_data @ rotation_matrix

    return rotated_data


        # except Exception as e:
        #     print(f'Error {e} in file {fileName}')


def get_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (nir - red) / (nir + red), with 0 where nir + red == 0."""
    a = (nir - red)
    b = (nir + red)
    c = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return c


######################################### augmentations #########################################


###################################################### samplings #######################################################


def get_sampled_sequence(pc: torch.Tensor, ids: torch.Tensor,
                         n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    reshape tensor into sequence of n_points

    The points are shuffled (torch.randperm) and cut into consecutive groups of n_points, so
    that every point goes through the model. The last group is completed with points from the
    start of the shuffled cloud; a cloud with fewer than n_points points is completed with
    randomly repeated points.

    :param pc: Input tensor float32 of shape: [points, dims]
    :param ids: point ids int
    :param n_points: Number of points in each sequence
    :return: Sampled sequences pc tensor [n_sequences, n_points, dims], the ids of the points in
        the same order (flattened), and the permutation that was applied
    """

    # Shuffle
    indices = torch.randperm(pc.shape[0])

    dims = pc.shape[1]
    pc = pc[indices]
    ids = ids[indices]

    # Get needed points for sampling 
    if pc.shape[0] > n_points and  pc.shape[0] % n_points !=0:
        remain = n_points - (pc.shape[0] % n_points)
        pc3 = torch.cat((pc, pc[:remain, :]), dim=0)
        ids = torch.cat((ids, ids[:remain]), dim=0)
    elif pc.shape[0] < n_points:
        points_needed = n_points - pc.shape[0]
        # duplicate points 
        rdm_list = np.random.randint(0, pc.shape[0], points_needed)
        extra_points = pc[rdm_list, :]
        pc3 = torch.cat([pc, extra_points], dim=0)
        ids = torch.cat((ids, ids[rdm_list]), dim=0)
    else:
        pc3=pc

    try:
        # Add dimension with sequence
        pc3 = torch.unsqueeze(pc3, dim=0)
        pc3 = pc3.view(-1, n_points, dims)
        
    except Exception as e:
        print(f"Error during reshaping: {e}")
        print(f"Shape pc: {pc.shape}")
        print(f"Shape pc3: {pc3.shape}")

    return pc3, ids, indices
