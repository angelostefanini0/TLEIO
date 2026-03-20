from pathlib import Path
from typing import List, Union
import numpy as np

PathLike = Union[str, Path]


def ensure_exists(path: PathLike) -> Path:
    """
    Simple utility to ensure the path exists
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def load_gt(path: PathLike) -> np.ndarray: 
    """
    Returns:
        gt: np.ndarray of shape (N,8), dtype=float64 
        columns are : timestamp[s] px py pz qx qy qz qw
    """

    path = ensure_exists(path)
    gt = np.loadtxt(path, dtype=np.float64)

    return gt

def load_timestamps_from_gt(gt: np.ndarray) -> np.ndarray:
    """
    Returns:
        timestamps_us: np.ndarray of shape (N,), dtype=float64
    """
    timestamps = gt[:, 0].astype(np.int64)

    return timestamps

def load_pos_from_gt(gt: np.ndarray) -> np.ndarray:
    """
    Returns:
        timestamps_us: np.ndarray of shape (N,), dtype=float64
    """
    pos = gt[:, 1:4]

    return pos

def load_quat_from_gt(gt: np.ndarray) -> np.ndarray:
    """
    Returns:
        timestamps_us: np.ndarray of shape (N,), dtype=float64
    """
    pos = gt[:, 4:]

    return pos