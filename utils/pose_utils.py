# Adapted from https://github.com/clvrai/furniture-bench/blob/main/furniture_bench/utils/pose.py#L50
from typing import List, Union

from mujoco_inspire.utils.averageQuaternions import averageQuaternions
from mujoco_inspire.utils import transform as T

import numpy as np
import numpy.typing as npt


def mat_to_roll_pitch_yaw(rmat):
    """Convert rotation matrix to roll-pitch-yaw angles.
    Args:
        rmat: 3x3 rotation matrix.
    Returns:
        roll, pitch, yaw: roll-pitch-yaw angles.
    """
    roll = np.arctan2(rmat[2, 1], rmat[2, 2])
    pitch = np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
    yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
    return roll, pitch, yaw


def rot_mat(angles, hom: bool = False):
    """Given @angles (x, y, z), compute rotation matrix
    Args:
        angles: (x, y, z) rotation angles.
        hom: whether to return a homogeneous matrix.
    """
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    if hom:
        M = np.zeros((4, 4), dtype=np.float32)
        M[:3, :3] = R
        M[3, 3] = 1.0
        return M
    return R


def get_mat(pos: List[float], angles: Union[List[float], npt.NDArray[np.float32]]):
    """Get homogeneous matrix given position and rotation angles.
    Args:
        pos: relative positions (x, y, z).
        angles: relative rotations (x, y, z) or 3x3 matrix.
    """
    transform = np.zeros((4, 4), dtype=np.float32)
    if not isinstance(angles, np.ndarray) or not len(angles.shape) == 2:
        transform[:3, :3] = np.eye(3) if not np.any(angles) else rot_mat(angles)
    else:
        if len(angles[0, :]) == 4:
            transform[:4, :4] = angles
        else:
            transform[:3, :3] = angles
    transform[3, 3] = 1.0
    transform[:3, 3] = pos
    return transform


def comp_avg_pose(poses):
    np.set_printoptions(suppress=True)
    quats = []
    positions = []
    for pose in poses:
        if pose is None:
            continue

        quats.append(T.convert_quat(T.mat2quat(pose[:3, :3]), "wxyz"))
        positions.append(pose[:3, 3])
    if quats == []:
        return None

    quats = np.stack(quats, axis=0)
    positions = np.stack(positions, axis=0)

    avg_quat = averageQuaternions(quats).astype(np.float32)

    avg_rot = T.quat2mat(T.convert_quat(avg_quat, "xyzw"))
    avg_pos = positions.mean(axis=0)
    return T.to_homogeneous(avg_pos, avg_rot)
