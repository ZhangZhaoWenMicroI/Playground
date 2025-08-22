import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from playground.Utils.TransformUtils import *


def linear_interpolation(start_pose: np.ndarray, target_pose: np.ndarray, num_steps: int) -> np.ndarray:
    """
    在笛卡尔空间中进行直线插值，包括位置的线性插值和姿态的球面线性插值 (SLERP)。

    Args:
        start_pose (np.ndarray): 起始位姿，格式为 [x, y, z, qw, qx, qy, qz]。
        target_pose (np.ndarray): 目标位姿，格式为 [x, y, z, qw, qx, qy, qz]。
        num_steps (int): 插值的步数。

    Returns:
        np.ndarray: 生成的插值轨迹，每行为一个位姿 [x, y, z, qw, qx, qy, qz]。
    """

    # 位置线性插值
    pos_start = np.array(start_pose[:3])
    pos_end = np.array(target_pose[:3])
    pos_traj = np.linspace(pos_start, pos_end, num_steps)

    # 姿态球面插值(SLERP)
    rot_start = np.roll(start_pose[3:], -1)  # [w,x,y,z] → [x,y,z,w]
    rot_end = np.roll(target_pose[3:], -1)  # [w,x,y,z] → [x,y,z,w]

    rot_start /= np.linalg.norm(rot_start)  # 强制归一化
    rot_end /= np.linalg.norm(rot_end)

    key_rots = R.from_quat([rot_start, rot_end])
    slerp = Slerp([0, 1], key_rots)
    times = np.linspace(0, 1, num_steps)
    rot_traj = slerp(times).as_quat()
    rot_traj = np.roll(rot_traj, 1, axis=1)  # [x,y,z,w] → [w,x,y,z]

    return np.hstack([pos_traj, rot_traj])  # 列拼接

