"""
Transformation utilities for robotics and MuJoCo.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any, Sequence


def quaternion_to_axis_angle(q: Sequence[float]) -> tuple[np.ndarray, float]:
    """
    Convert quaternion to axis-angle representation.
    Args:
        q: Quaternion [w, x, y, z]
    Returns:
        axis: Rotation axis (3,)
        theta: Rotation angle (rad)
    """
    w, x, y, z = q

    w = np.clip(w, -1, 1)
    theta = 2 * np.arccos(w)  # 旋转角度
    sin_theta_over_2 = np.sqrt(1 - w**2)
    if sin_theta_over_2 < 1e-6:
        # 角度非常小，轴任意选择
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([x, y, z]) / sin_theta_over_2
    return axis, theta


def axis_angle_to_quaternion(axis, theta):
    """将轴角表示转换为四元数"""
    half_theta = theta / 2.0
    w = np.cos(half_theta)
    xyz = np.sin(half_theta) * axis

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def xyz_rxryrz2tquat(
    xyz_rxryrz: np.ndarray | list[float],
    rotation_order: str = "xyz",
    is_degree: bool = False,
) -> np.ndarray:
    transformation = np.identity(3)
    transformation[:3, :3] = R.from_euler(
        seq=rotation_order, angles=xyz_rxryrz[3:], degrees=is_degree
    ).as_matrix()
    quat = R.from_matrix(transformation).as_quat()

    return quat


def xyz_rxryrz2xyz_mjquat(
    xyz_rxryrz: np.ndarray | list[float],
    rotation_order: str = "xyz",
    is_degree: bool = False,
) -> np.ndarray:
    transformation = np.identity(3)
    transformation[:3, :3] = R.from_euler(
        seq=rotation_order, angles=xyz_rxryrz[3:], degrees=is_degree
    ).as_matrix()
    quat = R.from_matrix(transformation).as_quat()
    mjquat = np.roll(quat, 1)  # [x,y,z,w] → [w,x,y,z]

    return np.concatenate([xyz_rxryrz[:3], mjquat])


def mjxmat2quat(mjxmat: np.ndarray | list[float]) -> np.ndarray:
    """
    将 MuJoCo 的旋转矩阵转换为四元数 [w, x, y, z] 格式
    :param mjxmat: MuJoCo 的旋转矩阵 [3, 3]
    :return: 转换后的四元数 [w, x, y, z]
    """
    quat_scipy = R.from_matrix(mjxmat.reshape(3, 3)).as_quat()  # [x, y, z, w]
    quat_mj = np.roll(quat_scipy, 1)  # [x, y, z, w] → [w, x, y, z]
    return quat_mj


def mjxpos_mjxmat2xyz_rxryrz(
    mjxpos: np.ndarray | list[float],
    mjxmat: np.ndarray | list[float],
    rotation_order: str = "xyz",
    is_degree: bool = False,
) -> np.ndarray:
    """
    将 MuJoCo 的位置和旋转矩阵转换为 [x, y, z, rx, ry, rz] 格式
    :param mjxpos: MuJoCo 的位置 [x, y, z]
    :param mjxmat: MuJoCo 的旋转矩阵 [3, 3]
    :param rotation_order: 旋转顺序
    :param is_degree: 是否以度为单位
    :return: 转换后的 [x, y, z, rx, ry, rz]
    """
    rxryrz = R.from_matrix(mjxmat).as_euler(seq=rotation_order, degrees=is_degree)
    return np.concatenate([mjxpos, rxryrz])


def mjxpos_mjxquat2xyz_rxryrz(
    mjxpos: np.ndarray | list[float],
    mjxquat: np.ndarray | list[float],
    rotation_order: str = "xyz",
    is_degree: bool = False,
) -> np.ndarray:
    """
    将 MuJoCo 的位置和四元数转换为 [x, y, z, rx, ry, rz] 格式
    :param mjxpos: MuJoCo 的位置 [x, y, z]
    :param mjxquat: MuJoCo 的四元数 [w, x, y, z]
    :param rotation_order: 旋转顺序
    :param is_degree: 是否以度为单位
    :return: 转换后的 [x, y, z, rx, ry, rz]
    """
    # 将四元数从 [w, x, y, z] 转换为 [x, y, z, w] 格式
    quat_scipy = np.roll(mjxquat, -1)

    # 使用 Scipy 提取欧拉角
    rxryrz = R.from_quat(quat_scipy).as_euler(seq=rotation_order, degrees=is_degree)

    return np.concatenate([mjxpos, rxryrz])


def mjxpos_mjxquat2trans(
    mjxpos: np.ndarray | list[float],
    mjxquat: np.ndarray | list[float],
):
    """
    将 MuJoCo 的位置和四元数转换为 4x4 齐次变换矩阵
    :param mjxpos: MuJoCo 的位置 [x, y, z]
    :param mjxquat: MuJoCo 的四元数 [w, x, y, z]
    :return: 4x4 齐次变换矩阵
    """
    transformation = np.eye(4)
    transformation[:3, 3] = mjxpos
    transformation[:3, :3] = R.from_quat(np.roll(mjxquat, -1)).as_matrix()

    return transformation


def trans2xyz_rxryrz(transformation: np.ndarray, is_degree: bool = True):
    rxryrz = R.from_matrix(transformation[:3, :3]).as_euler(
        seq="xyz", degrees=is_degree
    )
    return np.concatenate([transformation[:3, 3], rxryrz])


def xyz_rxryrz2trans(
    xyz_rxryrz: np.ndarray | list[float],
    rotation_order: str = "xyz",
    is_degree: bool = False,
) -> np.ndarray:
    """
    将 [x, y, z, rx, ry, rz] 转换为 4x4 齐次变换矩阵
    :param xyz_rxryrz: [x, y, z, rx, ry, rz]
    :param rotation_order: 旋转顺序
    :param is_degree: 是否以度为单位
    :return: 4x4 齐次变换矩阵
    """
    translation = np.eye(4)
    translation[:3, 3] = xyz_rxryrz[:3]

    rotation = np.eye(4)
    rotation[:3, :3] = R.from_euler(
        seq=rotation_order, angles=xyz_rxryrz[3:], degrees=is_degree
    ).as_matrix()

    return translation @ rotation
