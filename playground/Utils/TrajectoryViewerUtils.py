"""
Trajectory visualization utilities for MuJoCo viewer.
"""
import mujoco
import mujoco.viewer as viewer
import numpy as np
from typing import Any, Sequence


def plot_trajectory(
    viewer: Any,
    xpos_plot: Sequence[Any],
    plot_size: float = 0.008,
    plot_color: np.ndarray = np.array([1, 0, 0, 1])
) -> None:
    """
    Plot a trajectory as a sequence of spheres in the MuJoCo viewer.
    Args:
        viewer: MuJoCo viewer object
        xpos_plot: Sequence of 3D positions
        plot_size: Sphere size
        plot_color: RGBA color
    """
    viewer.user_scn.ngeom = 0
    i = 0
    for p in xpos_plot:
        x, y, z = p
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[plot_size, 0, 0],
            pos=np.array([x, y, z]),
            mat=np.eye(3).flatten(),
            rgba=plot_color,
        )
        i += 1
    viewer.user_scn.ngeom = i


def connect_trajectory(viewer: Any, xpos_plot: Sequence[Any], width: float = 0.002) -> None:
    """
    Connect trajectory points with capsules in the MuJoCo viewer.
    Args:
        viewer: MuJoCo viewer object
        xpos_plot: Sequence of 3D positions
        width: Capsule radius
    """
    i = 0
    viewer.user_scn.ngeom = 0
    for j in range(len(xpos_plot) - 1):
        p1 = xpos_plot[j]  # 当前点
        p2 = xpos_plot[j + 1]  # 下一个点

        # 初始化连接器几何体
        mujoco.mjv_connector(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=width,  # 连接器半径
            from_=np.array(p1),  # 起点
            to=np.array(p2),  # 终点
        )
        i += 1
    viewer.user_scn.ngeom = i
