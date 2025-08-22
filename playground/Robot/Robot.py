"""
Robot class for MuJoCo-based robot simulation and control.
"""

import threading
import time
from copy import copy
from typing import List, Any, Optional, Dict, Sequence

import mujoco
from mujoco_viewer import MujocoViewer

class DummyUserScn:
    def __getattr__(self, name):
        # 避免访问不存在属性时报错
        def dummy(*args, **kwargs):
            pass
        return dummy

class PassiveViewer(MujocoViewer):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.user_scn = DummyUserScn()  # 兼容老版本

    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def lock(self):
        # 兼容老版 API，直接返回一个空上下文管理器
        from contextlib import contextmanager
        @contextmanager
        def dummy():
            yield
        return dummy()
    # 兼容老版本的 opt 属性
    @property
    def opt(self):
        return self.vopt

    # 兼容老版本的 sync()
    def sync(self):
        self.render()
def launch_passive(model, data):
    return PassiveViewer(model, data)
# 给 mujoco 注入一个 viewer 属性，保持老代码可用
mujoco.viewer = type("viewer", (), {"launch_passive": launch_passive})



import numpy as np

from playground.Algorithm.BasicAlgorithm import *
from playground.Algorithm.InverseKinematics import InverseKinematics
from playground.MotionPlan.TrajectoryPlanner import TrajectoryPlanner
from playground.Mujoco.Interface import MjInterface
from playground.Utils.TrajectoryViewerUtils import (
    connect_trajectory,
    plot_trajectory,
)
from playground.Utils.TransformUtils import *
from playground.Utils.LoggerUtils import get_logger

logger = get_logger(__name__)


class MjVisFlag:
    """Python版mjVIS_可视化flag枚举，便于查阅和补全。"""

    CONVEXHULL = 0
    TEXTURE = 1
    JOINT = 2
    CAMERA = 3
    ACTUATOR = 4
    ACTIVATION = 5
    LIGHT = 6
    TENDON = 7
    RANGEFINDER = 8
    CONSTRAINT = 9
    INERTIA = 10
    SCLINERTIA = 11
    PERTFORCE = 12
    PERTOBJ = 13
    CONTACTPOINT = 14
    ISLAND = 15
    CONTACTFORCE = 16
    CONTACTSPLIT = 17
    TRANSPARENT = 18
    AUTOCONNECT = 19
    COM = 20
    SELECT = 21
    STATIC = 22
    SKIN = 23
    FLEXVERT = 24
    FLEXEDGE = 25
    FLEXFACE = 26
    FLEXSKIN = 27
    BODYBVH = 28
    FLEXBVH = 29
    MESHBVH = 30
    SDFITER = 31

    @classmethod
    def from_name(cls, name: str):
        """支持字符串转flag，忽略大小写和下划线。"""
        key = name.upper().replace(" ", "").replace("_", "")
        for attr in dir(cls):
            if not attr.startswith("_"):
                if attr.replace("_", "") == key:
                    return getattr(cls, attr)
        raise ValueError(f"Unknown MjVisFlag: {name}")


class MjLabel:
    """Python版mjLABEL_标签枚举，便于查阅和补全。"""

    NONE = 0
    BODY = 1
    JOINT = 2
    GEOM = 3
    SITE = 4
    CAMERA = 5
    LIGHT = 6
    TENDON = 7
    ACTUATOR = 8
    CONSTRAINT = 9
    FLEX = 10
    SKIN = 11
    SELECTION = 12
    SELPNT = 13
    CONTACTPOINT = 14
    CONTACTFORCE = 15
    ISLAND = 16

    @classmethod
    def from_name(cls, name: str):
        """支持字符串转label，忽略大小写和下划线。"""
        key = name.upper().replace(" ", "").replace("_", "")
        for attr in dir(cls):
            if not attr.startswith("_"):
                if attr.replace("_", "") == key:
                    return getattr(cls, attr)
        raise ValueError(f"Unknown MjLabel: {name}")


class MjFrame:
    """Python版mjFRAME_帧枚举，便于查阅和补全。"""

    NONE = 0
    BODY = 1
    GEOM = 2
    SITE = 3
    CAMERA = 4
    LIGHT = 5
    CONTACT = 6
    WORLD = 7

    @classmethod
    def from_name(cls, name: str):
        """支持字符串转frame，忽略大小写和下划线。"""
        key = name.upper().replace(" ", "").replace("_", "")
        for attr in dir(cls):
            if not attr.startswith("_"):
                if attr.replace("_", "") == key:
                    return getattr(cls, attr)
        raise ValueError(f"Unknown MjFrame: {name}")


class Robot:
    """
    Robot class for MuJoCo-based robot simulation and control.
    Provides high-level motion, planning, and control interfaces.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        ee_site_name: str = "robot1/attachment_site",
        is_viewer: bool = True,
        is_render: bool = False,
        input_viewer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.mj_interface = MjInterface(model, data)
        self._lock = threading.Lock()
        self.is_viewer = is_viewer
        self.is_render = is_render
        if self.is_render:
            self.renderer = mujoco.Renderer(self.model, width=4500, height=3000)
        if self.is_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif input_viewer is not None:
            self.viewer = input_viewer

        self.set_frame("NONE")
        self.set_label("NONE")
        self.set_vis_flag("CONTACTFORCE", False)

        self.joint_range = self.model.jnt_range
        self.simulate_timestep = self.model.opt.timestep
        self.qpos_ids = self.mj_interface._qpos_ids
        self.qvel_ids = self.mj_interface._qvel_ids
        self.site_jnt_info: Dict[str, Dict[str, Any]] = {}
        self.__complete_joint_site_info()
        self.robot_dof: int = len(self.site_jnt_info[ee_site_name]["joint_ids"])

        self.ee_site_name = ee_site_name
        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name
        )

        self.ctrl_type = self.get_actuator_type()  # "motor" or "position"
        self.traj_planner = TrajectoryPlanner(dofs=self.robot_dof)
        self.ik_solver = InverseKinematics(self.model, self.data, self.site_jnt_info)

        self.__target_qpos = copy(self.data.qpos)[self.qpos_ids]
        self.__target_qvel = copy(self.data.qvel)[self.qvel_ids]
        self.__robot_num = len(self.site_jnt_info)

        # logger.info(f"site_jnt_info: {self.site_jnt_info}")
        # logger.info(self.__end_effector_info)

    def servoj_blocking(
        self,
        target_qpos: np.ndarray,
        tol: float = 1e-6,
        max_steps: int = 1500,
        move_target_robot_site: Optional[str] = None,
    ) -> None:
        """
        使用伺服控制移动到目标关节位置，阻塞直到收敛或达到最大步数。

        Args:
            target_qpos (np.ndarray): 目标关节位置。
            tol (float): 收敛容差，默认为1e-6。
            max_steps (int): 最大迭代步数，默认为1500。
            move_target_robot_site (Optional[str]): 目标机器人站点的名称，如果为None，则使用所有关节。
        """
        if move_target_robot_site is not None:
            qpos_slice = self.site_jnt_info[move_target_robot_site]["joint_ids"]
        else:
            qpos_slice = self.qpos_ids
        current_qpos = copy(self.data.qpos)[qpos_slice]
        self.servoj(target_qpos, move_target_robot_site=move_target_robot_site)
        for i in range(max_steps):
            start = time.time()
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            if self.ctrl_type == "motor":
                self.servoj(target_qpos, move_target_robot_site=move_target_robot_site)
            current_qpos = self.data.qpos[qpos_slice]
            error_diff = max(abs(current_qpos - target_qpos))
            if error_diff < tol:
                logger.debug(f"Converged in {i} steps.")
                break
            duration = time.time() - start
            if duration < 0.002:
                time.sleep(0.002 - duration)
        self.__target_qpos[qpos_slice] = target_qpos

    def servoj_multipoint(self, jpos_traj, move_target_robot_site=None):
        """
        使用伺服控制移动到多个目标关节位置点。

        Args:
            jpos_traj: 关节位置轨迹，一个包含多个关节位置的列表或NumPy数组。
            move_target_robot_site: 目标机器人站点的名称，如果为None，则使用所有关节。
        """
        if move_target_robot_site is not None:
            qpos_slice = self.site_jnt_info[move_target_robot_site]["joint_ids"]
        else:
            qpos_slice = self.qpos_ids
        for idx, jpos in enumerate(jpos_traj):
            if idx < len(jpos_traj) - 1:
                start_loop = time.time()

                self.servoj(jpos, move_target_robot_site=move_target_robot_site)
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                # error = jpos - self.mj_interface.qpos[qpos_slice]
                duration = time.time() - start_loop
                if duration < 0.002:
                    time.sleep(0.002 - duration)
            else:
                self.servoj_blocking(
                    jpos, move_target_robot_site=move_target_robot_site
                )

    def movej_multipoints(
        self, jpos_traj: List | np.array, method="ruckig", move_target_robot_site=None
    ):
        """
        通过多点轨迹移动机器人关节。

        Args:
            jpos_traj (List | np.array): 关节位置轨迹，可以是列表或NumPy数组。
            method (str): 轨迹规划方法，默认为"ruckig"。
            move_target_robot_site: 目标机器人站点的名称，如果为None，则使用所有关节。
        """
        jpos_traj = np.array(jpos_traj)
        if method == "ruckig":

            jpos_traj_ruckig, _ = self.traj_planner.traj_plan_j_multipoint_ruckig(
                jpos_traj
            )
            self.servoj_multipoint(
                jpos_traj_ruckig, move_target_robot_site=move_target_robot_site
            )

    def move_line_blocking(self, target_tcp, step=20, move_target_robot_site=None):
        """
        使用逆运动学将机器人末端执行器沿直线移动到目标TCP位置，阻塞直到完成。

        Args:
            target_tcp: 目标TCP（工具中心点）的位置和姿态。
            step (int): 轨迹规划的步数，默认为20。
            move_target_robot_site: 目标机器人站点的名称，如果为None，则使用所有关节。
        """
        if move_target_robot_site is not None:
            qpos_slice = self.site_jnt_info[move_target_robot_site]["joint_ids"]
        else:
            qpos_slice = self.qpos_ids
        sitepos = self.data.site_xpos[self.ee_site_id]  # 获取目标site的位置
        sitexmat = self.data.site_xmat[self.ee_site_id]
        sitexquat = np.zeros(4)
        mujoco.mju_mat2Quat(sitexquat, sitexmat)
        sitex = np.concatenate([sitepos, sitexquat])
        xpos_traj = linear_interpolation(
            sitex,
            target_tcp,
            step,
        )

        """---------------------plot-----------------------------------"""
        xpos_plot = np.array(xpos_traj)[:, :3]
        plot_trajectory(self.viewer, xpos_plot)
        connect_trajectory(self.viewer, xpos_plot)
        """---------------------逆解-----------------------"""
        current_qpos = copy(self.data.qpos)[qpos_slice]
        jpos_traj = []
        start_debug = time.time()
        for idx, xpos in enumerate(xpos_traj):
            jpos = self.ik_solver.IK(
                self.ee_site_name,
                current_qpos,
                xpos[:3],
                xpos[3:],
            )

            current_qpos = jpos

            jpos_traj.append(jpos)
        logger.debug(f"IK time cost {time.time() - start_debug}")

        self.movej_multipoints(
            jpos_traj, "ruckig", move_target_robot_site=move_target_robot_site
        )

    def move_keypos(
        self, keypos="home", move_target_robot_site="robot1/attachment_site"
    ):
        """
        移动机器人到预设的关键姿态。

        Args:
            keypos (str): 关键姿态的名称，默认为"home"。
            move_target_robot_site (str): 目标机器人站点的名称，默认为"robot1/attachment_site"。
        """
        if move_target_robot_site is not None:
            qpos_slice = self.site_jnt_info[move_target_robot_site]["joint_ids"]
        else:
            qpos_slice = self.qpos_ids
        key = self.mj_interface.mj_name2id(mujoco.mjtObj.mjOBJ_KEY, keypos)
        self.__target_qpos = copy(self.model.key_qpos[key])
        mujoco.mj_kinematics(self.model, self.data)
        self.movej_multipoints(
            [self.data.qpos[qpos_slice], self.__target_qpos[qpos_slice]],
            move_target_robot_site=move_target_robot_site,
        )

    def servoj(self, target_q, target_dq=None, move_target_robot_site=None):
        """
        使用伺服控制将机器人关节移动到目标位置和速度。

        Args:
            target_q: 目标关节位置。
            target_dq: 目标关节速度，默认为None。
            move_target_robot_site: 目标机器人站点的名称，如果为None，则使用所有关节。
        """
        if move_target_robot_site is not None:
            qpos_slice = self.site_jnt_info[move_target_robot_site]["joint_ids"]
        else:
            qpos_slice = self.qpos_ids

        if target_dq is None:
            target_dq = np.zeros(len(qpos_slice))
        if self.ctrl_type == "position":
            # only for position control
            self.data.ctrl[qpos_slice] = target_q

        elif self.ctrl_type == "motor":
            # only for torque control
            target_q = np.array(target_q)
            target_dq = np.array(target_dq)
            qacc_temp = self.data.qacc.copy()
            dq_temp = self.data.qvel.copy()
            # 只补偿重力
            self.data.qacc = np.zeros(len(qacc_temp))
            self.data.qvel = np.zeros(len(dq_temp))
            mujoco.mj_inverse(self.model, self.data)

            self.data.qacc = qacc_temp
            self.data.qvel = dq_temp
            # 带重力前馈的PD控制
            self.data.ctrl[qpos_slice] = self.pid_with_ff.update(
                target_q,
                self.data.qpos[qpos_slice],
                target_dq,
                self.data.qvel[qpos_slice],
                ff=self.data.qfrc_bias[qpos_slice],
            )

    def capture_camera_image(self, cam_name):
        """
        捕获指定摄像头的图像。

        Args:
            cam_name (str): 摄像头的名称。

        Returns:
            np.ndarray: 渲染的RGB图像数组，形状为(height, width, 3)。
        """

        cam_id = self.mj_interface.mj_name2id(mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        self.renderer.update_scene(self.data, camera=cam_id)

        # 渲染RGB图像（返回形状为(height, width, 3)的数组）
        rgb = self.renderer.render()

        return rgb

    def get_ft_value(self):
        """
        获取末端执行器的力和扭矩值。

        Returns:
            np.ndarray: 包含力和扭矩值的NumPy数组。
        """

        force = self.data.sensor("end_effector_force").data.copy()
        torque = self.data.sensor("end_effector_torque").data.copy()
        return np.concatenate([force, torque])

    def get_actuator_type(self):
        """
        获取执行器类型（"motor"或"position"）。

        Returns:
            str: 执行器类型。
        """
        if self.model.actuator_gainprm[0][0] == 1:
            return "motor"
        else:
            return "position"

    def complete_end_effector_info(self, end_effector_info):
        """
        补全末端执行器信息。

        Args:
            end_effector_info (dict): 包含末端执行器信息的字典。
        """
        for name in end_effector_info:
            value = end_effector_info.get(name)
            if value is None:
                end_effector_info[name] = self.__end_effector_info[name]
            else:
                for key in self.__end_effector_info[name]:
                    if value.get(key) is None:
                        value[key] = self.__end_effector_info[name][key]

    def __complete_joint_site_info(self):
        """
        补全关节站点信息，包括关节名称、ID、qpos ID、qvel ID和控制ID。
        """
        for i in range(self.model.nsite):
            site_name = self.model.site(i).name
            if "attachment_site" not in site_name:
                continue
            joint_ids = self.__get_associated_jnt(site_name)
            joint_names = [self.model.joint(id).name for id in joint_ids]
            qpos_ids = self.qpos_ids
            qvel_ids = self.qvel_ids
            try:
                ctrl_ids = [
                    list(self.mj_interface.get_actuated_joint_inds()).index(id)
                    for id in joint_ids
                ]
            except:
                ctrl_ids = []
            self.site_jnt_info[site_name] = {
                "joint_names": joint_names,
                "joint_ids": joint_ids,
                "qpos_ids": qpos_ids,
                "qvel_ids": qvel_ids,
                "ctrl_ids": ctrl_ids,
            }

    def __get_associated_jnt(self, site_name):
        """
        获取与给定站点相关联的关节ID。

        Args:
            site_name (str): 站点的名称。

        Returns:
            list: 相关联的关节ID列表。
        """
        site_id = self.model.site(site_name).id
        body_id = self.model.site_bodyid[site_id]
        joint_ids = []
        while self.model.body(body_id).name != "world":
            body = self.model.body(body_id)
            for id in body.jntadr:
                if id == -1:
                    break
                joint_ids.append(id)
            body_id = body.parentid[0]

        return joint_ids[::-1]

    @property
    def __end_effector_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取末端执行器信息，包括位置和四元数。

        Returns:
            Dict[str, Dict[str, Any]]: 包含末端执行器信息的字典。
        """
        info: Dict[str, Dict[str, Any]] = {}
        for i in range(self.model.nsite):
            site_name = self.model.site(i).name
            if "attachment_site" not in site_name:
                continue
            xpos = copy(self.data.site(site_name).xpos)
            xmat = copy(self.data.site(site_name).xmat)
            xquat = np.zeros(4)
            mujoco.mju_mat2Quat(xquat, xmat)
            info[site_name] = {
                "xpos": xpos,
                "xquat": xquat,
            }

        return info

    def set_vis_flag(self, flag_name: str, enabled: bool):
        """
        控制MuJoCo viewer的可视化选项开关。

        Args:
            flag_name (str): MjVisFlag的名字（如 'CONTACTPOINT', 'JOINT', ...）。
            enabled (bool): True为显示，False为隐藏。
        """
        flag_enum = MjVisFlag.from_name(flag_name)
        with self.viewer.lock():
            self.viewer.opt.flags[flag_enum] = True

    def set_label(self, label_name: str) -> None:
        """
        设置MuJoCo viewer的标签类型。

        Args:
            label_name (str): MjLabel的名字。
        """
        with self.viewer.lock():
            self.viewer.opt.label = MjLabel.from_name(label_name)

    def set_frame(self, frame_name: str) -> None:
        """
        设置MuJoCo viewer的坐标系类型。

        Args:
            frame_name (str): MjFrame的名字。
        """
        with self.viewer.lock():
            self.viewer.opt.frame = MjFrame.from_name(frame_name)
