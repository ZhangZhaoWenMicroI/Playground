from typing import List, Tuple
import numpy as np
from ruckig import Ruckig, InputParameter, OutputParameter, Result, Synchronization
from playground.Utils.LoggerUtils import *

logger = get_logger(__name__)


class TrajectoryPlanner:
    """轨迹规划器类，用于生成机器人关节轨迹。"""

    def __init__(self, dofs: int = 6, delta_time: float = 0.002):
        """
        初始化轨迹规划器。

        Args:
            dofs (int): 机器人自由度。
            delta_time (float): 规划时间步长。
        """
        self.dofs = dofs
        self.delta_time = delta_time
        self.ruckig = Ruckig(dofs=self.dofs, delta_time=self.delta_time)
        self.max_velocity = np.full(dofs, 3.14)
        self.max_acceleration = np.full(dofs, 10.0)
        self.max_jerk = np.full(dofs, 100.0)

    def _validate_constraints(self, constraints):
        """确保约束为numpy数组格式"""
        return (
            np.asarray(constraints)
            if not isinstance(constraints, np.ndarray)
            else constraints
        )

    def traj_plan_j_multipoint_ruckig(
        self,
        joint_targets: np.ndarray | List,
        speed_ratio=0.4,
        acc_ratio=1.0,
        jerk_ratio=1.0,
    ):
        """
        使用Ruckig算法进行多点关节轨迹规划。

        Args:
            joint_targets (np.ndarray | List): 目标关节位置点数组或列表。
            speed_ratio (float): 速度比例因子。
            acc_ratio (float): 加速度比例因子。
            jerk_ratio (float): 加加度比例因子。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 包含位置轨迹和速度轨迹的元组。
        """
        # 初始化存储数组
        position_traj = []  # 存储所有位置点 (形状 N x dofs)
        velocity_traj = []  # 存储所有速度点 (形状 N x dofs)
        joint_targets = np.array(joint_targets)
        # 初始化Ruckig输入和输出参数
        inp = InputParameter(self.dofs)
        out = OutputParameter(self.dofs)
        inp.synchronization = Synchronization.Phase

        # 设置约束
        inp.max_velocity = self.max_velocity * speed_ratio
        inp.max_acceleration = self.max_acceleration * acc_ratio
        inp.max_jerk = self.max_jerk * jerk_ratio

        # 初始化状态 (从第一个目标点开始)
        inp.current_position = joint_targets[0].copy()
        inp.current_velocity = np.zeros(self.dofs)  # 初始速度设为0
        inp.current_acceleration = np.zeros(self.dofs)  # 初始加速度设为0
        position_traj.append(joint_targets[0])

        # 遍历所有中间点
        for i in range(len(joint_targets) - 1):
            # 设置目标状态
            inp.target_position = joint_targets[i + 1]
            # 目标速度设为0
            inp.target_velocity = np.zeros(self.dofs)
            # 目标加速度设为0
            inp.target_acceleration = np.zeros(self.dofs)

            # 生成轨迹段
            while True:
                result = self.ruckig.update(inp, out)
                if result == Result.Working or result == Result.Finished:
                    # 记录当前点
                    position_traj.append(out.new_position.copy())  # 必须复制！
                    velocity_traj.append(out.new_velocity.copy())

                    # 更新输入状态
                    inp.current_position = out.new_position
                    inp.current_velocity = out.new_velocity
                    inp.current_acceleration = out.new_acceleration

                    if result == Result.Finished:
                        break
                else:
                    raise RuntimeError(f"Ruckig error: {result}")

        # 转换为numpy数组并返回
        return np.array(position_traj), np.array(velocity_traj)
