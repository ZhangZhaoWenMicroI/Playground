import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from typing import Tuple, Optional


class MjForwardPipeline:
    """基于原生 MuJoCo C API 的前向动力学管道封装"""

    # ========== 位置相关计算阶段 (Steps 1-10) ==========

    def step_1_kinematics(self, model, data):
        """步骤1: 运动学计算 - 计算身体位置、方向和变换矩阵"""
        mujoco.mj_kinematics(model, data)

    def step_2_com_pos(self, model, data):
        """步骤2: 质心位置计算 - 计算cdof和子树质心"""
        mujoco.mj_comPos(model, data)

    def step_3_camlight(self, model, data):
        """步骤3: 相机和光源计算"""
        mujoco.mj_camlight(model, data)

    def step_4_flex(self, model, data):
        """步骤4: 柔性体计算"""
        mujoco.mj_flex(model, data)

    def step_5_tendon(self, model, data):
        """步骤5: 腱索长度和雅可比矩阵计算"""
        mujoco.mj_tendon(model, data)

    def step_6_crb(self, model, data):
        """步骤6: 复合刚体惯性计算"""
        mujoco.mj_crb(model, data)

    def step_7_factor_m(self, model, data):
        """步骤7: 质量矩阵分解"""
        mujoco.mj_factorM(model, data)

    def step_8_collision(self, model, data):
        """步骤8: 碰撞检测"""
        mujoco.mj_collision(model, data)

    def step_9_make_constraint(self, model, data):
        """步骤9: 约束生成"""
        mujoco.mj_makeConstraint(model, data)

    def step_10_transmission(self, model, data):
        """步骤10: 传动系统计算"""[10]
        mujoco.mj_transmission(model, data)

    # ========== 速度相关计算阶段 (Steps 11-16) ==========

    def step_11_flexedge_velocity(self, model, data):
        """步骤11: 柔性边缘速度计算"""
        # 这部分在 mj_fwdVelocity 中实现
        if mujoco.mj_isSparse(model):
            # 稀疏矩阵乘法 - 在C代码中实现
            pass
        else:
            # 稠密矩阵乘法 - 在C代码中实现
            pass

    def step_12_tendon_velocity(self, model, data):
        """步骤12: 腱索速度计算"""
        # 腱索速度计算在 mj_fwdVelocity 中实现
        pass

    def step_13_actuator_velocity(self, model, data):
        """步骤13: 执行器速度计算"""
        # 执行器速度计算在 mj_fwdVelocity 中实现
        pass

    def step_14_com_vel(self, model, data):
        """步骤14: COM基础速度计算 - 计算cvel和cdof_dot"""
        mujoco.mj_comVel(model, data)

    def step_15_passive(self, model, data):
        """步骤15: 被动力计算 - 弹簧、阻尼、重力补偿等"""
        mujoco.mj_passive(model, data)

    def step_16_reference_constraint(self, model, data):
        """步骤16: 约束参考计算"""
        # 这个函数在 mj_fwdVelocity 中调用 [16](#8-15)
        mujoco.mj_referenceConstraint(model, data)

    def step_17_rne(self, model, data):
        """步骤17: 递归牛顿-欧拉算法 - 计算偏置力"""
        # flg_acc=0 表示不包含加速度项，只计算偏置力
        mujoco.mj_rne(model, data, 0, data.qfrc_bias)

    # ========== 执行器和加速度计算阶段 (Steps 18-22) ==========

    def step_18_fwd_actuation(self, model, data):
        """步骤18: 执行器力计算"""
        mujoco.mj_fwdActuation(model, data)

    def step_19_fwd_acceleration(self, model, data):
        """步骤19: 加速度计算 - 合并所有非约束力"""
        mujoco.mj_fwdAcceleration(model, data)

    def step_20_fwd_constraint(self, model, data):
        """步骤20: 约束求解"""
        mujoco.mj_fwdConstraint(model, data)

    def step_21_sensor_pos(self, model, data):
        """步骤21: 位置相关传感器计算"""
        mujoco.mj_sensorPos(model, data)

    def step_22_sensor_vel(self, model, data):
        """步骤22: 速度相关传感器计算"""
        mujoco.mj_sensorVel(model, data)

    def step_23_sensor_acc(self, model, data):
        """步骤23: 加速度相关传感器计算"""
        mujoco.mj_sensorAcc(model, data)

    # ========== 时间积分阶段 (Steps 24-25) ==========

    def step_24_euler_integration(self, model, data):
        """步骤24: 欧拉积分器"""
        mujoco.mj_Euler(model, data)

    def step_25_rk4_integration(self, model, data):
        """步骤25: 四阶龙格-库塔积分器"""
        mujoco.mj_RungeKutta(model, data, 4)

    # ========== 高级管道执行函数 ==========

    def execute_fwd_position(self, model, data):
        """执行位置相关计算阶段"""
        mujoco.mj_fwdPosition(model, data)

    def execute_fwd_velocity(self, model, data):
        """执行速度相关计算阶段"""
        mujoco.mj_fwdVelocity(model, data)

    def execute_forward_complete(self, model, data):
        """执行完整的前向动力学计算"""
        mujoco.mj_forward(model, data)


class MjxForwardPipeline:
    """MuJoCo 前向动力学管道的完整封装"""

    def __init__(self, model):

        self.mjx_model = mjx.put_model(model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        # JIT 编译所有函数以提高性能
        self._compile_functions()

    def _compile_functions(self):
        """编译所有管道函数"""
        self.step_1_kinematics = jax.jit(mjx.kinematics)
        self.step_2_com_pos = jax.jit(mjx.com_pos)
        self.step_3_camlight = jax.jit(mjx.camlight)
        self.step_4_tendon = jax.jit(mjx.tendon)
        self.step_5_crb = jax.jit(mjx.crb)
        self.step_6_factor_m = jax.jit(mjx.factor_m)
        self.step_7_collision = jax.jit(mjx.collision)
        self.step_8_make_constraint = jax.jit(mjx.make_constraint)
        self.step_9_transmission = jax.jit(mjx.transmission)
        self.step_10_sensor_pos = jax.jit(mjx.sensor_pos)
        self.step_11_actuator_velocity = jax.jit(self._compute_actuator_velocity)
        self.step_12_tendon_velocity = jax.jit(self._compute_tendon_velocity)
        self.step_13_com_vel = jax.jit(mjx.com_vel)
        self.step_14_passive = jax.jit(mjx.passive)
        self.step_15_rne = jax.jit(mjx.rne)
        self.step_16_sensor_vel = jax.jit(mjx.sensor_vel)
        self.step_17_fwd_actuation = jax.jit(mjx.fwd_actuation)
        self.step_18_fwd_acceleration = jax.jit(mjx.fwd_acceleration)
        self.step_19_sensor_acc = jax.jit(mjx.sensor_acc)
        self.step_20_solve_constraints = jax.jit(mjx.solve)
        self.step_21_euler_integration = jax.jit(mjx.euler)
        self.step_22_rk4_integration = jax.jit(mjx.rungekutta4)
        self.step_23_implicit_integration = jax.jit(mjx.implicit)
        self.step_24_advance_time = jax.jit(self._advance_time)
        self.step_25_update_sensors = jax.jit(self._update_final_sensors)

    # ========== 位置相关计算阶段 (Steps 1-10) ==========

    def step_1_kinematics(self, m, d):
        """步骤1: 运动学计算 - 计算身体位置、方向和变换矩阵"""
        return mjx.kinematics(m, d)

    def step_2_com_pos(self, m, d):
        """步骤2: 质心位置计算 - 计算cdof和子树质心"""
        return mjx.com_pos(m, d)

    def step_3_camlight(self, m, d):
        """步骤3: 相机和光源计算"""
        return mjx.camlight(m, d)

    def step_4_tendon(self, m, d):
        """步骤4: 腱索长度和雅可比矩阵计算"""
        return mjx.tendon(m, d)

    def step_5_crb(self, m, d):
        """步骤5: 复合刚体惯性计算"""
        return mjx.crb(m, d)

    def step_6_factor_m(self, m, d):
        """步骤6: 质量矩阵分解"""
        return mjx.factor_m(m, d)

    def step_7_collision(self, m, d):
        """步骤7: 碰撞检测"""
        return mjx.collision(m, d)

    def step_8_make_constraint(self, m, d):
        """步骤8: 约束生成"""
        return mjx.make_constraint(m, d)

    def step_9_transmission(self, m, d):
        """步骤9: 传动系统计算"""
        return mjx.transmission(m, d)

    def step_10_sensor_pos(self, m, d):
        """步骤10: 位置相关传感器计算"""
        return mjx.sensor_pos(m, d)

    # ========== 速度相关计算阶段 (Steps 11-16) ==========

    def _compute_actuator_velocity(self, m, d):
        """计算执行器速度"""
        return d.replace(actuator_velocity=d.actuator_moment @ d.qvel)

    def _compute_tendon_velocity(self, m, d):
        """计算腱索速度"""
        return d.replace(ten_velocity=d.ten_J @ d.qvel)

    def step_11_actuator_velocity(self, m, d):
        """步骤11: 执行器速度计算"""
        return self._compute_actuator_velocity(m, d)

    def step_12_tendon_velocity(self, m, d):
        """步骤12: 腱索速度计算"""
        return self._compute_tendon_velocity(m, d)

    def step_13_com_vel(self, m, d):
        """步骤13: COM基础速度计算 - 计算cvel和cdof_dot"""
        return mjx.com_vel(m, d)

    def step_14_passive(self, m, d):
        """步骤14: 被动力计算 - 弹簧、阻尼、重力补偿等"""
        return mjx.passive(m, d)

    def step_15_rne(self, m, d):
        """步骤15: 递归牛顿-欧拉算法 - 计算偏置力"""
        return mjx.rne(m, d)

    def step_16_sensor_vel(self, m, d):
        """步骤16: 速度相关传感器计算"""
        return mjx.sensor_vel(m, d)

    # ========== 执行器和加速度计算阶段 (Steps 17-20) ==========

    def step_17_fwd_actuation(self, m, d):
        """步骤17: 执行器力计算"""
        return mjx.fwd_actuation(m, d)

    def step_18_fwd_acceleration(self, m, d):
        """步骤18: 加速度计算 - 合并所有非约束力"""
        return mjx.fwd_acceleration(m, d)

    def step_19_sensor_acc(self, m, d):
        """步骤19: 加速度相关传感器计算"""
        return mjx.sensor_acc(m, d)

    def step_20_solve_constraints(self, m, d):
        """步骤20: 约束求解"""
        if d.efc_J.size == 0:
            return d.replace(qacc=d.qacc_smooth)
        return mjx.solve(m, d)

    # ========== 时间积分阶段 (Steps 21-25) ==========

    def step_21_euler_integration(self, m, d):
        """步骤21: 欧拉积分器"""
        return mjx.euler(m, d)

    def step_22_rk4_integration(self, m, d):
        """步骤22: 四阶龙格-库塔积分器"""
        return mjx.rungekutta4(m, d)

    def step_23_implicit_integration(self, m, d):
        """步骤23: 隐式积分器"""
        return mjx.implicit(m, d)

    def _advance_time(self, m, d):
        """推进时间步"""
        return d.replace(time=d.time + m.opt.timestep)

    def step_24_advance_time(self, m, d):
        """步骤24: 时间推进"""
        return self._advance_time(m, d)

    def _update_final_sensors(self, m, d):
        """更新最终传感器数据"""
        # 这里可以添加任何最终的传感器更新逻辑
        return d

    def step_25_update_sensors(self, m, d):
        """步骤25: 最终传感器更新"""
        return self._update_final_sensors(m, d)
