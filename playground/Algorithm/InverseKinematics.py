from copy import copy

import mujoco
import numpy as np
from mujoco import minimize

from playground.Algorithm.BasicAlgorithm import *



class InverseKinematics:
    def __init__(self, model, data, site_jnt_info):
        self.model = model
        self.data = data
        self.site_jnt_info = site_jnt_info

    def IK(
        self,
        effector_name,
        q0,
        pos,
        quat,
        res_target=None,
        max_inter=1000,
        radius=0.04,
        reg=0.001,  # 离res_target项
    ):
        """
        effector_name：末端执行器名称（如机械臂末端）。
        x0：初始关节角度猜测。
        pos：目标位置。
        quat：目标姿态（四元数）。
        max_inter：最大迭代次数。
        默认系数是mujoco源文档给出的默认系数
        """

        current_pos = copy(self.data.qpos)
        joint_ids = self.site_jnt_info[effector_name]["joint_ids"]

        jnt_range = self.model.jnt_range
        jnt_range_ = jnt_range[joint_ids, :]

        q0_ = q0[joint_ids]
        bounds = [
            jnt_range_[joint_ids, 0],
            jnt_range_[joint_ids, 1],
        ]
        ik_res = lambda current_qpos: self.__ik_residual(
            effector_name,
            current_qpos,
            pos,
            quat,
            reg_target=res_target,
            radius=radius,
            reg=reg,
        )
        ik_jac = lambda current_qpos, r: self.__ik_jac(
            effector_name, current_qpos, r, pos=pos, quat=quat, radius=radius, reg=reg
        )
        qpos_, _ = minimize.least_squares(
            q0_,
            ik_res,
            bounds,
            ik_jac,
            verbose=0,
            max_iter=max_inter,
        )

        self.data.qpos = current_pos
        mujoco.mj_kinematics(self.model, self.data)  # 回到起始位置

        return qpos_

    def __ik_residual(
        self,
        effector_name,
        current_qpos,
        pos,
        quat=[1.0, 0.0, 0.0, 0.0],
        radius=0.05,
        reg=0.01,
        reg_target=None,
    ):
        """Residual for inverse kinematics.

        Args:
            current_qpos: joint angles.
            pos: target position for the end effector.
            quat: target orientation for the end effector.
            radius: scaling of the 3D cross.

        Returns:
            The residual of the Inverse Kinematics task.
        """

        res = []
        joint_ids = self.site_jnt_info[effector_name]["joint_ids"]
        for i in range(current_qpos.shape[1]):

            current_qpos_i = current_qpos[:, i]
            self.data.qpos[joint_ids] = current_qpos_i

            mujoco.mj_kinematics(self.model, self.data)
            res_pos = self.data.site(effector_name).xpos - pos
            # 旋转矩阵转四元数
            effector_quat = np.empty(4)
            mujoco.mju_mat2Quat(effector_quat, self.data.site(effector_name).xmat)

            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, quat, effector_quat)
            res_quat *= radius

            reg_target = (
                self.model.key("home").qpos if reg_target is None else reg_target
            )
            reg_target_ = reg_target[joint_ids]

            res_reg = reg * (current_qpos[:, i] - reg_target_)
            res_i = np.hstack((res_pos, res_quat, res_reg))
            res.append(np.atleast_2d(res_i).T)

        return np.hstack(res)

    def __ik_jac(self, effector_name, x, res, pos, quat, radius=0.05, reg=0.001):
        """Analytic Jacobian of inverse kinematics residual

        Args:
            x: joint angles.
            pos: target position for the end effector.
            quat: target orientation for the end effector.
            radius: scaling of the 3D cross.

        Returns:
            The Jacobian of the Inverse Kinematics task.
        """
        # least_squares() passes the value of the residual at x which is sometimes
        # useful, but we don't need it here.
        del res
        joint_ids = self.site_jnt_info[effector_name]["joint_ids"]

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        jac_pos = np.empty((3, self.model.nv))
        jac_quat = np.empty((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jac_pos, jac_quat, self.data.site(effector_name).id
        )

        jac_pos_ = jac_pos[:, joint_ids]
        jac_quat_ = jac_quat[:, joint_ids]

        effector_quat = np.empty(4)
        mujoco.mju_mat2Quat(effector_quat, self.data.site(effector_name).xmat)

        Deffector = np.empty((3, 3))
        mujoco.mjd_subQuat(quat, effector_quat, None, Deffector)

        target_mat = np.empty(3 * 3)
        mujoco.mju_quat2Mat(target_mat, np.array(quat))

        target_mat = target_mat.reshape(3, 3)

        mat = radius * Deffector.T @ target_mat.T
        # 将优化方向转移到目标点
        jac_quat_ = mat @ jac_quat_

        # Regularization Jacobian.
        jac_reg = reg * np.eye(jac_quat_.shape[1])

        return np.vstack((jac_pos_, jac_quat_, jac_reg))
