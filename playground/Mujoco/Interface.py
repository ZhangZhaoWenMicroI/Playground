from copy import copy
from typing import List

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from playground.Algorithm.BasicAlgorithm import *


class MjInterface(object):

    def __init__(self, model, data) -> None:

        self.model = model
        self.data = data

    def mj_name2id(self, mj_type, name):
        """
        mjOBJ_BODY,                     // body
        mjOBJ_XBODY,                    // body, used to access regular frame instead of i-frame
        mjOBJ_JOINT,                    // joint
        mjOBJ_DOF,                      // dof
        mjOBJ_GEOM,                     // geom
        mjOBJ_SITE,                     // site
        mjOBJ_CAMERA,                   // camera
        mjOBJ_LIGHT,                    // light
        mjOBJ_FLEX,                     // flex
        mjOBJ_MESH,                     // mesh
        mjOBJ_SKIN,                     // skin
        mjOBJ_HFIELD,                   // heightfield
        mjOBJ_TEXTURE,                  // texture
        mjOBJ_MATERIAL,                 // material for rendering
        mjOBJ_PAIR,                     // geom pair to include
        mjOBJ_EXCLUDE,                  // body pair to exclude
        mjOBJ_EQUALITY,                 // equality constraint
        mjOBJ_TENDON,                   // tendon
        mjOBJ_ACTUATOR,                 // actuator
        mjOBJ_SENSOR,                   // sensor
        mjOBJ_NUMERIC,                  // numeric
        mjOBJ_TEXT,                     // text
        mjOBJ_TUPLE,                    // tuple
        mjOBJ_KEY,                      // keyframe
        mjOBJ_PLUGIN,                   // plugin instance
        """
        """获取物体ID"""
        return mujoco.mj_name2id(self.model, mj_type, name)

    @property
    def qpos(self):
        return copy(self.data.qpos[self._qpos_ids])

    def get_geom_xpos(self, geom_name_or_id):
        """获取 geom 的全局位置 (xpos)"""
        if isinstance(geom_name_or_id, str):
            # 通过名称访问
            geom_id = self.data.geom(geom_name_or_id).id
        else:
            # 直接通过ID访问
            geom_id = geom_name_or_id
        return self.data.geom_xpos[geom_id].copy()

    def get_geom_xquat(self, geom_name_or_id):
        """获取 geom 的全局姿态四元数 (xquat)"""
        if isinstance(geom_name_or_id, str):
            geom_id = self.data.geom(geom_name_or_id).id
        else:
            geom_id = geom_name_or_id
        return self.data.geom_xquat[geom_id].copy()

    @property
    def _qpos_ids(self):
        """获取可控关节位置索引"""
        return [self.model.jnt_qposadr[i] for i in self._get_actuated_joints()]

    @property
    def _qvel_ids(self):
        """获取可控关节速度索引"""
        return [self.model.jnt_dofadr[i] for i in self._get_actuated_joints()]

    def _get_actuated_joints(self):
        """解析受控关节列表"""
        return [self.model.actuator(i).trnid[0] for i in range(self.model.nu)]

    def _get_home_config(self):
        """从XML读取初始位形"""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        return self.model.key_qpos[key_id][self._qpos_ids]

    def get_ee_pose(self, site_name):
        """获取末端执行器当前位姿"""
        site = self.data.site(site_name)
        pos = site.xpos.copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site.xmat)
        return pos, quat

    def _compute_mass_matrix(self):
        """计算质量矩阵逆"""
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M[np.ix_(self._qvel_ids, self._qvel_ids)]

    def get_frc_tau(self):
        return self.data.qfrc_bias[
            self._qvel_ids
        ].copy()  # 偏致力，包括离心力，；克里奥利力，重力

    def nq(self):
        return self.model.nq

    def nu(self):
        return self.model.nu

    def nv(self):
        return self.model.nv

    def sim_dt(self):
        return self.model.opt.timestep

    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)

    def get_qpos(self):
        return self.data.qpos.copy()

    def get_qvel(self):
        return self.data.qvel.copy()

    def get_qacc(self):
        return self.data.qacc.copy()

    def get_cvel(self):
        return self.data.cvel.copy()

    def get_jnt_name(self):
        """
        Returns the list of joint names.
        """
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        return joint_names

    def get_jnt_id_by_name(self, name):
        return self.model.joint(name)

    def get_jnt_qposadr_by_name(self, name):
        return self.model.joint(name).qposadr

    def get_jnt_qveladr_by_name(self, name):
        return self.model.joint(name).dofadr

    def get_jnt_type(self):
        return self.model.jnt_type

    def get_body_ext_force(self):
        return self.data.cfrc_ext.copy()

    def get_body_transform(self, body_name):
        """Get the position and rotation matrix of a specified body."""
        try:
            body = self.data.body(body_name)
        except KeyError:
            raise ValueError(f"Body name '{body_name}' does not exist")
        pos = body.xpos
        mat = np.reshape(body.xmat, (3, 3))

        return pos, mat

    def get_gear_ratios(self):
        """
        Returns transmission ratios.
        """
        return self.model.actuator_gear[:, 0]

    def get_motor_names(self):
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return actuator_names

    def get_actuated_joint_inds(self):
        """
        Returns list of joint indices to which actuators are attached.
        """
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return [
            idx for idx, jnt in enumerate(joint_names) if jnt + "_pos" in actuator_names
        ]

    def get_actuated_joint_names(self):
        """
        Returns list of joint names to which actuators are attached.
        """
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return [
            jnt for idx, jnt in enumerate(joint_names) if jnt + "_pos" in actuator_names
        ]

    def get_motor_qposadr(self):
        """
        Returns the list of qpos indices of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        return [self.model.jnt_qposadr[i] for i in indices]

    def get_motor_positions(self):
        """
        Returns position of actuators.
        """
        return self.data.actuator_length

    def get_motor_velocities(self):
        """
        Returns velocities of actuators.
        """
        return self.data.actuator_velocity

    def get_act_joint_torques(self):
        """
        Returns actuator force in joint space.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_torques = self.data.actuator_force
        return motor_torques * gear_ratios

    def get_act_joint_positions(self):
        """
        Returns position of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_positions = self.get_motor_positions()
        return motor_positions / gear_ratios

    def get_act_joint_velocities(self):
        """
        Returns velocities of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_velocities = self.get_motor_velocities()
        return motor_velocities / gear_ratios

    def get_act_joint_position(self, act_name):
        """
        Returns position of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).length) == 1
        return (
            self.data.actuator(act_name).length[0]
            / self.model.actuator(act_name).gear[0]
        )

    def get_act_joint_velocity(self, act_name):
        """
        Returns velocity of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).velocity) == 1
        return (
            self.data.actuator(act_name).velocity[0]
            / self.model.actuator(act_name).gear[0]
        )

    def get_act_joint_ranges(self):
        """
        Returns the lower and upper limits of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        low, high = self.model.jnt_range[indices, :].T
        return low, high

    def get_act_joint_range(self, act_name):
        """
        Returns the lower and upper limits of given joint.
        """
        low, high = self.model.joint(act_name).range
        return low, high

    def get_actuator_ctrl_range(self):
        """
        Returns the acutator ctrlrange defined in model xml.
        """
        low, high = self.model.actuator_ctrlrange.copy().T
        return low, high

    def get_sensordata(self, sensor_name):
        sensor = self.model.sensor(sensor_name)
        sensor_adr = sensor.adr[0]
        data_dim = sensor.dim[0]
        return self.data.sensordata[sensor_adr : sensor_adr + data_dim]

    def get_interaction_force(self, body1, body2):
        """
        Returns contact force beween a body1 and body2.
        """
        frc = 0
        for i, con in enumerate(self.data.contact):
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            b1 = self.model.body(self.model.geom(con.geom1).bodyid)
            b2 = self.model.body(self.model.geom(con.geom2).bodyid)
            if (b1.name == body1 and b2.name == body2) or (
                b1.name == body2 and b2.name == body1
            ):
                frc += np.linalg.norm(c_array)
        return frc

    def get_body_vel(self, body_name, frame=0):
        """
        Returns translational and rotational velocity of a body in body-centered frame, world/local orientation.
        """
        body_vel = np.zeros(6)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, body_id, body_vel, frame
        )
        return [body_vel[3:6], body_vel[0:3]]

    def get_object_xpos_by_name(self, object_name, object_type):
        if object_type == "OBJ_BODY":
            return self.data.body(object_name).xpos
        elif object_type == "OBJ_GEOM":
            return self.data.geom(object_name).xpos
        elif object_type == "OBJ_SITE":
            return self.data.site(object_name).xpos
        else:
            raise Exception("object type should either be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    def get_object_xquat_by_name(self, object_name, object_type):
        if object_type == "OBJ_BODY":
            return self.data.body(object_name).xquat
        if object_type == "OBJ_GEOM":
            xmat = self.data.geom(object_name).xmat
            return mjxmat2quat(xmat)
        if object_type == "OBJ_SITE":
            xmat = self.data.site(object_name).xmat
            return mjxmat2quat(xmat)
        else:
            raise Exception("object type should be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    def get_object_affine_by_name(self, object_name, object_type):
        """Helper to create transformation matrix from position and quaternion."""
        pos = self.get_object_xpos_by_name(object_name, object_type)
        quat = self.get_object_xquat_by_name(object_name, object_type)
        return mjxpos_mjxquat2trans(pos, quat)

    def get_robot_com(self):
        """
        Returns the center of mass of subtree originating at root body
        i.e. the CoM of the entire robot body in world coordinates.
        """
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreecom" not in sensor_names:
            raise Exception("subtree_com sensor not attached.")
        return self.data.subtree_com[1].copy()

    def get_robot_linmom(self):
        """
        Returns linear momentum of robot in world coordinates.
        """
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreelinvel" not in sensor_names:
            raise Exception("subtree_linvel sensor not attached.")
        linvel = self.data.subtree_linvel[1].copy()
        total_mass = self.get_robot_mass()
        return linvel * total_mass

    def get_robot_angmom(self):
        """
        Return angular momentum of robot's CoM about the world origin.
        """
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreeangmom" not in sensor_names:
            raise Exception("subtree_angmom sensor not attached.")
        com_pos = self.get_robot_com()
        lin_mom = self.get_robot_linmom()
        return self.data.subtree_angmom[1] + np.cross(com_pos, lin_mom)

    def check_self_collisions(self):
        """
        Returns True if there are collisions other than any-geom-floor.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        for i, c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_robot = (
                self.model.body(geom1_body.rootid).name == self.model.body(1).name
            )
            geom2_is_robot = (
                self.model.body(geom2_body.rootid).name == self.model.body(1).name
            )
            if geom1_is_robot and geom2_is_robot:
                return True
        return False

    def get_contact_num(self, qpos):
        """
        Calculates the number of contacts (ncon) in the simulation for a given set of joint positions (qpos).

        This method temporarily updates the simulation's joint positions (`qpos`) to the provided values,
        computes the forward dynamics to determine the number of contacts, and then restores the original
        joint positions to maintain the simulation state.

        Args:
            qpos (array-like): The desired joint positions to set in the simulation.

        Returns:
            int: The number of contacts (`ncon`) in the simulation for the given joint positions.
        """

        current_qpos = copy(self.data.qpos)

        self.data.qpos[self._qpos_ids] = qpos
        mujoco.mj_forward(self.model, self.data)
        ncon = copy(self.data.ncon)
        self.data.qpos = current_qpos
        mujoco.mj_forward(self.model, self.data)

        return ncon
