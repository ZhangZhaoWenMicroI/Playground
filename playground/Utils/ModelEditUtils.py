"""
Model editing utilities for MuJoCo XML models (pre-compile).
"""
import numpy as np
import mujoco as mj
from typing import Any, Optional
from playground.Utils.LoggerUtils import get_logger

logger = get_logger(__name__)


# -------------------- 基础操作 --------------------
def load_spec_from_xml(xml_path: str) -> Any:
    """Load mjSpec model from XML file."""
    return mj.MjSpec.from_file(xml_path)


def load_spec_from_string(xml_string: str) -> Any:
    """Load mjSpec model from XML string."""
    return mj.MjSpec.from_string(xml_string)


def save_spec_to_xml(spec: Any, xml_path: str) -> None:
    """Save mjSpec model to XML file."""
    with open(xml_path, "w") as f:
        f.write(spec.to_xml())


def spec_to_string(spec: Any) -> str:
    """Convert mjSpec model to XML string."""
    return spec.to_xml()


# -------------------- 遍历与查找 --------------------
def find_body(spec, name):
    """根据名称查找body。"""
    return spec.body(name)


def find_geom(spec, name):
    """根据名称查找geom。"""
    for geom in spec.geoms:
        if geom.name == name:
            return geom
    return None


def find_joint(spec, name):
    """根据名称查找joint。"""
    for joint in spec.joints:
        if joint.name == name:
            return joint
    return None


def find_actuator(spec, name):
    """根据名称查找actuator。"""
    for actuator in spec.actuators:
        if actuator.name == name:
            return actuator
    return None


# -------------------- 添加与删除 --------------------
def add_body(parent, name, pos=None, **kwargs):
    """在parent下添加body。"""
    return parent.add_body(name=name, pos=pos, **kwargs)


def add_geom(body, type, size, pos=None, rgba=None, **kwargs):
    """在body下添加geom。"""
    return body.add_geom(type=type, size=size, pos=pos, rgba=rgba, **kwargs)


def add_joint(body, type, axis=None, pos=None, **kwargs):
    """在body下添加joint。"""
    return body.add_joint(type=type, axis=axis, pos=pos, **kwargs)


def detach_body(spec, body):
    """从spec中分离body。"""
    return spec.detach_body(body)


# -------------------- 拼接与合成 --------------------
def attach_body(frame, body, *args, **kwargs):
    """将body拼接到frame。"""
    return frame.attach_body(body, *args, **kwargs)


def add_frame(parent, pos=None, quat=None, **kwargs):
    """在parent下添加frame。"""
    return parent.add_frame(pos=pos, quat=quat, **kwargs)


# -------------------- 缩放 --------------------
def scale_spec(spec, scale, scale_actuators=False):
    """对spec模型进行缩放。"""
    scaled_spec = spec.copy()

    def scale_bodies(parent, scale=1.0):
        body = parent.first_body()
        while body:
            if body.pos is not None:
                body.pos = np.array(body.pos) * scale
            for geom in body.geoms:
                if hasattr(geom, "fromto") and geom.fromto is not None:
                    geom.fromto = np.array(geom.fromto) * scale
                if hasattr(geom, "size") and geom.size is not None:
                    geom.size = np.array(geom.size) * scale
                if geom.pos is not None:
                    geom.pos = np.array(geom.pos) * scale
            scale_bodies(body, scale)
            body = parent.next_body(body)

    scale_bodies(scaled_spec.body("world"), scale)
    if scale_actuators:
        for actuator in scaled_spec.actuators:
            if hasattr(actuator, "gear") and actuator.gear is not None:
                actuator.gear = np.array(actuator.gear) * scale * scale
    return scaled_spec


# -------------------- 其他实用函数 --------------------
def print_bodies(parent, level=0):
    """递归打印body树结构。"""
    body = parent.first_body()
    while body:
        logger.info("  " * level + f"- {body.name}")
        print_bodies(body, level + 1)
        body = parent.next_body(body)


# -------------------- 删除元素 --------------------
def delete_body(parent, body):
    """从parent中删除body。"""
    return parent.remove_body(body)


def delete_geom(body, geom):
    """从body中删除geom。"""
    return body.remove_geom(geom)


def delete_joint(body, joint):
    """从body中删除joint。"""
    return body.remove_joint(joint)


def delete_actuator(spec, actuator):
    """从spec中删除actuator。"""
    return spec.actuators.remove(actuator)


def delete_frame(parent, frame):
    """从parent中删除frame。"""
    return parent.remove_frame(frame)


# -------------------- 复制元素 --------------------
def copy_body(body):
    """深复制一个body（不挂载到任何parent）。"""
    return body.copy()


def copy_geom(geom):
    """深复制一个geom。"""
    return geom.copy()


def copy_joint(joint):
    """深复制一个joint。"""
    return joint.copy()


# -------------------- 移动元素 --------------------
def move_body(body, new_pos):
    """移动body到新位置。"""
    body.pos = np.array(new_pos)
    return body


def move_geom(body, geom_name, new_pos):
    """移动body下指定geom到新位置。"""
    geom = find_geom(body, geom_name)
    if geom:
        geom.pos = np.array(new_pos)
    return geom


# -------------------- 批量操作 --------------------
def batch_rename_bodies(parent, prefix):
    """递归批量重命名body，添加前缀。"""
    body = parent.first_body()
    while body:
        body.name = f"{prefix}_{body.name}"
        batch_rename_bodies(body, prefix)
        body = parent.next_body(body)


def batch_set_geom_rgba(parent, rgba):
    """递归批量设置所有body下geom的颜色。"""
    body = parent.first_body()
    while body:
        for geom in body.geoms:
            geom.rgba = rgba
        batch_set_geom_rgba(body, rgba)
        body = parent.next_body(body)


# -------------------- 合并模型 --------------------
def merge_specs(spec1, spec2, attach_to="worldbody", pos=None):
    """将spec2合并到spec1，默认挂到spec1的worldbody。"""
    if pos is None:
        pos = [0, 0, 0]
    frame = getattr(spec1, attach_to).add_frame(pos=pos)
    frame.attach_body(spec2.worldbody, "merged", "")
    return spec1


# -------------------- 随机化 --------------------
def randomize_geom_rgba(parent, alpha=1.0):
    """递归随机设置所有body下geom的颜色。"""
    body = parent.first_body()
    while body:
        for geom in body.geoms:
            geom.rgba = [np.random.rand(), np.random.rand(), np.random.rand(), alpha]
        randomize_geom_rgba(body, alpha)
        body = parent.next_body(body)


# -------------------- 导出/导入body子树 --------------------
def export_body_subtree(body):
    """导出body及其子树为新的mjSpec。"""
    spec = mj.MjSpec()
    new_body = body.copy()
    spec.worldbody.add_body(new_body)
    return spec


def import_body_subtree(spec, parent, subtree_spec):
    """将subtree_spec的worldbody挂载到parent。"""
    for body in subtree_spec.worldbody.bodies:
        parent.add_body(body.copy())
    return parent
