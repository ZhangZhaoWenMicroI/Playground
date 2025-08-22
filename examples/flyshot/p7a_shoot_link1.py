import os
import imageio  # 用于图像读写
import mujoco  # MuJoCo物理引擎库
import numpy as np  # 数值计算库

# 导入项目内部模块
from playground.Algorithm import BasicAlgorithm  # 算法工具
from playground.Robot.Robot import Robot  # 机器人控制类
from playground.Utils.LoggerUtils import get_logger  # 日志工具

logger = get_logger(__name__)

# --- 常量定义 ---
# 获取项目根目录，假设当前文件在 PROJECT_ROOT/examples/flyshot/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 定义图像和TCP数据保存目录
SAVE_DIR = os.path.join(PROJECT_ROOT, "dataset")

# 确保保存目录存在
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    logger.info(f"创建保存目录: {SAVE_DIR}")


def setup_scene():
    """设置MuJoCo仿真场景，包括机器人、工作站、PC和工件。

    通过从XML文件加载模型并将其组装到父场景中来构建仿真环境。
    """
    # 检查模型文件是否存在并加载主场景文件
    scene_path = os.path.join(PROJECT_ROOT, "model/scene.xml")
    if not os.path.exists(scene_path):
        logger.error(f"场景文件不存在: {scene_path}")
        raise FileNotFoundError(f"场景文件不存在: {scene_path}")
    parent_spec = mujoco.MjSpec.from_file(scene_path)

    # --- 添加机器人模型 ---
    # 定义机器人基座在世界坐标系中的位置和姿态
    robot_frame = parent_spec.worldbody.add_frame(pos=[-0.325, 0, 0.899], euler=[0, 0, 0])
    robot_model_path = os.path.join(PROJECT_ROOT, "model/robot/p7a_900_robot.xml")
    if not os.path.exists(robot_model_path):
        logger.error(f"机器人模型文件不存在: {robot_model_path}")
        raise FileNotFoundError(f"机器人模型文件不存在: {robot_model_path}")
    robot_spec = mujoco.MjSpec.from_file(robot_model_path)

    # 添加末端执行器（相机）模型
    tool_model_path = os.path.join(PROJECT_ROOT, "model/end_effector/camera.xml")
    if not os.path.exists(tool_model_path):
        logger.error(f"末端执行器模型文件不存在: {tool_model_path}")
        raise FileNotFoundError(f"末端执行器模型文件不存在: {tool_model_path}")
    tool_spec = mujoco.MjSpec.from_file(tool_model_path)
    # 将末端执行器连接到机器人的指定站点
    robot_spec.attach(tool_spec, site="attachment_site")
    # 将机器人模型连接到父场景的指定框架，并设置前缀以避免命名冲突
    parent_spec.attach(robot_spec, frame=robot_frame, prefix="robot1/")

    # --- 添加工作站模型 ---
    workstation_frame = parent_spec.worldbody.add_frame(pos=[0, 0, 0])
    workstation_model_path = os.path.join(PROJECT_ROOT, "model/obstacle/workstation.xml")
    if not os.path.exists(workstation_model_path):
        logger.error(f"工作站模型文件不存在: {workstation_model_path}")
        raise FileNotFoundError(f"工作站模型文件不存在: {workstation_model_path}")
    workstation_spec = mujoco.MjSpec.from_file(workstation_model_path)
    parent_spec.attach(workstation_spec, frame=workstation_frame)

    # --- 添加PC（电脑）模型 ---
    pc_frame = parent_spec.worldbody.add_frame(pos=[-0.174, -0.40382, 1.0473], euler=[1.5708, 0, 0])
    pc_model_path = os.path.join(PROJECT_ROOT, "model/obstacle/pc.xml")
    if not os.path.exists(pc_model_path):
        logger.error(f"PC模型文件不存在: {pc_model_path}")
        raise FileNotFoundError(f"PC模型文件不存在: {pc_model_path}")
    pc_spec = mujoco.MjSpec.from_file(pc_model_path)
    parent_spec.attach(pc_spec, frame=pc_frame)

    # --- 添加工件台和工件模型 ---
    stage_frame = parent_spec.worldbody.add_frame(pos=[0.0571, 0.0572, 0.9266], euler=[1.5708, 0, -1.5708])
    stage_model_path = os.path.join(PROJECT_ROOT, "model/stage/stage.xml")
    if not os.path.exists(stage_model_path):
        logger.error(f"工件台模型文件不存在: {stage_model_path}")
        raise FileNotFoundError(f"工件台模型文件不存在: {stage_model_path}")
    stage_spec = mujoco.MjSpec.from_file(stage_model_path)

    workpiece_model_path = os.path.join(PROJECT_ROOT, "model/workpiece/workpiece_link1.xml")
    if not os.path.exists(workpiece_model_path):
        logger.error(f"工件模型文件不存在: {workpiece_model_path}")
        raise FileNotFoundError(f"工件模型文件不存在: {workpiece_model_path}")
    workpiece_spec = mujoco.MjSpec.from_file(workpiece_model_path)

    # 将工件连接到工件台的指定站点
    stage_spec.attach(workpiece_spec, site="stage/attachment_site")
    # 将工件台模型连接到父场景的指定框架
    parent_spec.attach(stage_spec, frame=stage_frame)

    # 编译并返回完整的MuJoCo模型
    return parent_spec.compile()


def capture_images(controller: Robot, gen_tcp_list: np.ndarray):
    """移动机器人到每个目标位姿，捕获相机图像，并记录实际的TCP位姿。

    Args:
        controller (Robot): 机器人控制器实例，用于控制机器人运动和图像捕获。
        gen_tcp_list (np.ndarray): 包含生成的目标TCP位姿列表，每行为 [x, y, z, rx, ry, rz] (毫米, 角度)。
    """
    if not isinstance(controller, Robot):
        logger.error("控制器实例类型不正确，期望 Robot 类型。")
        raise TypeError("控制器实例类型不正确")
    if not isinstance(gen_tcp_list, np.ndarray) or gen_tcp_list.ndim != 2 or gen_tcp_list.shape[1] != 6:
        logger.error("生成TCP列表格式不正确，期望N行6列的numpy数组。")
        raise ValueError("生成TCP列表格式不正确")

    tcp_list_actual = []  # 存储机器人实际到达的TCP位姿
    # 初始化用户场景几何体数量，避免显示旧的几何体
    if controller.viewer and controller.viewer.user_scn:
        controller.viewer.user_scn.ngeom = 0
    else:
        logger.warning("Viewer或user_scn未初始化，无法进行轨迹点可视化。")

    for i, tcp_target in enumerate(gen_tcp_list):
        # 复制一份，避免修改原始数据
        tcp = tcp_target.copy()
        
        # 校验TCP数据范围
        if not np.all(np.isfinite(tcp)):
            logger.warning(f"第 {i+1} 个TCP位姿包含非有限值，跳过。")
            continue

        # 转换单位：毫米转米，角度转弧度
        # MuJoCo通常使用米和弧度作为单位
        tcp[:3] /= 1000.0  # 毫米转米
        tcp[3:] *= np.pi / 180.0  # 角度转弧度

        # 将XYZ-RxRyRz位姿转换为MuJoCo所需的XYZ-四元数格式
        tcp_quat = BasicAlgorithm.xyz_rxryrz2xyz_mjquat(tcp)
        
        # 移动机器人到目标位姿
        # move_line_blocking 会阻塞直到机器人到达目标位姿
        try:
            controller.move_line_blocking(tcp_quat, 2, move_target_robot_site="robot1/attachment_site")
        except Exception as e:
            logger.error(f"移动机器人到第 {i+1} 个位姿失败: {e}")
            continue # 继续下一个位姿

        # 可视化当前轨迹点（如果viewer可用）
        if controller.viewer and controller.viewer.user_scn:
            mujoco.mjv_initGeom(
                controller.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # 球体
                size=[0.01, 0, 0],  # 球体半径0.01米
                pos=tcp_quat[:3],  # 位置
                mat=np.eye(3).flatten(),  # 单位旋转矩阵
                rgba=np.array([1, 0, 0, 1])  # 红色，不透明
            )
        elif i >= mujoco.mjMAXUGEOM:
            logger.warning(f"已达到最大可视化几何体数量 {mujoco.mjMAXUGEOM}，后续轨迹点将不被可视化。")

        # 更新MuJoCo运动学，确保获取最新的机器人状态
        mujoco.mj_kinematics(controller.model, controller.data)
        
        # 获取机器人末端（attachment_site）的实际位姿
        # 将MuJoCo的位姿（位置和旋转矩阵）转换为XYZ-RxRyRz格式（角度）
        tcp_actual = BasicAlgorithm.mjxpos_mjxmat2xyz_rxryrz(
            controller.data.site("robot1/attachment_site").xpos,
            controller.data.site("robot1/attachment_site").xmat.reshape(3, 3),
            "xyz",
            is_degree=True  # 返回角度制
        )
        tcp_list_actual.append(tcp_actual)
        
        # 捕获相机图像并保存
        try:
            rgb_image = controller.capture_camera_image("robot1/end_effector_cam")
            image_save_path = os.path.join(SAVE_DIR, f"{i+1}.jpg")
            imageio.imwrite(image_save_path, rgb_image)
            logger.info(f"图像保存成功: {image_save_path}")
        except Exception as e:
            logger.error(f"捕获或保存第 {i+1} 张图像失败: {e}")

    # 更新可视化场景中几何体的数量，使其与实际捕获的位姿数量一致
    if controller.viewer and controller.viewer.user_scn:
        controller.viewer.user_scn.ngeom = len(tcp_list_actual)

    # 将实际TCP位姿列表转换为numpy数组
    tcp_list_actual = np.array(tcp_list_actual)
    
    # 校验实际TCP列表是否为空
    if tcp_list_actual.size == 0:
        logger.warning("没有捕获到任何实际TCP位姿，跳过保存。")
        return

    # 将实际TCP位姿的XYZ部分从米转回毫米，以便与输入单位保持一致
    tcp_list_actual[:, :3] *= 1000.0  # 转回毫米单位
    
    # 保存实际TCP数据到文件
    tcp_real_path = os.path.join(SAVE_DIR, "tcp_real.txt")
    try:
        np.savetxt(
            tcp_real_path,
            tcp_list_actual,
            header="x,y,z,rx,ry,rz",  # 文件头，描述列内容
            comments="",  # 不添加注释前缀
            delimiter=",",  # 使用逗号作为分隔符
            fmt="%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"  # 浮点数格式，保留三位小数
        )
        logger.info(f"实际TCP数据保存成功: {tcp_real_path}")
    except Exception as e:
        logger.error(f"保存实际TCP数据失败: {e}")


def main():
    """主函数：设置仿真环境，初始化机器人，执行图像捕获流程。"""
    logger.info("--- 开始设置仿真场景 ---")
    try:
        # 设置MuJoCo场景并编译模型
        model = setup_scene()
        data = mujoco.MjData(model)
        logger.info("仿真场景设置成功。")
    except FileNotFoundError as e:
        logger.error(f"场景或模型文件缺失，无法启动仿真: {e}")
        return
    except Exception as e:
        logger.error(f"设置仿真场景时发生未知错误: {e}")
        return

    logger.info("--- 初始化机器人控制器 ---")
    # 打印模型关键信息，用于调试
    logger.info(f"模型信息: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, nsite={model.nsite}")
    # 初始化机器人控制器，并启用可视化和渲染
    controller = Robot(model, data, is_viewer=True, is_render=True)
    logger.info("机器人控制器初始化成功。")

    logger.info("--- 移动机器人到初始位置 ---")
    try:
        # 移动机器人到预设的“p7a_home”位置
        controller.move_keypos("p7a_home")
        # 更新运动学，确保机器人状态正确
        mujoco.mj_kinematics(model, data)
        logger.info("机器人已移动到初始位置 'p7a_home'。")
    except Exception as e:
        logger.error(f"移动机器人到初始位置失败: {e}")
        return

    logger.info("--- 读取目标位姿列表 ---")
    tcp_generate_path = os.path.join(SAVE_DIR, "tcp_generate.txt")
    if not os.path.exists(tcp_generate_path):
        logger.error(f"目标位姿文件不存在: {tcp_generate_path}")
        return
    try:
        # 从文件中加载目标TCP位姿列表，跳过第一行（通常是表头）
        gen_tcp_list = np.loadtxt(tcp_generate_path, delimiter=",", skiprows=1)
        logger.info(f"成功读取 {len(gen_tcp_list)} 个目标位姿。")
    except Exception as e:
        logger.error(f"读取目标位姿文件失败: {e}")
        return

    logger.info("--- 执行图像捕获 ---")
    # 执行图像捕获和TCP数据记录
    capture_images(controller, gen_tcp_list)
    logger.info("图像捕获和TCP数据记录完成。")


if __name__ == "__main__":
    main()
