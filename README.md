# Playground

基于 MuJoCo 的轻量化仿真工具，支持机械臂运动和相机采图:
- 轨迹规划与执行
- 机械臂运动学/动力学仿真
- 相机采图与图像集自动落盘
- 友好的可视化与交互

![flyshot](https://private-user-images.githubusercontent.com/227360208/480925619-c8034837-c9ca-40cc-af3c-abdcd2494306.mp4)



## 为什么值得一试？
- 一行命令即可跑通示例场景: 环境建模 → 机械臂运动 → 相机采图 → 图像数据落盘
- 真实工业场景建模: 机械臂、载台、末端相机等组件可复用
- 机械臂运动学: 内置逆解和多点轨迹规划，支持平滑执行
- 数据生产友好: 自动保存 RGB 图像与末端 TCP 轨迹
- 代码工程化: 核心能力模块化封装，易于二次开发



## 目录结构
```
├── model/                      # 3D 模型和场景文件
│   ├── assets/                 # STL 模型文件
│   ├── end_effector/           # 末端执行器（相机）
│   ├── obstacle/               # 障碍物
│   ├── robot/                  # 机械臂
│   ├── stage/                  # 载台
│   └── workpiece/              # 工件
├── dataset/                    # 图像数据与 TCP
├── docs/                       # 文档
├── examples/
│   └── flyshot/
│       └── p7a_shoot_link1.py  # 飞拍采图示例（推荐从这里开始）
└── playground/                 # 核心代码
    ├── Algorithm/              # 基础算法
    ├── MotionPlan/             # 运动规划
    ├── Mujoco/                 # MuJoCo 接口
    ├── Robot/                  # 机械臂封装
    └── Utils/                  # 工具类
```



## 快速开始

环境安装
```bash
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv sync
```

运行示例
```bash
python -m examples.flyshot.p7a_shoot_link1
```

你将看到:
- 自动组装并启动包含机械臂/相机/载台/工件的 MuJoCo 场景
- 读取 dataset/tcp_generate.txt 的目标 TCP 点位
- 机器人按位姿序列平滑运动并在每个点位采图
- 输出每个点位采集的图像
  - dataset/1.jpg, 2.jpg, ...          



## Star 与贡献
如果这个项目对你有帮助，欢迎点亮 Star！也欢迎提交 Issue/PR 一起完善 :)

 





