# LBot ROS2 SDK 资料总览

本机相机启动、实时预览、采图及模型训练命令见 [视觉模块指令手册](视觉模块指令手册.md)。

## 螺母视觉与双臂抓放

当前工作区提供采图、YOLO 大中小螺母检测、中心定位、深度坐标转换和双臂抓放接入。
完整任务说明见 [tools/NUT_TASK.md](tools/NUT_TASK.md)，标定说明见
[tools/CALIBRATION.md](tools/CALIBRATION.md)。下列命令使用 **zsh**；Bash 终端将
`setup.zsh` 换为 `setup.bash`。在仓库根目录运行。

### 环境与相机

- ROS 2 Jazzy、系统 Python 3、`rclpy`、`cv_bridge`、OpenCV、NumPy、SciPy、PyYAML。
- 实时中文界面还需要 Pillow 和 Noto CJK 字体；Ubuntu 可安装
  `sudo apt install python3-pil fonts-noto-cjk`。
- 模型推理使用独立 Conda 环境，与 ROS Python 隔离：

```zsh
conda create -n nut-yolo python=3.11 -y
conda activate nut-yolo
python -m pip install ultralytics
conda deactivate
```

已安装 Orbbec ROS 2 驱动后，在独立终端启动 Gemini 2：

```zsh
source /opt/ros/jazzy/setup.zsh
source ~/orbbec_ws/install/setup.zsh
ros2 launch orbbec_camera gemini2.launch.py \
  enable_color:=true enable_depth:=true depth_registration:=true \
  enable_point_cloud:=false
```

需要能订阅 `/camera/color/image_raw`、`/camera/depth/image_raw` 和
`/camera/color/camera_info`。远程运行时确认 ROS 网络和 `ROS_DOMAIN_ID` 一致。
只有话题名不足以证明有画面，可用 `ros2 topic info /camera/color/image_raw` 检查发布者。

### 实时视频、中心与三维坐标

```zsh
source /opt/ros/jazzy/setup.zsh
/usr/bin/python3 tools/nut_yolo_live.py --device 0

# 四类别模型（大、中、小、白色）
/usr/bin/python3 tools/nut_yolo_live.py --device 0 \
  --model weights/nut_white_best.pt
```

铁杆关键点使用同一入口，只更换模型：

```zsh
/usr/bin/python3 tools/nut_yolo_live.py --device 0 --conf 0.5 \
  --model weights/needle_pose_best.pt
```

自动使用 Pose 960 输入（螺母为 640）；显示 tip/base 像素坐标，铁杆暂不输出 XYZ。
相机仍需发布同步彩色、深度和内参。按 `s` 保存的 JSON 包含 `keypoints`。

窗口左侧显示大/中/小检测框（四类模型还显示白色）和中心十字，右侧显示置信度、像素中心、相机 XYZ 和
机器人 `base_link` XYZ（米）。模型常驻后台，仅处理最新配对帧。
`--device cpu` 使用 CPU，`--scale 0.6` 缩小窗口，`--conf 0.6` 调整检测阈值。
实时预览默认 `--device auto`，在 Conda 推理进程内自动选择 CUDA GPU，无法使用 CUDA
时选择 CPU；指定 `--device 0` 可强制使用 GPU。首次加载需要预热。
单次推理超过 `--inference-timeout 60` 秒或子进程出错时，窗口保留并自动重启推理进程。
相机订阅仅保留最新消息，帧以未压缩数组传给推理进程，减少排队和编码延迟。
按 `s` 保存画面和 JSON；按 `q`、ESC 或关闭窗口退出。正常退出保存最后一次快照。
输出目录为 `recordings/yolo_live/<时间戳>/`。深度无效时显示错误，断流或帧过期时隐藏旧坐标。
这个入口没有机械臂运动客户端。
`--model` 仅覆盖本次预览的模型，不修改默认抓取配置。白色目标的 JSON 标签为
`white`，与其他目标使用相同的深度坐标解算；机械臂业务仍使用原有 l/m/s 抓取流程。

### 单次检测与离线复现

```zsh
# 实时获取一对彩色/深度并打印坐标，不运动
/usr/bin/python3 tools/nut_yolo_preview.py
# 只有照片：只输出像素中心
/usr/bin/python3 tools/nut_yolo_preview.py --image /path/to/photo.png
# 已有同帧彩色、对齐深度和内参：计算 XYZ
/usr/bin/python3 tools/nut_yolo_preview.py --image /path/to/color.png \
  --depth /path/to/depth.npy --camera-info /path/to/camera_info.yaml
```

单次预览输出到 `recordings/yolo_preview/<时间戳>/`，包含中心标注图、原彩色图、深度、
内参和 `detections.json`。深度支持 uint16 毫米 PNG 或浮点米 NPY，不能使用深度伪彩图。

### 主业务入口

```zsh
# 读取实时检测并打印抓放计划，不运动
/usr/bin/python3 tools/nut_pick_place.py --detector yolo
# 实际驱动双臂：先回 home，再检测、IK 预检、抓取与交接
source install/setup.zsh
/usr/bin/python3 tools/nut_pick_place.py --detector yolo --execute
```

链路为 `YOLO 框中心 → 对齐深度中位数 → 内参反投影 → 相机 XYZ → 外参换算 →
base_link XYZ → 左臂抓取 → 双臂交接序列`。缺少要求的类别或同类多目标时，主业务中止。
配置位于 [nut_task.yaml](开发资源/nut_sort/nut_task.yaml) 的 `detector` 段，默认仍为 manual，
用 `--detector yolo` 切换。更换电脑时修改 Conda `python` 路径；更换相机安装后重新核对标定。

### 抓取失败后的安全回退

抓取失败或任务中断后，先沿示教的 ready 轨迹**逆向**退回开机安全起始位，再重跑主业务：

```zsh
source /opt/ros/jazzy/setup.zsh && source install/setup.zsh
/usr/bin/python3 tools/nut_return.py            # dry-run：只打印逆向链与关节差
/usr/bin/python3 tools/nut_return.py --execute  # 真机：自动上使能 + 逆向回放（左先右后）
```

终点是 `left/right.ready` 段第 0 点，也就是框架开机时接入的那个安全起点，因此重跑
`nut_pick_place` 会安全重放 ready，而不会从桌面上的姿态无避障盲动。已在安全位则一步不发，
可反复执行。

**抓取到一半（臂悬在桌面中央、或停在任务段回位点）也能收回来**：这种姿态不在 ready 轨迹上，
脚本自动走「安全再接近」——竖直上到中转平面，再同高横移到 ready 末点正上方、竖直压入；
若是直线横移会撞上逆解奇异区，就改走关节空间整段插值到 ready 末点的**记录臂型**，并用驱动
正解逐点校验工具离桌面的高度。两条路线都是**全部算完、校验通过才开始运动**，任何一点不过就
在运动前中止（臂一行都不动）。`--blind-join` 是不做再接近、直接低速接入 ready 末点的逃生口
（无避障，只在现场空旷时用）。详见 [tools/NUT_TASK.md](tools/NUT_TASK.md) 第 10 节。

### 拍照与标注数据

```zsh
/usr/bin/python3 tools/capture_nut_images.py
# 每 3 秒一张，共 100 张；SSH 无窗口可追加 --no-preview
/usr/bin/python3 tools/capture_nut_images.py --interval 3 --count 100
```

空格/`s` 拍照，`a` 切换定时采集，`q` 退出。默认原图保存到
`/home/dawntildusk/nut_vision/raw/<时间戳>/images/`，可用 `--output` 修改根目录。
上传原图至 CVAT，使用 large/medium/small 矩形标签，导出 Ultralytics YOLO Detection 格式。
训练时图片与标签保持同名，类别编号以导出配置为准。

### 模型、验证与限制

运行使用 [weights/nut_best.pt](weights/nut_best.pt)，来源和 SHA256 在
[weights/nut_best.json](weights/nut_best.json)。这是在 61 张标注照片上追加训练 100 轮的选定模型；
训练集自检 mAP50 约 97.8%，**不代表独立测试效果**。未训练照片测试仍存在漏检与机械臂部件误检。
中心取检测框中心，未精定位孔轮廓；中心邻域深度可能落到桌面，不能直接认为是指尖接触点。
当前外参记录平均残差约 10.6 mm，双臂交接点差异约 69 mm，执行前需完成现场坐标与路径核对。

```zsh
source /opt/ros/jazzy/setup.zsh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s tools -p 'test_*.py'
# 只读连接真实相机，限时验证实时检测
/usr/bin/python3 tools/nut_yolo_live.py --device 0 --no-preview --duration 15
```

训练环境为 Python 3.11、Ultralytics 8.4.146；现场已验证 GPU 常驻推理和真实 RGB/深度重放。
测试使用模拟运动接口，不会驱动实物。

## 这份资料是什么

本目录是本次赛事提供给参赛队的 ROS2 SDK 工作空间：

```bash
/home/simon/lbot_ws
```

它主要用于控制双臂机器人、灵巧手、遥操作，以及查看随包提供的机器人模型资源。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `src/lbot_arm_interfaces` | ROS2 自定义消息和服务 |
| `src/lbot_driver` | 机械臂驱动，负责连接控制器、发布状态、提供服务 |
| `src/lbot_demo` | 示例程序 |
| `src/lbot_teleop` | 遥操作示例 |
| `开发资源/机械臂控制接口文档v1.0.5.pdf` | 底层接口说明 |
| `开发资源/机器人控制平台说明文档v1.1.1.pdf` | Web 控制平台说明 |
| `开发资源/assets` | URDF、MJCF、STL 模型资源 |

## 推荐阅读顺序

1. `开发环境说明.txt`
2. `控制接口说明.txt`
3. `机器人模型说明.txt`
4. `灵巧手说明.txt`
5. `相机与视觉传感器说明.txt`
6. `手眼标定说明.txt`
7. `安全操作手册.txt`
8. `现场快速检查表.txt`
9. `常见问题FAQ.txt`

## 快速启动

```bash
cd /home/simon/lbot_ws
source /opt/ros/jazzy/setup.bash

cd src/lbot_driver/lib
sudo ./lib_install.sh

cd /home/simon/lbot_ws
colcon build
source install/setup.bash

ros2 launch lbot_driver lbot_start_driver.launch.py
```

默认机器人 IP：

```text
192.168.10.21
```

默认命名空间：

```text
/robot1
```

## 常用命令

| 目的 | 命令 |
| --- | --- |
| 查看节点 | `ros2 node list` |
| 查看话题 | `ros2 topic list` |
| 查看服务 | `ros2 service list` |
| 查看左臂关节状态 | `ros2 topic echo /robot1/left_arm/joint_states` |
| 启动驱动 | `ros2 launch lbot_driver lbot_start_driver.launch.py` |
| 启动示例 | `ros2 launch lbot_demo lbot_start_demo.launch.py` |
| 启动遥操作 | `ros2 launch lbot_teleop lbot_start_teleop.launch.py` |

## 视觉说明

本次资料记录的相机型号是 Orbbec Gemini2。相机 SDK 不在当前 `lbot_ws/src` 里，参赛队需要按 `相机与视觉传感器说明.txt` 单独安装 Orbbec ROS2 或 Python SDK。

如果比赛任务需要视觉抓取，请同时确认：

- Gemini2 图像、深度图、点云能正常发布。
- 相机内参已经保存。
- 相机到机器人 base 的外参已经完成标定。

## 重要提醒

- 运动前确认急停未触发、机械臂已使能、工作区无人。
- 关节角单位是弧度 `rad`，不是角度。
- 位置单位是米 `m`。
- 首次运行请使用低速、小幅度动作。
- 本资料以当前目录里的 SDK 和模型文件为准。
