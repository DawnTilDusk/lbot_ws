# ROS2 机器人监控指令手册

本手册汇总日常监控双臂机器人（lbot）时最常用的 ROS2 命令行指令，覆盖**节点、话题、服务、参数、TF、日志、相机、录包**等内容，并结合本机实际的话题和节点命名。

- 默认命名空间：`/robot1`
- 驱动内部三个节点：`lbot_main_node`（连接管理 + 状态发布 + 关节跟随）、`lbot_left_arm_node`（左臂服务）、`lbot_right_arm_node`（右臂服务）
- 驱动约 **50 Hz** 发布状态
- 关节角单位 `rad`，末端位置单位 `m`，欧拉角单位 `rad`

> 控制相关（运动、使能、急停、灵巧手）详见《控制接口说明.md》，本手册只讲**监控/查看**，不涉及下发动作。

---

## 0. 每个终端先做的事

打开新终端后，先 source 工作空间（否则找不到 `lbot_*` 包和自定义接口）：

```bash
source /opt/ros/<distro>/setup.bash          # 系统 ROS2，<distro> 按实际版本替换，如 humble
cd ~/lbot_ws && source install/setup.bash    # 本工作空间
```

查看 ROS2 版本 / 是否 source 成功：

```bash
echo $ROS_DISTRO
ros2 pkg list | grep lbot                    # 能看到 lbot_driver / lbot_arm_interfaces 等即正常
```

如果是多机通信（机器人主机 + 自己电脑），需保证 `ROS_DOMAIN_ID` 一致：

```bash
export ROS_DOMAIN_ID=42                      # 两端必须相同，按现场约定
```

---

## 1. 节点状态（node）

### 1.1 查看所有在线节点

```bash
ros2 node list
```

正常情况下应能看到（在 `/robot1` 命名空间下）：

```
/robot1/lbot_main_node
/robot1/lbot_left_arm_node
/robot1/lbot_right_arm_node
/相机节点...   # 如 /camera/camera
```

> 如果只能看到自己、看不到驱动节点：先确认驱动是否启动、`ROS_DOMAIN_ID` 是否一致、网络是否互通。

### 1.2 查看某个节点的详细信息

```bash
ros2 node info /robot1/lbot_main_node
```

会列出该节点订阅/发布的所有话题、提供的服务、参数，是排查"话题对不对、谁在发谁在收"的最快方法。

```bash
ros2 node info /robot1/lbot_left_arm_node     # 左臂服务节点（含运动/使能/工具坐标系等服务）
```

---

## 2. 话题监控（topic）—— 获取当前状态的核心

### 2.1 列出所有话题

```bash
ros2 topic list                # 所有话题
ros2 topic list -t            # 同时显示话题类型（推荐）
```

与本机状态相关的关键话题：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/robot1/left_arm/joint_states`  | `sensor_msgs/msg/JointState`  | 左臂 7 关节状态（位置/速度/力矩） |
| `/robot1/right_arm/joint_states` | `sensor_msgs/msg/JointState`  | 右臂 7 关节状态 |
| `/robot1/left_arm/pose_states`   | `geometry_msgs/msg/PoseStamped` | 左臂末端位姿 |
| `/robot1/right_arm/pose_states`  | `geometry_msgs/msg/PoseStamped` | 右臂末端位姿 |
| `/robot1/left_arm/joint_follow`  | `lbot_arm_interfaces/msg/FollowJoint` | 左臂跟随（订阅端，勿乱发） |
| `/robot1/right_arm/joint_follow` | `lbot_arm_interfaces/msg/FollowJoint` | 右臂跟随 |
| `/robot1/left_hand/set_l6_*`     | `std_msgs/msg/UInt8MultiArray` | 左手位置/力矩/速度（订阅端） |
| `/robot1/right_hand/set_l6_*`    | `std_msgs/msg/UInt8MultiArray` | 右手位置/力矩/速度（订阅端） |
| `/camera/color/image_raw` 等     | `sensor_msgs/msg/Image` | 相机图像（见第 7 节） |

### 2.2 实时查看话题内容

```bash
# 左臂关节状态（持续刷新，Ctrl+C 退出）
ros2 topic echo /robot1/left_arm/joint_states

# 只看一帧就退出（脚本里常用）
ros2 topic echo /robot1/left_arm/joint_states --once

# 右臂末端位姿
ros2 topic echo /robot1/right_arm/pose_states
```

`joint_states` 里重点看：
- `name`：7 个关节名
- `position`：7 个关节角（rad）
- `velocity` / `effort`：速度 / 力矩（若驱动提供）

### 2.3 查看发布频率（判断驱动是否活着）

```bash
ros2 topic hz /robot1/left_arm/joint_states     # 应接近 50 Hz
ros2 topic hz /robot1/right_arm/pose_states
```

- 频率稳定在 ~50 Hz：驱动与控制器通信正常。
- 频率为 0 / 卡住 / 大幅波动：驱动断连或控制器异常，去看驱动日志（第 6 节）。

### 2.4 查看带宽占用

```bash
ros2 topic bw /camera/depth/color/points        # 点云带宽很大，排查网络/卡顿时常看
ros2 topic bw /camera/color/image_raw
```

### 2.5 查看话题类型与字段结构

```bash
ros2 topic info /robot1/left_arm/joint_states          # 看类型、发布者/订阅者数量
ros2 interface show sensor_msgs/msg/JointState         # 看消息字段定义
ros2 interface show lbot_arm_interfaces/msg/ArmState   # 自定义接口：joints/euler/pose
```

### 2.6 快速统计发布者/订阅者数量

`ros2 topic info -v <topic>` 可看到具体是哪些节点在发布、哪些在订阅，排查"没人发"或"没人收"很有用：

```bash
ros2 topic info -v /robot1/left_arm/joint_states
```

---

## 3. 服务查看（service）—— 查询状态与配置

### 3.1 列出所有服务

```bash
ros2 service list                          # 所有服务
ros2 service list -t                       # 带类型
ros2 service list | grep left_arm          # 只看左臂相关
```

### 3.2 查询类服务（只读，安全，可随意调用）

```bash
# 查询当前工具坐标系
ros2 service call /robot1/left_arm/get_current_tool_frame lbot_arm_interfaces/srv/GetCurrentFrame

# 查询全部已保存的工具坐标系名称
ros2 service call /robot1/left_arm/get_all_tool_frames lbot_arm_interfaces/srv/GetAllFrames

# 正解：给 7 个关节角，算末端位姿（不动作，纯计算，可用来验证目标点）
ros2 service call /robot1/left_arm/forward_kinematics lbot_arm_interfaces/srv/ForwardKinematics \
  "{joints: [0, 0, 0, 0, 0, 0, 0]}"

# 逆解：给目标位姿，算关节角（不动作，纯计算）
ros2 service call /robot1/left_arm/inverse_kinematics lbot_arm_interfaces/srv/InverseKinematics \
  "{position: {x: 0.3, y: 0.1, z: -0.2}, euler: {x: 0.0, y: -1.57, z: 0.0}}"
```

> 运动类服务（`move_joint` / `move_pose` / `move_linear`）和系统类服务（`set_enable` / `set_emergency_stop` / `set_zero`）**会让机器人动作或改变状态**，不属于监控范畴，详见《控制接口说明.md》。`set_zero` 尤其危险，调试时不要乱调。

### 3.3 查看服务类型

```bash
ros2 service type /robot1/left_arm/move_joint
ros2 interface show lbot_arm_interfaces/srv/MoveJ
```

---

## 4. 参数查看（param）

```bash
ros2 param list /robot1/lbot_main_node               # 列出某节点所有参数
ros2 param get /robot1/lbot_main_node arm_ip         # 查看单个参数（控制器 IP）
ros2 param dump /robot1/lbot_main_node               # 导出该节点全部参数到文件
```

驱动参数里最常看的是 `arm_ip`（默认 `192.168.10.21`），连不上控制器时先确认它。

---

## 5. 坐标变换监控（tf）—— 手眼/相机标定相关

```bash
# 查看完整 TF 树里所有 frame
ros2 run tf2_tools view_frames        # 生成 frames.pdf，直观看树结构

# 实时查看两个 frame 之间的变换
ros2 run tf2_ros tf2_echo base_torso_root camera_color_optical_frame
ros2 run tf2_ros tf2_echo arm_left_L8_Link camera_color_optical_frame

# 查看某节点发布的 TF 频率
ros2 topic hz /tf
ros2 topic hz /tf_static
```

手眼标定、相机外参排查时常用。标定流程见《手眼标定说明.md》与 `tools/` 下的标定脚本。

---

## 6. 日志与诊断

### 6.1 看驱动节点日志

驱动用 `output='screen'` 启动，正常信息直接打印在启动它的终端里。也可用 ROS2 日志目录：

```bash
# 最近一次运行的日志
ls -lt ~/.ros/log/latest/
# 按节点看日志
ros2 run rqt_console rqt_console       # 图形化日志查看，可按级别过滤
```

### 6.2 设置日志级别（临时多看调试信息）

```bash
# 仅运行时生效，重启恢复
ros2 run rqt_logger_level rqt_logger_level
# 或命令行：
ros2 logger set /robot1/lbot_main_node DEBUG
```

### 6.3 快速健康检查清单

| 检查项 | 命令 | 正常表现 |
| --- | --- | --- |
| 驱动节点在线 | `ros2 node list \| grep lbot` | 三个节点都在 |
| 状态在发布 | `ros2 topic hz /robot1/left_arm/joint_states` | ~50 Hz |
| 关节角合理 | `ros2 topic echo /robot1/left_arm/joint_states --once` | position 7 个值、无 NaN |
| 末端位姿 | `ros2 topic echo /robot1/left_arm/pose_states --once` | 位置在工作空间内 |
| 控制器 IP | `ros2 param get /robot1/lbot_main_node arm_ip` | `192.168.10.21` |
| 相机识别 | `ros2 run orbbec_camera list_devices_node` | 能列出 Gemini2 |
| 相机图像 | `ros2 topic hz /camera/color/image_raw` | 有稳定帧率 |
| TF 完整 | `ros2 run tf2_tools view_frames` | camera/arm frame 都在树里 |

---

## 7. 相机监控（Orbbec Gemini2）

相机话题命名空间默认 `/camera`：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | 彩色图 |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 彩色内参 |
| `/camera/depth/image_raw`（或 `image_rect_raw`） | `sensor_msgs/msg/Image` | 深度图 |
| `/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | 深度内参 |
| `/camera/depth/color/points` | `sensor_msgs/msg/PointCloud2` | 点云 |

```bash
# 相机是否被系统识别
ros2 run orbbec_camera list_devices_node

# 有没有图像话题
ros2 topic list | grep image

# 图像帧率
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_rect_raw

# 看内参（标定/投影要用）
ros2 topic echo /camera/color/camera_info --once
ros2 topic echo /camera/depth/camera_info --once

# 点云带宽（很大，卡顿时重点看）
ros2 topic bw /camera/depth/color/points
```

图形化看画面：

```bash
ros2 run rqt_image_view rqt_image_view      # 下拉选 image 话题
```

相机启动、标定、外参详见《相机与视觉传感器说明.md》。

---

## 8. 录包与回放（ros2 bag）

工作区下已有 `recordings/` 目录用于存放数据包。录包用于**离线复盘、给没在现场的同学复现问题**。

```bash
# 录制指定话题（状态 + 相机），存成带时间戳的目录
ros2 bag record -o recordings/debug_$(date +%Y%m%d_%H%M%S) \
  /robot1/left_arm/joint_states \
  /robot1/right_arm/joint_states \
  /robot1/left_arm/pose_states \
  /robot1/right_arm/pose_states \
  /camera/color/image_raw

# 录制某命名空间下所有话题
ros2 bag record -o recordings/all_robot /robot1

# 查看包信息（时长、话题、消息数、类型）
ros2 bag info recordings/debug_xxxx/

# 回放（默认按原速）
ros2 bag play recordings/debug_xxxx/
ros2 bag play -r 0.5 recordings/debug_xxxx/     # 半速回放，便于慢看
ros2 bag play -l recordings/debug_xxxx/         # 循环回放
```

> 点云/图像话题占空间大，长时间录包注意磁盘；只排查机械臂逻辑时可不录相机。

---

## 9. 图形化工具（rqt 全家桶）

```bash
rqt                                  # 总面板，Plugins 里按需选
ros2 run rqt_graph rqt_graph         # 节点-话题连接图，看数据流最直观
ros2 run rqt_topic rqt_topic         # 所有话题 + 频率 + 带宽，勾选即看
ros2 run rqt_plot rqt_plot           # 把数值画成曲线
ros2 run rqt_image_view rqt_image_view  # 看相机画面
ros2 run rqt_console rqt_console     # 日志
```

`rqt_plot` 实时画关节角曲线，观察运动是否平稳特别有用：

```bash
ros2 run rqt_plot rqt_plot \
  /robot1/left_arm/joint_states/position[0] \
  /robot1/left_arm/joint_states/position[1]
```

---

## 10. 常用一行速查

```bash
ros2 node list                                    # 谁在线
ros2 node info /robot1/lbot_main_node             # 某节点详情
ros2 topic list -t                                # 所有话题+类型
ros2 topic echo <topic> --once                    # 看一帧
ros2 topic hz <topic>                             # 发布频率
ros2 topic bw <topic>                             # 带宽
ros2 topic info -v <topic>                        # 谁发谁收
ros2 service list | grep left_arm                 # 找服务
ros2 param get /robot1/lbot_main_node arm_ip      # 看控制器 IP
ros2 interface show <pkg/msg/Type>                # 看消息结构
ros2 run tf2_ros tf2_echo <parent> <child>        # 查变换
ros2 bag info <bag_dir>                           # 看录包
```

---

## 11. 监控时的安全提醒

- 本手册命令**默认只读**（echo / hz / list / info / 查询类 service call），不会让机器人动作。
- `joint_follow`、`set_l6_*` 是**订阅端话题**，用 `ros2 topic pub` 往里发消息会直接驱动机械臂/灵巧手，监控时**不要 pub**。
- `set_enable`、`set_emergency_stop`、`set_zero`、`move_*` 会改变机器人状态或产生运动，排查问题前确认机械臂周边安全，必要时先急停。
- 看到 `joint_states` 出现 **NaN**、频率掉到 0、或位姿跑到工作空间外，立即停止下发动作并检查驱动/控制器连接。

相关文档：《控制接口说明.md》《相机与视觉传感器说明.md》《手眼标定说明.md》《现场快速检查表.md》《常见问题FAQ.md》。
