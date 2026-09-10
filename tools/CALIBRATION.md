# 手眼标定操作手册（Eye-to-Hand）

本手册配套 `tools/` 下的标定脚本，用于求 **Gemini2 相机光学帧 → 机器人 base 帧** 的外参，
把相机看到的目标点转换到机械臂坐标系，供 `move_pose / move_linear` 抓取使用。

- 标定方式：**Eye-to-Hand**（相机固定看工作区，标定板由右臂夹持）。
- 使用机械臂：**右臂** `/robot1/right_arm`。
- 标定板：**ChArUco**（脚本生成、自行打印）。
- 全部脚本**只读订阅、不发运动指令**；机械臂运动由人工/遥操作完成。

## 脚本与产物

| 脚本 | 作用 |
| --- | --- |
| `calib_charuco_board.py` | 生成可打印 ChArUco 板 PNG + 板配置 YAML |
| `calib_handeye_sample.py` | 采样：同步记录右臂末端位姿 + 板在相机中位姿 |
| `calib_handeye_solve.py` | 手眼标定求解，输出外参/内参/末端→板 YAML，报告残差 |
| `calib_handeye_publish_tf.py` | 读外参发布静态 TF（base_link→相机光学帧） |
| `calib_handeye_verify.py` | 在线/离线核验外参（视觉路径 vs 运动学路径） |

产物目录 `开发资源/calibration/`：

```text
charuco_board.png / charuco_board.yaml     # 标定板与板尺寸（需实测回填）
gemini2_extrinsics.yaml                    # 外参 base_link -> camera_color_optical_frame ★
gemini2_color_camera_info.yaml             # 彩色相机内参（采样时自动保存）
gemini2_gripper_to_board.yaml              # 末端->标定板常量（核验用）
```

采样原始数据在 `recordings/calib_<时间戳>/samples.jsonl`。

---

## 第 0 步：安装并启动 Orbbec 相机驱动（仅首次）

ROS2 驱动源码已在 `~/Project/1_Competition/THUEI/OrbbecSDK_ROS2`（v2-main），但它依赖独立的
**OrbbecSDK（libobsensor）** 二进制库，需先安装，再 colcon 编译。

```bash
# 0.1 系统依赖
sudo apt install libgflags-dev nlohmann-json3-dev \
  ros-jazzy-image-transport ros-jazzy-image-transport-plugins \
  ros-jazzy-compressed-image-transport ros-jazzy-camera-info-manager \
  ros-jazzy-diagnostic-updater ros-jazzy-diagnostic-msgs ros-jazzy-statistics-msgs \
  ros-jazzy-backward-ros libdw-dev libssl-dev libgl1 libgoogle-glog-dev

# 0.2 安装独立 OrbbecSDK（libobsensor）：从 OrbbecSDK 发布页下载 Linux x64 压缩包，
#     解压后运行其中的 install.sh（会装到 /usr/local/lib、/usr/local/include 并装 udev）。
#     发布页：https://github.com/orbbec/OrbbecSDK/releases
sudo ldconfig

# 0.3 编译 ROS2 驱动（源码已在本地，可直接用；或放到 colcon workspace 的 src 下）
mkdir -p ~/orbbec_ws/src
ln -sfn ~/Project/1_Competition/THUEI/OrbbecSDK_ROS2 ~/orbbec_ws/src/OrbbecSDK_ROS2
cd ~/orbbec_ws
source /opt/ros/jazzy/setup.bash
colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# 0.4 udev 规则（否则普通用户打不开相机）
cd ~/Project/1_Competition/THUEI/OrbbecSDK_ROS2/orbbec_camera/scripts
sudo bash install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

启动相机（彩色+深度+对齐）：

```bash
source ~/orbbec_ws/install/setup.bash
ros2 launch orbbec_camera gemini2.launch.py \
  depth_registration:=true enable_color:=true enable_depth:=true \
  enable_point_cloud:=true publish_tf:=false
```

> 这里先用 `publish_tf:=false`：相机内部 TF 交给我们标定得到的静态外参统一发布，
> 避免 `camera_color_optical_frame` 出现两个父级。确认话题：
> `ros2 topic list | grep camera`，应能看到 `/camera/color/image_raw`、`/camera/color/camera_info`。

每个脚本都依赖 ROS2 环境，运行前先 `source /opt/ros/jazzy/setup.bash`（以及相机/机器人的 setup）。

---

## 第 1 步：生成并打印标定板

```bash
python3 tools/calib_charuco_board.py
# 可选：--square-mm 30 --marker-mm 22 --squares-x 8 --squares-y 6 --dpi 300
```

- 打印 `开发资源/calibration/charuco_board.png`：**等比例打印，不要“适应页面”缩放**。
- 板贴在平整硬板上（不能卷曲、反光）。
- **用尺实测棋盘方格边长**，回填 `charuco_board.yaml` 的 `square_length_m`，并按
  marker/方格 比例核对 `marker_length_m`（默认 22/30≈0.73）。
- 尺寸不准会直接导致外参尺度错误，务必实测。

## 第 2 步：夹持标定板并采样

1. 把标定板**牢固**装到右臂末端（灵巧手抓稳或加装支架），全程不能松动、不能相对末端移动。
2. 机器人使能、低速；人工/遥操作带动右臂。
3. 启动采样节点：

```bash
python3 tools/calib_handeye_sample.py --arm right --save-images
```

4. 弹出预览窗口，绿色 `READY` 时按 **空格/回车** 记录一个样本，`q` 结束。
   脚本在板检测成功、内参到位、且**机械臂停稳**（0.5s 窗口内抖动 <1mm / <0.3°，已按反馈噪声放宽；
   可用 `--still-pos`/`--still-rot` 调整）时才允许记录。

**采样要点（直接影响精度）：**
- 数量 **15~25 个**姿态，越多越稳；至少 10 个。
- 板要在相机视野内**铺满不同位置**（左/右/上/下/近/远）。
- **多角度倾斜**标定板（绕各轴转动），不要所有姿态板都正对相机——否则旋转不可观。
- 每个姿态**先停稳再记录**；板完全可见、无遮挡、避免强反光。
- 移动时人与工件保持安全距离，低速。

## 第 2b 步：半自动采样（机械臂自动走位，推荐）

手动拖动手感不稳、姿态重复性差时，改用上使能 + `move_pose` 自动走位：机械臂依次移动到
一组**已验证可达、且相机能看到板**的目标点，到位停稳后按回车记录、再自动去下一帧。

**右臂可达 / 相机可见工作区（base_link，单位 m，来自实测优质样本）：**

| 轴 | 推荐采样区（板被高质量看到） | 备注 |
|----|------------------------------|------|
| X  | 0.15 ~ 0.40 | 机械臂正前方 |
| Y  | -0.22 ~ 0.10 | 右臂在负 Y 侧工作 |
| Z  | -0.42 ~ 0.01 | 竖直方向 |

> 运动学可达范围比上表更大，但超出该区域板容易出视野或角点不足。目标点即从该区域的
> 真实到达位姿中提取，既可达又可见。欧拉角为 intrinsic XYZ（弧度），已随位置一起给出。

```bash
# 1) 生成目标点（已生成 开发资源/calibration/auto_waypoints.yaml，取最新会话全部 30 个可达位姿）
#    --all-sessions 会话=全部到达过的位姿都纳入；其中板没看全的点标 expected: maybe，
#    自动采样到位后若识别不到会超时跳过，不会卡住。
python3 tools/calib_gen_waypoints.py \
    --all-sessions recordings/calib_20260908_214053_781626 --num 80
#   若想合并多批：位置参数=只取板完整可见的高质量点；--all-sessions=全部可达位姿都纳入

# 2) 半自动采样（会真实驱动机械臂！清空活动范围、手持急停、板刚性固定在法兰）
python3 tools/calib_handeye_auto.py --arm right --save-images
#   更稳可再降速：--speed 0.3 --acce 0.3 ；跳过逆解预检：--no-ik
#   到位后等就绪的超时：--ready-timeout 12（秒）
```

流程：脚本上使能 → 逐点 `move_pose`（block，低速）→ 等待停稳+板检测。默认**就绪后自动连采**：
检到板且停稳并持续 `--auto-delay`（默认 0.6s）即**自动记录并去下一个位姿**，全程无需按键。
交互：**`空格/回车`=直接记录并立刻去下一个**（当前帧有板就强制采、无视停稳门；没板则直接跳过
到下一个，到位后再按，运动中按会录到运动位姿）；`s`=跳过本帧；`q`=结束；到位后 `--ready-timeout`
（默认 12s）仍识别不到足够格点/没停稳，窗口显示橙色 **TIMEOUT**，此时**空格/s 跳到下一个位姿**，
`r`=再等一个周期——不会死等卡住。想回到“完全手动、不自动采”，加 `--auto-delay 0`。
写出与手动完全相同的 `samples.jsonl`（目录前缀 `calib_auto_`），求解命令不变。

⚠️ 自动走位只提升**重复性**，**不能**替代刚性安装：板若仍用手/胶带软固定，外参差的
根因（末端→板平移发散）依旧存在，残差仍会在 ~10mm 量级。务必把板刚性装到法兰/夹爪。

## 第 3 步：求解外参

```bash
python3 tools/calib_handeye_solve.py recordings/calib_<时间戳>
```

- 自动对 TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS 五种方法求解，按闭环残差择优。
- 写出 `gemini2_extrinsics.yaml`、`gemini2_color_camera_info.yaml`、`gemini2_gripper_to_board.yaml`。
- **残差判据**：闭环重投影误差均值建议 ≤2~3mm；末端→板平移 std 越小越好。
  - 偏大常见原因：板尺寸没按实测回填、采样姿态不够多样/未停稳、夹持松动、板被遮挡。

无相机时可先验证数学管线：`python3 tools/calib_handeye_solve.py --selftest`（合成数据，
噪声下平移 <1mm、旋转 <0.5° 即通过）。

## 第 4 步：发布静态 TF

```bash
python3 tools/calib_handeye_publish_tf.py
```

保持进程运行即常驻 `base_link → camera_color_optical_frame`。脚本会同时打印等价的
`ros2 run tf2_ros static_transform_publisher ...` 命令备查。

核验 TF：

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
ros2 run tf2_tools view_frames    # 生成 frames.pdf 查看坐标树
```

## 第 5 步：核验外参

```bash
# 在线：实时检测板，分别用“外参(视觉)”和“机械臂位姿×末端→板(运动学)”估计板原点 base 坐标并比对
python3 tools/calib_handeye_verify.py --arm right

# 离线：对采样目录逐样本比对
python3 tools/calib_handeye_verify.py --session recordings/calib_<时间戳>
```

窗口中 `vis`（视觉路径）与 `kin`（运动学路径）两条 base 坐标的差值稳定在**数毫米内**即合格。
也可做物理核验：让臂带动板到一个已知位置，比较转换后的 base 坐标与实测。

---

## 与抓取对接

完整链路：像素 `(u,v)` + **对齐深度** → 反投影成相机系 3D 点 → 外参 `B_T_C` 变到
`base_link` → 加安全高度 → `move_pose` / `move_linear`。

### 画面点选工具 `camera_pick_move.py`（会真实驱动机械臂）

相机驱动装在独立工作区 `~/orbbec_ws`，点选工具既要相机话题又要机械臂服务，
终端里 ROS + 相机 + 机器人三个 setup 都要 source（否则报 `Package 'orbbec_camera' not found`）。

**执行命令（复制即用，共 3 个终端）：**

```bash
# 终端1：相机（深度对齐彩色）。只需 ROS + orbbec_ws
source /opt/ros/jazzy/setup.bash
source ~/orbbec_ws/install/setup.bash
ros2 launch orbbec_camera gemini2.launch.py depth_registration:=true

# 终端0（机器人侧/已启动）：机器人驱动必须在跑，提供
#   /robot1/right_arm/{pose_states,move_pose,move_linear,set_enable,inverse_kinematics}

# 终端2：点选工具。三个 setup 全 source，在 lbot_ws 根目录执行
cd /home/lionheart/Project/1_Competition/THUEI/Build_ws/lbot_ws
source /opt/ros/jazzy/setup.bash
source ~/orbbec_ws/install/setup.bash
source /home/lionheart/Project/1_Competition/THUEI/lbot_ws/install/setup.bash
python3 tools/camera_pick_move.py
#   三连窗出现后：左键点目标 -> 看绿字 IK OK -> 回车运动到目标正上方 5cm
#   红 IK UNREACHABLE = 该点不可达，换个点；s=清除；q=退出
```

可选：另开一个同样 source 三个 setup 的终端发布/核验 TF（点选工具本身直接读外参 yaml，不依赖此步）：

```bash
python3 tools/calib_handeye_publish_tf.py                 # 常驻发布 base_link->相机光学帧
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame   # 另开终端核验数值
```

**参数与其它用法：**

```bash
python3 tools/camera_pick_move.py                 # 打开画面，左键点目标
#   左键=选点（用对齐深度反投影，显示该点 cam/base 坐标 + 逆解可达性）；回车=移动到目标正上方(默认抬5cm)
#   s=清除选点；q=退出；--lift 0.03 改预抓高度；--approach 到位后再 move_linear 直线贴近
#   选点后自动调 inverse_kinematics 预检：绿色 IK OK 才能回车运动，红色 IK UNREACHABLE 直接不动
#   --keep-down：全程保持工具朝下（先原地转正，再 MoveL 直线平移，路径上姿态不变）
python3 tools/camera_pick_move.py --euler-deg 0 -90 0           # 指定姿态（默认工具朝下+自动搜yaw）

# 纯坐标模式：相机系 xyz(米, 光学系 Z前/Y下/X右) -> base 换算，仅打印；--go 才动（同样先过逆解）
python3 tools/camera_pick_move.py --cam-point 0.0 -0.02 0.55 --go
```

- 界面为横向三联：**彩色（仅此面板可点选）｜伪彩对齐深度｜反投影 3D 点云**（3/4 视角，
  选中点在深度图和点云里黄十字高亮，点云自动以选中点为中心）。
- 深度话题默认 `/camera/depth/image_raw`（`depth_registration:=true` 时与彩色对齐，用彩色
  camera_info 的 K 反投影；分辨率不一致会自动最近邻缩放）；点到无深度处会提示，邻域取中位深度。
- **内参默认直接读标定产物 `开发资源/calibration/gemini2_color_camera_info.yaml`**（最新一次
  采集求解时由 `/camera/color/camera_info` 保存），启动即有、不再等内参话题；顶栏显示
  `K:FILE gemini2_color_camera_info.yaml`。找不到该文件才回退实时话题（显示 `K:TOPIC`）；
  可用 `--camera-info 路径` 指定别的内参文件。改分辨率/重新标定后重跑
  `calib_handeye_solve.py` 即会刷新该文件。
- 默认末端姿态为**工具竖直向下**（intrinsic XYZ 欧拉 `(0,-90,yaw)`，厂商右臂抓取示例朝向），
  自动按 `0/-90/90/...` 顺序绕竖直轴搜索第一个逆解成功的腕部 yaw；`--euler-deg` 可锁定姿态。
- 选点即调 `/robot1/right_arm/inverse_kinematics` 做**可达性预检**（`--approach` 时预抓点和
  贴近点都要过），画面显示 `IK OK`（绿，回车才会动）或 `IK UNREACHABLE`（红，拒绝运动）。
- **不设任何人为坐标范围框（bound box）**：可达性唯一判据是驱动实时逆解，任何坐标都不会被
  脚本提前挡掉；只要逆解成功就允许运动。逆解默认以 `/robot1/right_arm/joint_states` 当前
  7 关节为种子，解更贴近当前臂型。默认只走到目标正上方 5cm；外参差 ~10.6mm（2026-09-09
  起用最新自动采集会话 calib_auto_20260909_101608；旧 9.3mm 最优已备份为
  `gemini2_extrinsics.yaml.bak_214053_9p3mm`，回退即覆盖回去），视觉点勿直接扎，
  先预抓。
- **`--keep-down`（全程工具朝下）**：默认模式只保证【到位时】手朝下（MoveJP 去程中手腕可能
  转动）。加该开关后分三段：① 若当前姿态不是朝下（差 >8°），先在**当前位置原地 MoveL 转正**
  （末端位置不动）；② MoveL **直线平移**到目标正上方，直线运动姿态恒定，手全程朝下；
  ③ `--approach` 再直线下降。预检会额外要求“当前位置 + 朝下姿态”逆解可达，否则判
  `IK UNREACHABLE`；MoveL 速度自动限制到 ≤0.3。顶栏显示 `KEEP-DOWN`。

> 驱动终端在 yaw 搜索时对每个失败候选打一条
> `Right arm inverse kinematics calculation failed:` —— 属正常现象（搜索试探），
> 只要点选窗口最终显示绿字 `IK OK` 即可运动；8 个候选全失败才会判 `IK UNREACHABLE`。
> 该日志表示该“位置+姿态”驱动求不出逆解（运动学上够不到或姿态无解），不是程序出错。

### 在自己的抓取节点里集成

1. 视觉算法输出相机坐标系目标点；
2. 用外参 yaml（`R @ p_cam + t`）或查 TF `base_link ← camera_color_optical_frame` 变到 base；
3. 加安全高度得到预抓点，规划 预抓→抓取→撤离；
4. 调 `/robot1/right_arm/move_linear`（MoveL）或 `move_pose`（MoveJP）；
5. 用 `/robot1/right_hand/set_l6_joint` 控制灵巧手抓/放。

## 坐标帧说明

- 机械臂 `pose_states` 的 `frame_id` 实测为 **`base_link`**；URDF 整机基座名为 `base_torso_root`。
  外参默认父帧用 `base_link`（与位姿数据一致）；若你的 TF 树以 `base_torso_root` 为根，
  求解时加 `--parent-frame base_torso_root`，或自行补发 `base_torso_root→base_link`。
- 相机光学帧 `camera_color_optical_frame`（Z 轴朝前、RH 光学坐标），与 Orbbec 驱动一致。

## 何时需要重新标定

- 相机/支架被移动或磕碰；
- 更换分辨率、深度对齐模式、相机固件/驱动参数（内参也需重新保存）；
- 更换夹持方式或标定板（末端→板变换变化）。

## 常见问题

| 现象 | 排查 |
| --- | --- |
| 打不开相机 | udev 未装/USB 权限；重插、`sudo bash install_udev_rules.sh` |
| 采样窗口“未收到内参” | 相机未启动或话题名不符；`ros2 topic echo /camera/color/camera_info --once` |
| 一直“未检测到 ArUco 码” | 板不在视野/太小/反光；调近、改善光照；确认字典一致 |
| 残差偏大 | 板尺寸实测回填、增加倾斜姿态、停稳再采、紧固夹持 |
| TF 光学帧两个父级 | 驱动 `publish_tf:=true` 与外参静态 TF 冲突，二选一 |
