# 队友现场采集操作手册：螺母 RGB-D 数据

更新日期：2026-09-10。依据当前 `rgbd_workbench.py`、`rgbd_core.py` 和 `replay_dataset.py` 的实际实现编写。

## 先看这一页

**你的任务是采集可用于离线开发的数据，不是调识别算法，也不是执行自主抓取。**

1. 确认机器人静止、相机连接正常，控制端不会根据视觉话题自动运动。
2. 启动 Orbbec 相机驱动，再启动 RGB-D 采集工作台。
3. 确认工作台显示 `LIVE`，左右分别有更新的彩色与深度画面。
4. ROI 使用 `0,0,1,1`，填写道具实测尺寸，保存配置。
5. 导出相机驱动参数；每换一种摆放或采集条件，点击“新场景”。
6. 先采 5～10 对并回传整组文件，确认格式后再批量补采。
7. 保留 `color.png`、`depth.npz`、`metadata.json` 和 `scene.json`，不能只传截图。

**现有识别会漏检、错分。没有识别标签、出现 `unknown` 或黄色轮廓，不意味着这帧不能采集。绿色也不是“可安全抓取”的证明。** 以彩色原图质量、文件完整性、同步状态为采集依据。

本轮没有训练完成的 YOLO 螺母模型。无需现场安装 YOLO、训练模型或修改检测阈值。

---

## 1. 双方怎么分工

| 现场队友 | 离线开发人员 |
|---|---|
| 启动设备，确认没有自动运动风险 | 提供采集代码和操作说明 |
| 摆放螺母、复核尺寸、记录条件 | 检查样本格式和时间同步 |
| 采集、分场景整理、回传 | 标注轮廓和类别、训练和评估 |
| 根据失败样例补拍 | 开发检测、大小分类和相机坐标定位 |

相机到机器人外参标定暂时不做。这里也不要求现场人员调整机械臂抓取策略。

## 2. 使用哪些代码

原电脑代码目录：`$HOME/projects/lbot_ws_team/tools/nut_vision`。

| 文件 | 功能 | 本轮是否需要 |
|---|---|---|
| `rgbd_workbench.py` | GUI 预览、同步图像配对、保存样本、单帧离线分析 | 必需 |
| `rgbd_core.py` | 工作台调用的原型检测与绘制代码 | 必需，即使只采集也会导入 |
| `vision_config.json` | 检测配置、全画面 ROI、道具实测尺寸 | 必需 |
| `replay_dataset.py` | 没有相机时回放一个已采场景 | 推荐一起带上 |
| `rgbd_core_candidate.py`、`rgbd_geometry.py` | 尚未通过评估的实验版本 | 不用于本轮现场采集 |
| `nut_detector.py` | 更早的独立检测脚本 | 本轮不用，不要额外启动 |

换电脑时，将前四个文件放进同一个可写目录，例如 `$HOME/nut_vision`。程序会在自身目录读取、更新 `vision_config.json`，不要只复制主脚本。

本工具通过团队仓库的视觉分支交付。不要复制其他独立仓库的 `.git`；大体积采集数据另传，不提交到 Git。

## 3. 设备与软件前提

当前使用的是 Orbbec Gemini 2。原电脑环境为 ROS 2 Jazzy，驱动工作区为 `$HOME/orbbec_ws`。

已有工作环境优先沿用，不要为了采集临时升级系统、相机固件或驱动。

程序用到：

- 系统 Python 3、OpenCV、NumPy、PyQt5。
- ROS 2 的 `rclpy`、`cv_bridge`、`message_filters`、`rosidl_runtime_py`。
- `sensor_msgs`、`geometry_msgs`、`std_msgs`、`tf2_msgs`。
- 已安装并构建好的 `orbbec_camera` 驱动。

优先使用 `/usr/bin/python3`，避免 Conda 或未配置 ROS 的虚拟环境导致模块找不到。

若是新电脑，应先完成 ROS 和相机驱动安装，再执行本手册。复制 Python 文件不会自动安装驱动。遇到依赖错误，把完整终端报错发回来，不要随意混装多个 Python 环境。

### 采集安全边界

- 机器人保持停止或现场团队认可的安全待机状态。
- 工作台不直接发运动命令，但会发布 `/nut_vision/detections`。必须确认机器人控制程序不会消费该话题后自动运动。
- 放置道具时不要进入可能运动的机械臂范围。需要改变机器人观察姿态时，由负责运动控制的队友按既有安全流程操作。
- 彩色画面可能拍到人员、电脑屏幕和其他队伍；尽量避开无关人员和敏感内容，数据只通过团队认可的渠道传输。

## 4. 启动相机：终端 A

如果相机驱动已经在运行，沿用现有进程，**不要重复启动**。可由现场负责人确认，或查看 `ros2 node list` 和 `ros2 topic list`。

在原电脑从头启动时使用：

```bash
source /opt/ros/jazzy/setup.bash
source $HOME/orbbec_ws/install/local_setup.bash

ros2 launch orbbec_camera gemini2.launch.py \
  color_width:=1280 color_height:=720 color_fps:=30 \
  depth_width:=640 depth_height:=400 depth_fps:=30 \
  enable_ir:=false enable_point_cloud:=false \
  enable_colored_point_cloud:=false \
  depth_registration:=true enable_frame_sync:=true
```

换电脑时，只把驱动工作区的路径改成实际位置。该命令针对当前 Gemini 2 配置，不适用于所有 Orbbec 型号。

注意：命令请求原生深度模式为 `640×400`，但历史现场运行中，深度注册后的两路输出均为 `1280×720`。**以实际输出及匹配的 CameraInfo 为准，不要自己缩放深度图来伪装对齐。** 如果不同驱动版本输出话题不同，需要先与开发人员对齐。

当前工作台订阅的是下面四个固定话题：

```text
/camera/color/image_raw
/camera/depth/image_raw
/camera/color/camera_info
/camera/depth/camera_info
```

相机、工作台终端必须使用相同 ROS 网络配置和 `ROS_DOMAIN_ID`；不要擅自改变团队正在使用的设置。

终端 A 保持运行。结束时在该终端按 `Ctrl+C` 正常停止，不要直接拔正在写数据的存储设备。

## 5. 启动工作台：终端 B

推荐显式指定保存目录，避免沿用原电脑写死的默认路径。

原电脑示例：

```bash
source /opt/ros/jazzy/setup.bash
source $HOME/orbbec_ws/install/local_setup.bash
export QT_QPA_PLATFORM=xcb
export QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1

export CAPTURE_OUT="$HOME/nut_capture/$(date +%Y%m%d_%H%M%S)_team"
mkdir -p "$CAPTURE_OUT"
printf '本批次保存目录：%s\n' "$CAPTURE_OUT"

/usr/bin/python3 $HOME/projects/lbot_ws_team/tools/nut_vision/rgbd_workbench.py \
  --output "$CAPTURE_OUT"
```

换电脑时，修改相机工作区路径，并把脚本路径换成实际位置，例如 `$HOME/nut_vision/rgbd_workbench.py`。

原电脑也有快捷脚本，但它内部包含原电脑绝对路径，不是可直接跨电脑使用的安装包：

```bash
bash $HOME/projects/lbot_ws_team/tools/nut_vision/start_capture.sh \
  --output "$HOME/nut_capture/team_batch01"
```

两种启动方法选一种即可。不要同时打开多个工作台进行采集。

## 6. 单独保存相机驱动参数：终端 C

**必须区分：`metadata.json` 中的 `parameters` 是视觉程序配置，不是完整相机驱动参数。**

工作台不会自动执行参数导出。它仅在创建场景时，检查输出根目录是否存在 `camera_parameters_720p.yaml`；存在时才复制为场景内的 `camera_driver_parameters.yaml`。

因此，在第一次点击保存之前：

1. 在新终端 source 与前面相同的 ROS 环境。
2. 执行 `ros2 node list`，找到实际相机节点，不是 `nut_vision_workbench`。
3. 将下面的保存目录改为终端 B 打印的本批次目录，将节点名改为刚找到的实际名称。

```bash
source /opt/ros/jazzy/setup.bash
source $HOME/orbbec_ws/install/local_setup.bash

CAPTURE_OUT="$HOME/nut_capture/请替换为本批次目录名"
CAMERA_NODE="/请替换为实际相机节点名"
ros2 param dump "$CAMERA_NODE" > "$CAPTURE_OUT/camera_parameters_720p.yaml"
```

确认命令没有报错，文件中确实有参数内容。不要直接照抄上述占位符。

这个固定文件名是当前程序的约定，不会改变相机分辨率。若现场调整曝光、增益、分辨率、深度对齐等设置，应先停止连采、重新导出参数，再创建新场景。

如果已经采了几帧才发现缺参数，不要把后来修改过的参数冒充当时配置。说明哪些场景缺失，下一场景开始补齐。

## 7. 工作台按钮：一步一步操作

### 7.1 先确认画面真的在更新

- 左侧是带原型检测标注的彩色预览；右侧是深度伪彩图。
- 状态栏应显示 `LIVE 1280x720` 或实际分辨率，以及同步时间差。
- 确认两路都有新画面，而不是静止的旧预览。
- `INVALID` 表示当前帧检查失败，不能采；`STALE` 表示近期没有新同步帧，也不能采。
- 深度预览黑色通常表示无效值，伪彩颜色不是原始深度数值。

彩色预览使用保持比例缩放。调整窗口大小不改变保存图像的分辨率；不要截取预览窗口来代替原始样本。

### 7.2 设置全画面和尺寸

1. 在 `ROI归一化 x0,y0,x1,y1` 输入框填写 `0,0,1,1`。
2. 点击“应用工作区ROI”。这表示全画面检测，不设置固定活动框。旧交接说明中的局部 ROI 已过时，不要照用。
3. 任务模式：基础三颗选 `basic`，包含大黑螺母的四颗场景可选 `advanced`。这主要影响原型识别逻辑；真实类别仍必须在场景说明中记录，不能依赖预测。
4. 在底部实测尺寸区域输入下表中的对边宽、厚度，单位都是 **mm**，然后点“保存实测尺寸”。

| 物体 | 对边宽/mm | 厚度（高度）/mm |
|---|---:|---:|
| 大银 `M45_silver` | 70 | 34 |
| 大黑 `M45_black` | 70 | 34，需现场确认是否与大银相同 |
| 中黑 `M33_black` | 50 | 26 |
| 小黑 `M27_black` | 40 | 22 |

这些是目前提供的测量值，现场有卡尺时请复核。对边宽是两条相对平行外侧面之间的距离，不是对角距离，也不是内孔直径。`M45/M33/M27` 为规格标签，不代表外部对边宽。

界面中的“未知”或数值 `0` 表示未测量。不要把算法显示的 `AF` 估计值抄成实测真值。

尺寸、ROI 和任务模式尽量在创建场景前设好。改变配置后创建新场景；当前程序并不会自动阻止所有配置在场景中途变化。

### 7.3 新建一个采集场景

1. 安全摆放道具，保证本次想拍的螺母清晰可见。
2. 把手移出画面，等待曝光和画面稳定。
3. 点击“新场景 / New scene”。
4. `数据用途` 选择 `train` 或 `validation`。
5. 类型选择 `nuts`；只有专门拍标定板时选 `calibration_board`。
6. 在说明输入框写清楚本场景条件，覆盖默认的“真实尺寸待提供”等旧文字。

建议直接采用这样的说明：

```text
scene=single_small_03; objects=M27_black; count=1; position=right_bottom;
rotation=changed; light=normal; background=white_paper; occlusion=none;
camera_pose=fixed_A; hand=removed; purpose=train
```

中文也可以：

```text
小黑单颗；画面右下，完整入画；已转动方向；正常顶灯；白纸背景；
无遮挡；手已移开；相机姿态A未变；训练用途。
```

“新场景”按钮只是准备新场景，**第一次成功保存时才真正创建文件夹**。

### 7.4 保存一对或十对

| 按钮 | 实际行为 |
|---|---|
| “采集1对 / Save pair” | 保存最近一组新鲜同步彩色、深度和元数据 |
| “采集10对 / Capture 10 pairs” | 自动保存10组，保存间隔至少约0.5秒，处理较慢时会更长 |
| “停止采集” | 取消后续自动保存，不删除已经保存的数据 |
| “新场景 / New scene” | 停止当前连采、准备下一个场景，不删除旧场景 |
| “保存当前界面截图 / Screenshot” | 只保存界面截图，不是训练所需的完整RGB-D样本 |

以底部实际“已保存”“剩余”和路径提示为准，不要仅看按钮已经点下。十对不保证恰好五秒完成。

同一个彩色时间戳只保存一次；连续点单张按钮可能不会产生新文件。这个工具不是30fps录像器。

**开始第一轮时先用“采集10对”完成小批试采，打包整个场景发给离线人员确认。**

### 7.5 换条件后再采

- 改变摆放、光照、相机姿态或数据用途：先停止连采，点击“新场景”，再填写说明。
- 同一场景中途修改 `train/validation`、类型或说明后直接保存，程序会拒绝并提示新建场景。
- 主训练样本不要一边用手拿着螺母移动，一边连拍；专门的遮挡/运动样本另建场景并注明。

## 8. 具体要采什么

### 8.1 采集优先级

| 优先级 | 场景 | 具体要求 |
|---|---|---|
| P0 | 四种物体分别单独出现 | 大银、大黑、中黑、小黑都要，不能只有三颗总是一起出现 |
| P0 | 基础三颗、进阶四颗组合 | 全部完整入画；分离、不叠放；改变相对位置和转角 |
| P0 | 两颗以及移走部分目标后的场景 | 改变类别组合，让模型不能靠“必须有三颗”判断 |
| P0 | 空场景和干扰物 | 没有目标螺母，保留机器人圆孔、线缆、标定板、反光等真实背景 |
| P1 | 画面位置变化 | 中心、左、右、上、下、靠近边缘但仍完整可见 |
| P1 | 平面内旋转 | 在桌面上旋转道具，覆盖不同六角朝向，不必精确量角 |
| P1 | 观察姿态和距离 | 安全条件下改变相机视角/距离，分别记录姿态编号；不能全部同一观察尺度 |
| P1 | 光照变化 | 正常、阴影、明显反光，不要只保留最漂亮的画面 |
| P2 | 遮挡、相互接触、部分出画 | 单独标记为困难样本，用于检测与拒识研究，不能冒充合规初始摆放 |
| P2 | 标定板专用画面 | 条件允许时补采，不要求现在做机器人外参标定 |

算法主场景是螺母平放在支撑面上。侧放、叠放等可以另采为异常样本，不与平放样本混写。

### 8.2 第一批建议数量

先做5～10对试采，再收集约30～40个有区别的场景，每场景5～10对，总量约150～400对。该数量只是起点，不保证准确率。

每种物体应有多个独立摆放；空场景至少覆盖几种真实干扰背景。真正重要的是场景变化，不是在同一摆法下连拍几百张。

合理顺序：先单颗和完整组合，再补光照、姿态，最后补困难样本。

### 8.3 图像质量要求

- 主样本中每颗螺母完整入画，周围留一点背景，不紧贴图像边缘。
- 放大原始彩色图时能辨认孔口和边缘，不严重失焦或拖影。
- 主体不要长期被手、夹爪或另一颗螺母挡住。
- 保持已确认的采集分辨率，不自行裁剪、缩放或叠加文字到原图。
- 黑色螺母深度可能存在空洞；如果RGB清楚，可以保留并记录问题，不要为了深度漂亮删除所有困难样本。
- 深度整片为空、两路画面明显不对应、时间同步持续失败时，应先排查设备，而不是继续批量采。
- 改曝光后记录参数，避免银色螺母长期过曝到外形消失。专门的强反光样本另外注明。

## 9. 训练、验证和标注

界面只有 `train`、`validation` 两个用途，没有 `test` 下拉项。

- 同一摆法的相邻帧属于同一组，不要一半放train、一半放validation。
- validation应来自重新摆放或另外的采集条件，尽量覆盖每种物体。
- 最终测试建议再留独立采集批次，由离线人员单独管理；仅选择validation不等于它永远不会被用于调参。
- 所有场景说明必须写真实物体身份及数量。不要照抄画面上的预测标签。
- `scene.json` 中的 `ground_truth_status` 初始是 `unannotated`；帧元数据中的 `ground_truth` 为 `null`。程序没有自动生成可靠训练标签。
- 现场不必手工勾全部轮廓，离线人员负责标注；但现场尺寸、摆放和遮挡说明非常重要。

建议每批另附 `batch_notes.txt`，记录采集人、日期、相机型号、批次目录、实际尺寸、参数变化、异常场景ID、哪些场景预留独立评估。

## 10. 数据到底保存在哪里、保存了什么

假设启动时传入：

```text
--output /home/某用户/nut_capture/batch01
```

数据结构是：

```text
batch01/
  camera_parameters_720p.yaml          # 手动导出到这里
  batch_notes.txt                     # 手动附加的批次说明
  workbench_screen.png                # 可选截图，多次点击会覆盖
  dataset/
    train/
      日期时间_随机后缀/
        scene.json
        camera_driver_parameters.yaml # 仅在参数源文件存在时复制
        秒_纳秒/
          color.png
          depth.npz
          metadata.json
    validation/
      日期时间_随机后缀/
        ...
```

帧目录直接位于场景目录下，**中间没有 `frames/` 这一层**。

| 文件 | 内容及注意事项 |
|---|---|
| `color.png` | 保存的原始彩色帧，没有预览界面的检测线；PNG无损，OpenCV读取为BGR |
| `depth.npz` | 原始数值深度的无损压缩，数组键是 `depth`；不是深度伪彩截图 |
| `metadata.json` | 两路原始header、CameraInfo、编码、深度缩放、同步差、视觉配置、哈希、原型预测等 |
| `scene.json` | 用途、场景说明、类型、创建时间和实测尺寸快照 |
| `camera_driver_parameters.yaml` | 建场景时复制的驱动参数；缺失时须说明，不是必然自动出现 |

程序当前按 `16UC1` 为毫米、`32FC1` 为米处理，并写入 `depth_scale_m`。这是本项目针对当前驱动的约定；更换设备/驱动时必须确认真实单位，不能仅凭编码名称推断所有相机的深度单位。

同步采用近似配对，允许差值不超过配置的20ms，**不等于硬件严格同时曝光**。两路时间戳均保留。

程序还会保存收到的静态TF，以及发现的 `/robot*` 路径下部分 `PoseStamped/JointState` 消息。这些机器人观测只是最近收到的值，没有严格同步，也可能根本没有；不构成合格手眼标定数据。

## 11. 标定板怎么额外采（可选）

如果现场已有尺寸已知的板：

1. 类型选择 `calibration_board`，点击新场景。
2. 在说明里记录板类型、格子尺寸、内角点行列数；ArUco/ChArUco/AprilTag还需字典或家族、ID及标记实际边长等信息。
3. 在不同画面位置、距离、倾角下采集，图案清晰可见，尽量覆盖画面各区域。
4. 不要仅拍螺母场景中远处一小块模糊标定板，作为唯一标定数据。

当前GUI不会替你填写 `calibration_board_specification` 结构化字段，务必写进场景说明和批次说明。不要假设选择这个类型后就已经完成标定。

本轮不安排机器人位姿配对采集。若将来做手眼标定，需另行约定机器人位姿、坐标系及时间配对流程。

## 12. 试采后怎么检查

现场先做人工检查：

1. 底部显示保存成功，输出路径正确。
2. 打开一张 `color.png`，确认不是仅有标注的界面截图，螺母清晰完整。
3. 每个帧目录有三个文件：`color.png`、`depth.npz`、`metadata.json`。
4. 场景说明、实测尺寸和驱动参数没有漏填。
5. 先把这一小组发回来，离线确认后再继续大批量采。

如现场会使用终端，可以用下面的单帧检查脚本。它只读数据，不会控制机器人，也不会修改采集文件。将帧目录替换为实际路径：

```bash
export FRAME_DIR="/请替换为实际路径/场景ID/秒_纳秒"
/usr/bin/python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np

p = Path(os.environ['FRAME_DIR'])
meta = json.loads((p/'metadata.json').read_text())
color = cv2.imread(str(p/'color.png'))
with np.load(p/'depth.npz', allow_pickle=False) as packed:
    depth = packed['depth']
assert color is not None, 'Color PNG cannot be decoded'
assert color.shape[:2] == depth.shape, 'Color/depth shape mismatch'
assert hashlib.sha256(color.tobytes()).hexdigest() == meta['color_sha256'], 'Color hash mismatch'
assert hashlib.sha256(depth.tobytes()).hexdigest() == meta['depth_sha256'], 'Depth hash mismatch'
assert meta['sync_delta_ms'] <= 20, 'Time delta too large'
for name in ['color', 'depth']:
    info = meta[name+'_camera_info']
    assert (info['height'], info['width']) == depth.shape, 'CameraInfo size mismatch'
    assert info['header']['frame_id'] == meta[name+'_header']['frame_id'], 'Frame ID mismatch'
assert meta['color_header']['frame_id'] == meta['depth_header']['frame_id'], 'Streams not in same frame'
assert np.allclose(meta['color_camera_info']['k'], meta['depth_camera_info']['k'], rtol=1e-4, atol=1e-4), 'Intrinsics mismatch'
meters = depth.astype(float)*meta['depth_scale_m']
valid = np.isfinite(meters) & (meters > .15) & (meters < 4.)
print('OK: shapes, hashes, timestamps and CameraInfo checks passed')
print('color:', color.shape, 'depth:', depth.shape, depth.dtype)
print('encoding:', meta['depth_encoding'], 'scale:', meta['depth_scale_m'])
print('sync_delta_ms:', meta['sync_delta_ms'])
print('depth fraction in 0.15..4m:', float(valid.mean()))
PY
```

通过上述文件检查，只能说明格式等条件基本一致，不能证明真实光学对齐正确或螺母表面深度准确。有效深度比例是全图统计，不是每颗螺母的深度质量。

出现失败时保留原文件和报错，暂缓批量采集，发给离线人员定位；不要修改数值或元数据让检查“通过”。

## 13. 常见问题

| 现象 | 该怎么做 |
|---|---|
| `No module named rclpy/cv_bridge/...` | 检查是否source了ROS、是否使用系统Python；把完整报错回传 |
| 等待相机，没有画面 | 确认驱动在运行、四个话题存在、ROS_DOMAIN_ID一致；不要重复启动多个驱动 |
| `Color/depth dimensions differ` | 检查深度注册及驱动输出；不要手动缩放原始深度来绕过检查 |
| `Color and depth are not in the same optical frame` | 检查真实对齐话题及驱动配置；不要只改frame_id字符串 |
| `CameraInfo timestamp is stale` | 检查是否混入历史回放、驱动/消息队列是否异常，记录日志 |
| `Aligned depth and color intrinsics differ` | 检查驱动是否提供与对齐图像匹配的CameraInfo，不能复制一份内参冒充 |
| `STALE` | 当前预览可能已过期。停止连采，检查驱动终端，必要时由现场负责人正常重启相机驱动 |
| 能看到螺母但没有识别标签 | 仍可采集，模型漏检正是有价值的样本；前提是同步状态正常 |
| 深度黑洞或反光噪声 | 保留RGB清楚的样本并注明，另补较好视角；不要填造深度 |
| 同一场景不能更改说明 | 点“新场景”后再保存 |
| 点保存后数量不增加 | 检查是否新鲜帧、是否重复时间戳、是否写权限问题，查看底部和终端 |
| 磁盘空间不足 | 程序在剩余空间低于1GiB时停止；事先留出数GiB以上空间，优先迁移已确认回传的数据 |
| 场景没有驱动参数文件 | 按第6节导出参数，并从下一个新场景开始采；说明前面缺失的场景 |
| 截图只有一个文件 | 截图按钮覆盖 `workbench_screen.png`，不是采集按钮；使用Save pair或Capture 10 pairs |
| 保存失败留下不完整帧目录 | 记录场景/帧ID和报错，另采新帧；不要将缺文件的目录算作成功样本 |

退出工作台不会停止独立相机驱动。不要用全局 `killall` 等方式误停其他队友进程。

## 14. 怎样打包回传

只打包已经停止写入的本批次目录，保留场景结构，不要重命名或单独分散传输每个PNG。

例如目录为 `$HOME/nut_capture/batch01`：

```bash
tar -czf "$HOME/nut_capture_batch01.tar.gz" \
  -C "$HOME/nut_capture" batch01
sha256sum "$HOME/nut_capture_batch01.tar.gz" \
  > "$HOME/nut_capture_batch01.tar.gz.sha256"
```

把两个文件通过团队认可的共享存储发送，同时附上以下信息：

```text
采集人：
采集日期与场地：
相机型号／驱动版本（知道则填写）：
代码版本或收到的代码包名称：
相机参数是否随包：
螺母尺寸是否复核，大银与大黑是否相同：
本批次包含哪些场景／哪些是validation：
异常场景或缺文件帧：
是否有刻意遮挡、出画、运动样本：
哪些场景预留独立测试：
```

不要只用聊天软件“发图片”，避免压缩和元数据丢失。不要把原型预测当成标签一起宣称已经标注。

## 15. 无设备时怎样回放（供离线人员）

回放会发布与相机相同的话题。**不要与真实相机或机器人控制端混在同一个ROS域里运行。**

以下使用示例隔离域77；先确认团队没有在该域运行硬件控制。两个回放相关终端使用相同域。

终端一：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=77
/usr/bin/python3 $HOME/projects/lbot_ws_team/tools/nut_vision/replay_dataset.py \
  /实际路径/场景ID --rate 0.5 --loop
```

终端二：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=77
export QT_QPA_PLATFORM=xcb
/usr/bin/python3 $HOME/projects/lbot_ws_team/tools/nut_vision/rgbd_workbench.py \
  --output "$HOME/nut_replay_preview"
```

使用场景目录，不是帧目录。回放保留原始时间戳，不发布 `/clock`，不重放机器人位姿或TF。回放画面不是现场实时观测。不要把回放中再次点击保存产生的数据混入新的独立采集样本。

## 16. 本手册与交付边界

- 本手册核对的是当前源码中的接口与行为；没有在队友的电脑上完成安装、驱动启动或硬件试采验证。
- 本轮只编写操作文档，没有修改采集代码，没有启动相机或机器人，也没有运行本文检查命令。
- 原识别、候选版、几何版均不是已经完成比赛验收的系统。
- 建议先完成小批样本交接，确认可读且质量合适，再进行完整采集。
