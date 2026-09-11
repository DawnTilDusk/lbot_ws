# 螺母抓放任务框架（双臂交接版 nut_pick_place）

桌面太大，一颗螺母由两只臂协作入盒：

```
左臂视觉抓螺母 → 预录轨迹放到桌面中央 → 右臂预录轨迹到中央重抓 → 预录轨迹放进右侧盒子
```

视觉只负责**左臂抓螺母这一下**；之后所有固定运动（离场、转运、中央重抓、入盒）都回放
用 `record_workpoints.py` 预录的关节序列。框架默认 dry-run，`--execute` 才动真机。

## 采集 YOLO 标注照片

`capture_nut_images.py` 只订阅彩色图，不依赖标定板、不调用机械臂服务。
先启动相机驱动；远程订阅时使用与机器人一致的 ROS_DOMAIN_ID 和可互通的 ROS 网络。
在要保存照片的电脑上运行（采图用 ROS 系统 Python，训练另用 Conda）：

```bash
cd /home/dawntildusk/lbot_ws
source /opt/ros/jazzy/setup.zsh     # 当前终端为 zsh；bash 终端改用 setup.bash
ros2 topic list                    # 确认有 /camera/color/image_raw
/usr/bin/python3 tools/capture_nut_images.py
```

预览窗口中：空格或 `s` 拍一张，`a` 开启/暂停每 3 秒采图，`q`/ESC 退出。
按 Ctrl+C 也可退出。定时采图启动即开启的用法：

```bash
/usr/bin/python3 tools/capture_nut_images.py --interval 3 --count 100
# SSH 无图形界面：
/usr/bin/python3 tools/capture_nut_images.py --interval 3 --count 100 --no-preview
```

默认保存到 `/home/dawntildusk/nut_vision/raw/<本次时间戳>/images/*.png`，
同一会话的 `frames.jsonl` 记录源话题、消息时间戳和分辨率。
上传 `images` 中的 PNG 到 CVAT 即可；图片保持原分辨率、不含预览文字。
`--output` 可修改保存根目录，`--color-topic` 可修改彩色话题。
断流超过 1 秒时暂停保存；同一接收帧或非零消息时间戳不会重复保存。
保持相机安装与实际抓取一致，变换螺母摆放后等手离开画面再拍；自动模式可按 `a` 暂停后摆放。
本脚本仅保存训练所需的彩色照片，不保存深度或生成标注。

## YOLO 中心定位与主业务接入

### 实时视频窗口

```zsh
cd /home/dawntildusk/lbot_ws
source /opt/ros/jazzy/setup.zsh
/usr/bin/python3 tools/nut_yolo_live.py --device 0
```

需要已启动彩色、对齐深度和彩色 camera_info。模型在独立 Python 子进程（venv/conda，
见下文解释器说明）中常驻，
后台推理只处理最新配对帧；窗口显示检测帧本身，不把旧框叠到新画面。
左侧标注大/中/小和中心十字，右侧显示置信度、像素中心、相机 XYZ 与基座 XYZ（米）。
深度无效的目标仍显示框，但不显示虚构坐标；断流或检测帧过期则隐藏旧坐标。
标题区域显示推理更新速率和检测帧龄。`q`/ESC 或关闭窗口退出，`s` 保存当前结果；
正常退出也保存最后一次检测快照，位于 `recordings/yolo_live/<时间戳>/`。
JSON 含彩色/深度消息时间戳，快照不应当作实时运动指令。

`--device cpu` 使用 CPU，`--conf 0.6` 调整置信度，`--scale 0.6` 缩小窗口。
中文依赖系统 Pillow 与 Noto CJK 字体，`--font` 可指定其它中文字库。
无图形界面测试：`--no-preview --duration 15`。
该脚本没有运动客户端；它复用中心邻域深度策略，仍需注意螺母孔可能测到桌面。

工作区模型为 `weights/nut_best.pt`，来源和 SHA256 记录在 `weights/nut_best.json`。
它是 61 张照片追加训练后的选定权重，独立照片测试仍有漏检/误检。
ROS 系统 Python 负责相机与运动接口，`nut_yolo_infer.py` 子进程用**独立解释器**执行
YOLO（子进程环境剔除 PYTHONPATH/PYTHONHOME，与 ROS 完全隔离），解释器路径由
`detector.python` 指定，两种已验证的装法：

- **本机（lionheart，无 conda）**：工作区内 venv `.venv-yolo`（yaml 现指向它）。
  Ubuntu 缺 `python3.12-venv` 且 PEP668 禁止 pip 装系统时的建法：
  `python3 -m venv --without-pip .venv-yolo` → `curl get-pip.py | .venv-yolo/bin/python`
  → `.venv-yolo/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  → `.venv-yolo/bin/pip install ultralytics`（CPU torch 2.14 + ultralytics 8.4.146 已验证，
  对同一张快照的输出与 dawntildusk 完全一致）。
- **dawntildusk**：Conda `nut-yolo` 环境（`~/miniconda3/envs/nut-yolo/bin/python`）。

相机驱动先开启彩色和对齐深度（`depth_registration:=true`），再运行只读预览：

```zsh
cd /home/dawntildusk/lbot_ws
source /opt/ros/jazzy/setup.zsh
/usr/bin/python3 tools/nut_yolo_preview.py
```

该命令采集一次快照，输出每颗的 `label/confidence/u/v/z/p_cam/p_base`。
XYZ 均为米，`p_cam` 属于彩色光学坐标系，`p_base` 属于 `base_link`。
预览结果写入 `recordings/yolo_preview/<时间戳>/`：`centers.jpg` 有中心十字，
`detections.json` 有完整坐标，另存原始彩色图、深度 NPY 和内参便于复现。
它只订阅相机，没有机械臂客户端。

离线照片可先只求像素中心；没有对应深度时不会输出虚构 XYZ：

```zsh
/usr/bin/python3 tools/nut_yolo_preview.py --image /path/to/photo.png
# 使用预览保存的同帧数据重放完整坐标链路：
/usr/bin/python3 tools/nut_yolo_preview.py --image /path/to/color.png \
  --depth /path/to/depth.npy --camera-info /path/to/camera_info.yaml
```

主业务选择 YOLO（不加 execute 时获取实时坐标并打印计划，不运动）：

```zsh
/usr/bin/python3 tools/nut_pick_place.py --detector yolo
# 确认实物坐标、抓取高度、交接点后，实际运行（识别弹窗随 show_window 自动弹出）：
source install/setup.zsh
/usr/bin/python3 tools/nut_pick_place.py --detector yolo --execute
# yaml 里关掉弹窗时可临时强制开窗：... --detector yolo --show
```

执行路径：YOLO 框中心 -> 配对深度邻域中位数 -> 内参反投影 `p_cam` ->
外参换算到 base_link 检测点 -> 加 `grasp_offset_xyz`（默认向机体退 15cm）得腕部目标 ->
`run_one_nut` 左臂 hover/down -> 既有双臂序列（dry-run 打印、IK 预检、实机共用同一换算）。
缺类由业务校验中止（require_all）；**单轮漏检不再立刻中止**（2026-09-11 起）：整轮识别后
若缺型号，等 `detector.missing_retry_seconds`（默认 1s）重新拍快照整体重识别，最多
`detector.detect_attempts`（默认 3）轮都缺才中止，每轮打印缺哪颗/第几轮；require_all=false
时缺料本就允许跳过，不重试。同型号多颗的重复目标，按 yaml
`detector.duplicate_policy` 处理：`random`（当前配置）随机抓同型号
候选项里的一颗并打印选了第几颗/坐标，`first` 取检测结果列表第一颗，`abort` 保留旧的
安全中止行为。注意随机策略不看置信度/位置——若同型号候选项里有明显误检框，优先调高
`detector.confidence` 或布置时分开螺母，而不是指望随机策略绕开它。
执行模式仍沿用现有行为：先使能并回 home，后检测及 IK 预检。

`nut_task.yaml` 的 detector 段提供模型、推理解释器路径（python，本机为 `.venv-yolo`）、
confidence、imgsz、device、
可选原图像素 `roi` 等设置；默认仍是 manual，传 `--detector yolo` 切换。
默认 CPU 推理，也可按机器情况设置 `device: '0'` 使用 GPU。
实时模式采用当前彩色 camera_info，要求彩色/深度分辨率一致、消息时间差不超过 0.1s，
接收帧龄不超过 1s；推理结果超过 `max_result_age`（默认 10s）则拒绝。不同尺寸不会用
简单缩放冒充深度对齐。CPU 冷启动模型偶发推理过慢导致「快照已过期」、或单次推理子进程
失败时，会**清空帧缓存重新取一对新帧并重发识别**，最多 `inference_retries`（默认 3）次，
每次都打印第几次/帧龄；3 次都失败才中止（调大 `max_result_age`/`inference_timeout`
或检查 CPU 负载）。取帧本身超时（`camera_timeout` 内无同步帧）不重试，直接中止——
那通常意味着相机话题断了。

识别弹窗（2026-09-11 起，`detector.show_window: true`，当前配置已开）：等同步帧期间持续
刷新实时画面，推理前显示 `YOLO inferring ...`，出框后在彩色帧上画检测框/类别/置信度，
取到深度后补中心深度（mm），结果画面停留 `detector.show_seconds`（默认 2s）后自动继续——
**空格/回车立即继续，q/ESC 中止任务**，`show_seconds: 0` 则必须按键才继续（人工确认门）。
缺类重拍的每一轮都会重新弹窗；某颗深度无效时该框画红框+`DEPTH INVALID`，停留后才报错中止。
无显示环境（SSH/无 DISPLAY）自动降级为无窗，不影响识别；命令行 `--show` 可在 yaml 关闭时
强制开窗。

注意：中心是检测框中心，未做孔轮廓精定位。沿用原点选工具的中心邻域深度策略，
螺母孔可能测到桌面，需要现场核对抓取高度。主流程会在检测点变到 base_link 后统一加
`motion.grasp_offset_xyz`（默认向机体方向退 15cm）补偿腕-指尖前后偏差（见「配置」一节），
但该偏移只改腕部目标 xy，不会修正深度打在孔/桌面上造成的 z 误差。
外参必须对应当前相机安装；纯预览坐标成功不等于实物抓取误差已经验证。

## 1. 执行流程与代码对应

任务开始只做一次，之后每颗螺母（顺序由 `order` 决定）循环：

| 阶段 | 内容 | 实现 |
|---|---|---|
| 0（一次） | 上使能 → 双手张开到 `hand.open`（保持 1s）→ 左右臂**各按一条预录 ready 轨迹**（`left.ready`/`right.ready`）先慢速接入 pt0、再逐点 MoveJ 回放到末点离场，打印 ready 末点目标/实际/误差并保持 | `nut_pick_place.initial_poses` 调 `SequenceRunner.run_leg`；可单独 `--execute --go-ready` 验证 |
| 1（一次） | 双臂离场后视觉检测大/中/小，像素+深度→相机系→base 系 | `nut_detectors.py` + `camera_pick_move` 标定链路 |
| 2 | 左臂 MoveJP 到螺母正上方（高 10cm）→ MoveL 竖直下探 → 按尺寸闭合 → MoveL 抬起；姿态取该尺寸的 `grasp_orientation_by_size`（未配则 `left_grasp_init`） | `nut_pick_place.run_one_nut` |
| 3 | 左臂回放 left_middle_grasp 段，终点张手放中央，再回放 left_middle_back 回位 | `nut_sequences.SequenceRunner.run_leg` |
| 4 | 右臂回放 right_grasp_middle1 段到中央，段尾闭合重抓 | 同上，`hand_after: close` |
| 5 | 右臂回放 right_middle_back1 段，段尾张手在右侧释放（当前三颗共用一条） | 同上，`hand_after: open` |

- 视觉 xyz 经外参 `p_base = R·p_cam + t` 变到 base_link；两臂 pose_states 共用同一个
  躯干 base_link，所以右臂标定的外参直接用于左臂目标。
- 视觉点（hover/down）运动前全部过**驱动实时 IK 预检**，任一不可达就在任何运动前中止。
- 序列段逐点 MoveJ，每点下发前校验反馈新鲜、实时关节名与记录一致、**另一只臂没动**
  （漂移超 `other_tolerance` 立即中止防干涉）；服务完成后等反馈进入
  `reached_tolerance`（最多 `reached_wait_seconds`，默认 3s），超差则**自动补发同目标
  MoveJ** 让伺服逐次收敛（默认最多补发 `reached_reissue_count=2` 次，即最多下发 3 次，
  全程只发同一目标、不做任何新移动），次数用尽仍超差才中止，且报错列出超差关节名与
  rad/度偏差。
- **视觉 MoveJP/MoveL 同样核对笛卡尔到位**：block 服务提前返回时末端可能还没到
  （关节残差 1.9° 在 400mm 臂展≈13mm），而驱动 MoveJP 逆解以当前关节角为种子，每颗起始
  臂型不同（第 1 颗从 ready、之后从回位段末点出发），相同坐标三次落点可能不一致。框架在
  hover/down/抬起三处都按 `pose_pos_tolerance`（默认 10mm）/`pose_ori_tolerance`（默认
  0.05rad）核对 pose_states。**残差仍在减小=运动还在执行，只继续等、绝不补发**（窗口末尾
  0.5s 内残差改善 >5mm 判为在动，顺延等待，期间打印「仍在收敛，不打断运动」；进容差后还要
  连续保持 0.15s 防过冲瞬间误判）；只有残差平台化（真稳态偏差）才补发同目标，补发 2 次仍
  超才中止。运动中重发同目标会打断驱动轨迹，表现为臂突然抖动偏离再回位——此规则即为消除
  该抖动。MoveJ 各点同样处理。
- **稳态残差 ≠ 收敛滞后**：若几次重发残差几乎不变（现场右臂持螺母伸展时肩滚转稳态停在
  0.033rad/1.9°，三次下发完全一样），说明是受力姿态下伺服补不掉的稳态偏差，补发无用；
  用分臂容差 `reached_tolerance_right/left`（rad，(0,0.1]）单独放宽该臂即可，当前右臂配
  0.05（≈2.9°），左臂维持全局 0.03。dry-run 会打印实际生效的分臂容差。
- 若相同输入落点仍抖动，可让每颗视觉抓取前先慢速 MoveJ 回到左臂首段 pt0（抓取区记录臂型，
  即 IK 种子姿态），使每次 MoveJP 的起点臂型一致（需要时调整 run_one_nut 顺序）。
- 手动作只在**段终点**由框架发出（`hand_after`），录轨迹时不要做手部动作。

## 2. 固定段（当前为 2026-09-10 正式录制的 4 段）

预录序列（schema v3，`record_workpoints.py` 产物，每段 ≥2 个点）：

| 序列 | 文件夹 | 点数 | 段尾手动作 | 作用 |
|---|---|---|---|---|
| `left_middle_grasp_001` | recordings/left_grasp_middle1 | 3 | **open** | 抓取区 → 桌面中央，末点释放螺母 (380,89,-338)；第 0 点=左臂 retreat home |
| `left_middle_back_001` | recordings/left_middle_back | 2 | 无（纯运动段） | 中央抬起 → 回位 |
| `right_grasp_middle1_001` | recordings/right_middle_grasp1 | 3 | **close** | retreat home（第 0 点）→ 桌面中央，末点重抓螺母 (361,22,-332) |
| `right_grasp_move1_001` | recordings/right_middle_back1 | 3 | **open** | 中央 → 右侧盒位释放点 (476,-353,-222)；三颗螺母共用这一条 |

> 2026-09-10 晚切换：旧三段 left_grasp_place_middle_001 / right_grasp_middle_001 /
> right_middle_back_001 已不在 yaml 引用（录文件保留；right_grasp_middle 末点曾抬高 3cm）。

段间端点不必严格重合：框架以 `join_speed`（0.15 rad/s）慢速 MoveJ 直动接入下一段首点，
dry-run「段间接入距离」报告会列出每条补动的距离（无避障，>250mm 会提醒）。

**开机 ready 段（与任务段分开配置）**：上使能、张开初始手型后，左右臂各回放一条专用轨迹
离场，视觉检测时停在其**末点**；当前配置：

| 臂 | 配置 | 文件 / 序列 | 点数 | ready 末点 xyz(mm) |
|---|---|---|---|---|
| 左 | `left.ready` | recordings/left_trace2 · `left_ready2_001` | 3 | (257, 390, -271) |
| 右 | `right.ready` | recordings/right_trace · `right_ready1_001` | 3 | (426, -215, -175) |

- 回放方式与任务段相同：先慢速 MoveJ 接入该序列 **pt0**（所以 pt0 要在开机安全位附近），
  再逐点 MoveJ；ready 段是纯运动段，**不许配 `hand_after`/`retreat`**（初始张手由框架统发）。
- **要改 ready 姿态：重录一条序列（或把点录进现有 trace 文件），改 yaml 里的
  `file`/`sequence` 即可，不用动代码**；删掉 ready 配置则回退为直接慢速 MoveJ 到任务段 pt0。
- ready 末点到左臂首段首点（dry-run 实测 133mm）这段不是框架直动：左臂第一下是
  MoveJP 到视觉 hover，由驱动规划；右臂 ready 末点到 approach 首点（53mm）由框架
  自动慢速 MoveJ 接入，dry-run「段间接入距离」前两行就是这两条。

唯一需要在桌面上对齐的关键点（容差建议 ≤20mm）：

```
left_middle_grasp_001 末点（左释放） ≈ right_grasp_middle1_001 末点（右重抓）
```

dry-run「交接点核对」超 20mm 标 ⚠。2026-09-10 切换新三段后此处差 **70mm**
（左释放 (380,89,-338) vs 右抓 (361,22,-332)），用户决定**先按现状试真机**，
若右臂抓空再以左臂释放点为基准重录右臂重抓末点。
（旧录段：左释放 (400,74,-350)，右抓原 (359,19,-338)、曾抬高 3cm 到 -308，
备份 events.jsonl.bak_20260910_raise3cm；右新录段末点曾抬高 0.6cm，
备份 right_middle_grasp1/events.jsonl.bak_20260910_raise6mm。）
新 place 段 right_grasp_move1_001 为 3 点轨迹，末点 (476,-353,-222)，
比旧段末点 (399,-314,-72) 低很多、更接近盒底（旧段是有意的高位空投点，已弃用）。

若以后三颗要分三个格位：把 yaml 的 `right.place` 从单段改成 `l:/m:/s:` 三段即可，
此时各 place 段**第 0 点必须就是中央重抓点**，dry-run 会逐段核对。

**右 approach 段按尺寸分段（中小螺母拇指高度不够时）**：approach 与 place 一样支持
`l:/m:/s:` 三选一写法（2026-09-11 起；仍必须 `hand_after: close`）。大螺母已调好、只改
中小螺母时，**复制原段文件后只微调副本末点**（三段首点必须一致，dry-run 会检查并警告）：

```bash
cd /home/lionheart/Project/1_Competition/THUEI/Build_ws/lbot_ws
# 1) 复制段目录（原段 right_middle_grasp1 保持给大螺母用，不动）
cp -r recordings/right_middle_grasp1 recordings/right_middle_grasp1_ms
# 2) 只算不动：副本末点降低 5mm（--dz 米，负值更低；需要俯仰再加 --drx 度），
#    驱动 IK 多种子求解+FK 复核，关节与 pose 快照一起回写，自动备份 events.jsonl.bak_*
/usr/bin/python3 tools/retune_waypoint.py \
  --file recordings/right_middle_grasp1_ms/events.jsonl \
  --sequence right_grasp_middle1_001 --arm right --dz -0.005
# 3) 真机低速预览这一段（默认 dry，--execute 才动；先慢速接入起点）
/usr/bin/python3 tools/replay_workpoints.py \
  recordings/right_middle_grasp1_ms/events.jsonl \
  --to right_grasp_middle1_001 --arm right --execute --move-to-start
```

然后把 yaml 改成三段（m/s 先共用同一副本，以后要分开再复制一个 `_s` 目录）：

```yaml
right:
  approach:
    l: {file: recordings/right_middle_grasp1/events.jsonl,    sequence: right_grasp_middle1_001, hand_after: close}
    m: {file: recordings/right_middle_grasp1_ms/events.jsonl, sequence: right_grasp_middle1_001, hand_after: close}
    s: {file: recordings/right_middle_grasp1_ms/events.jsonl, sequence: right_grasp_middle1_001, hand_after: close}
```

dry-run 段表会列出 3 条 approach、交接点逐尺寸核对；单尺寸首调：
`nut_pick_place.py --execute --detector ... --order m`（先用 --go-ready 离场）。
微调每次 2~5mm 小步，retune 默认拒绝与原臂型相差 >25° 的解（防 IK 翻转支），
需要更大姿态变化时应重新 `record_workpoints` 拖教而不是硬转。

另需 1 个抓取姿态（只用它的三个旋转角，xyz 由视觉覆盖）。`left.grasp_orientation` 两种写法：

1. 位姿库名（默认）：`grasp_orientation: left_grasp_init`，姿态存在 `task_poses.yaml`，
   当前为 2026-09-10 现场给定 euler=(65.22°, -6.67°, -89°)（中途值 -0.4,0.39,7.97 已弃用；
   最早值 52.9,0,-86.3 备份在 `task_poses.yaml.bak_20260910_euler52`）。重采：

   ```bash
   python3 tools/capture_task_pose.py --arm left --name left_grasp_init --force
   ```

2. 直接取记录段某点的旋转角（推荐与现场臂型一致时用）：

   ```yaml
   grasp_orientation:
     file: recordings/left_grasp_middle/events.jsonl   # 可省略，省略用 left.trace
     sequence: left_grasp_place_middle_001
     point: 0        # left_grasp_middle pt0 eul≈[63.7,-1.3,-95.4]°（抓取区臂型）
   ```

**按尺寸分别配置姿态与偏移（2026-09-11 起）**：大中小螺母可各自给抓取姿态和抓取几何，
未配置的尺寸自动回退上面的全局值。dry-run 计划的「视觉抓取点偏移」「左臂视觉抓取姿态」
两节按 l/m/s 逐行打印，覆盖项标注（覆盖全局）。

- 姿态：`left.grasp_orientation_by_size`，键 l/m/s，值的写法与全局 `grasp_orientation`
  完全相同（位姿库名或 `{file?, sequence, point?}`）。位姿名需先逐个采集，例如：

  ```bash
  python3 tools/capture_task_pose.py --arm left --name left_grasp_l --force
  python3 tools/capture_task_pose.py --arm left --name left_grasp_m --force
  python3 tools/capture_task_pose.py --arm left --name left_grasp_s --force
  ```

  注意按尺寸分的仍**只是三个欧拉角**；抓取点 xyz 永远来自视觉检测点 + 该尺寸偏移。
  某个名字没采/段点不存在时，dry-run 该行打 ⚠，`--execute` 在任何运动前中止并报缺哪个尺寸。

- 偏移/高度：`motion.grasp_by_size.<l/m/s>` 里 `offset_xyz` / `z_offset` / `hover_height`
  三个字段任选，单位和含义与全局 `grasp_offset_xyz` / `grasp_z_offset` / `hover_height`
  完全相同，逐字段回退（可以只覆盖某尺寸的 offset_xyz，其余仍用全局）。

**IK 预检与种子**：hover/down 运动前都要过驱动实时逆解。驱动逆解是数值法、以请求里的
关节角为初始种子（srv 注释明确），臂停在远处 ready 位时可能因种子不收敛而误报"逆解失败"。
框架会依次用 当前关节角 → 左臂首段（现为 `left_middle_grasp_001`）pt0/末点的记录臂型 → 空种子
（驱动自读当前角）做种子，任一通过即可达，并打印实际通过的种子；所以"hover 过、down 失败"
通常是种子问题而非姿态问题。所有种子都失败时自动打印 **IK 对照探针**（录段已知可达点、
曾手动验证的 MoveJP 位姿、失败点位置+录段姿态、录段位置+目标姿态，分别 ok/fail/TIMEOUT），
据此区分：服务异常（第 1 行都 fail/TIMEOUT）、位置问题（第 3 行 fail）、姿态问题（第 4 行 fail）。
确认是裸 IK 服务误判（探针 ok 但预检 fail）时可加 `--allow-ik-fail` 继续——真实 MoveJP/MoveL
仍由驱动求解把关，解不了会在该步安全中止，不会盲动；默认不加该参数仍是预检失败即中止。

## 3. 录轨迹操作

录制规范详见 [WORKPOINTS.md](WORKPOINTS.md)，要点：

```bash
cd /home/lionheart/Project/1_Competition/THUEI/Build_ws/lbot_ws
source /opt/ros/jazzy/setup.bash && source install/setup.bash

/usr/bin/python3 tools/record_workpoints.py --arm left    # 录左臂两段
/usr/bin/python3 tools/record_workpoints.py --arm right   # 录右臂四段
```

- 摇到点位按**回车**记录，一段录完按 `q`，输入名字（如 `left_ready1`），
  自动存成 `left_ready1_001`；同一进程可连录多段（编号递增 002/003…）。
- **录哪只臂的段，另一只臂必须全程停着不动**（框架加载/回放都校验，防双臂干涉）。
- 录制时**不要操作灵巧手**；手的张开/闭合由框架在段终点统一发。
- 相邻段的端点不必严格重合：框架会以 `join_speed`（默认 0.15 rad/s）在关节空间补一条
  MoveJ 接入下一段起点。但这条补动**没有避障规划**，所以两段端点要尽量近，或在段内多录
  过渡点；段内两点之间同理——点间是直 MoveJ，需要绕行就加点。
- 产物默认在 `recordings/日期_时间/events.jsonl`。当前配置按语义化文件夹引用：
  `recordings/{left_grasp_middle1,left_middle_back,right_middle_grasp1,right_middle_back1}/events.jsonl`
  （新录的时间戳文件夹可以直接 `mv` 改名；段内也可用 `file:` 单独指定任意文件，
  见 `nut_task.yaml`）。

可先用回放工具单段预览/验证（见 [REPLAY.md](REPLAY.md)）：

```bash
/usr/bin/python3 tools/replay_workpoints.py recordings/left_grasp_middle1/events.jsonl \
    --to left_middle_grasp_001 --arm left --move-to-start   # 预览
    # --execute 才真机回放
```

## 4. dry-run 与真机

```bash
# 离线核对：打印全部段点（base_link 系 xyz/euler）、交接点距离、手型/停顿参数；不动机器人
python3 tools/nut_pick_place.py

# 只验证开机动作（使能 -> 张开初始手型 -> 双臂按 left/right.ready 轨迹回放到 ready 末点），不到视觉/抓放：
python3 tools/nut_pick_place.py --execute --go-ready
```

开机顺序固定为：上使能 → 双手张开到 `hand.open`（当前 [255,80,255,255,255,255]，保持 1s）
→ 左右臂依次回放各自的 ready 轨迹：先以 join_speed(0.15 rad/s) 慢速 MoveJ 接入序列 pt0
（已在容差内则跳过并打印 Δ；实时关节名顺序与记录不一致直接中止），再按 sequence_speed
逐点 MoveJ 到末点 ready，到位后打印实际 xyz 与位置误差、保持 ready_hold_seconds 再检测。
ready 段和任务段在同一文件/配置体系内，改法见上文「开机 ready 段」。

```bash
# 临时跑通全流程：终端手动输入三颗螺母 base_link 位置（米；空格/逗号分隔，回车跳过）
python3 tools/nut_pick_place.py --execute --detector input
# 离线 dry-run（同样会提示输入，然后只打印计划不动机器人）
printf '0.378 0.326 -0.348\n0.35 0.30 -0.34\n0.32 0.27 -0.33\n' | \
  python3 tools/nut_pick_place.py --detector input

# 用 json 假检测额外核对「相机点→base→hover/down」坐标（detector.type 改 json）
python3 tools/nut_pick_place.py --detector json

# 真机：先双臂按 ready 轨迹离场，再弹窗依次点击 大/中/小（左键选点，u 撤销，回车确认，q 中止）
python3 tools/nut_pick_place.py --execute
# 临时覆盖：
python3 tools/nut_pick_place.py --execute --order sml       # 小→中→大
python3 tools/nut_pick_place.py --execute --detector external
python3 tools/nut_pick_place.py --execute --speed 1.5       # 整体提速 50%（倍率，可 <1 降速）
```

manual 点选窗口只在双臂到位后出现（避免臂挡住画面）。检测后、运动前会再打印一次
含每颗螺母 base 坐标/hover/down 的完整计划并做 IK 预检。

## 5. 配置（开发资源/nut_sort/nut_task.yaml）

- `order`：`[l,m,s]` / `[s,m,l]` / `[m,l,s]` 任意排列；命令行 `--order sml` 临时覆盖。
- `require_all`：缺螺母时 true=中止，false=只抓检测到的。
- `detector.duplicate_policy`：同型号多颗时 `random`=随机抓一颗（当前配置）、
  `first`=第一颗、`abort`=中止（缺省）。
- `detector.detect_attempts`：整轮识别后缺型号时，重新拍快照整体重识别的**轮数上限**
  （含首轮，默认 3，范围 1~10，必须整数）；轮间等 `detector.missing_retry_seconds`
  秒（默认 1.0，可设 0）。这是两层重试中的**业务层**：`inference_retries` 管单次识别内部
  快照过期/推理失败的重取帧，`detect_attempts` 管识别成功但整类漏检后的整轮重拍；
  `require_all: false` 时缺类允许跳过、不触发重试。螺母确实不在画面里时应改用 false 而不是
  一味调大轮数。
- `detector.show_window` / `detector.show_seconds`（YOLO）：识别阶段弹出实时窗口——等帧时
  显示实时画面，出框后画检测框/置信度/中心深度；结果停留 show_seconds 自动继续，空格立即
  继续、q 中止，0=必须按键确认；无 DISPLAY 自动不弹。命令行 `--show` 强制开窗。
- `left.grasp_orientation`：视觉抓取姿态源——位姿库名字符串，或
  `{file?, sequence, point?}` 直接取记录段某点欧拉角（见上文「固定段」一节）。
- `left.grasp_orientation_by_size`（可选）：l/m/s 各自的姿态源，写法同上；缺尺寸回退全局。
- `motion.grasp_by_size`（可选）：l/m/s 各自的 `offset_xyz`/`z_offset`/`hover_height`
  （三字段任选、逐字段回退全局），用于大中小螺母几何不同时分调，详见上文「固定段」一节。
- `left.ready` / `right.ready`：开机 ready 段 `{file, sequence}`（纯运动，不许带 hand_after）；
  不配则开机直接慢速 MoveJ 到首任务段 pt0。改 ready 姿态只需改这里指向的序列。
- `left.trace` / `right.trace`：该臂任务段的默认记录文件；段内可用 `file:` 覆盖。
- `motion.hover_height`：螺母正上方抬高，默认 0.10m。
- `motion.grasp_z_offset`：下探终点相对视觉点 z 的微调，想让指尖更低给负值（如 -0.01）。
- `motion.grasp_offset_xyz`：**检测点（螺母位置）→ 腕部目标**的 base_link 系平移（米，
  默认 `[-0.15, 0, 0]`）。视觉点选/YOLO 对准的是腕部（法兰），而指尖抓取中心在腕前约
  15cm：腕部目标整体向机体方向（base_link 负 X，已由外参朝向核实）退 15cm 后指尖才正好
  到螺母。三种检测器在变到 base_link 之后统一施加；dry-run 与执行日志会分别打印「检测点」
  与「腕部抓取目标」。实测指尖偏差不是 15cm 就改这三个数（单位米），不要偏移就设全 0。
  hover/down 都以补偿后的腕部目标为基准，IK 预检同样使用补偿后坐标。
- `motion.sequence_speed/acce`：预录段逐点 MoveJ 速度（≤0.5）；`join_speed/acce`：
  段间接入与 retreat 回 home 的慢速（≤0.3）。首调保持默认低值。
  命令行 `--speed N` 给整体倍率（视觉 MoveJP/MoveL、接入、逐点的速度加速度一起缩放，
  在 yaml 校验之后生效，故能临时超过 0.3/0.5 建议值，超过会打 ⚠ 提醒；yaml 基值不改）。
- `motion.other_tolerance/start_tolerance/reached_tolerance`：干涉保护与到位容差（rad）。
- `motion.reached_tolerance_left/right`：分臂到位容差覆盖（可选，(0,0.1] rad）；用于某臂
  持物受力姿态存在固定稳态残差（右 0.05 为 2026-09-10 现场值）。
- `motion.pose_pos_tolerance/pose_ori_tolerance`：视觉 MoveJP/MoveL 的末端到位容差
  （默认 0.01m / 0.05rad）；服务返回后核对 pose_states，超差补发同目标。
- `motion.reached_wait_seconds`：MoveJ 服务返回后等反馈进入容差的最长时间（默认 3.0，≥0.5）。
- `motion.reached_reissue_count`：未到位时同目标 MoveJ 最多补发次数（默认 2，范围 0~5；
  0=不补发直接中止）；每次补发后都重新等一个 `reached_wait_seconds`，仍不到位则中止并
  打印每个超差关节的 rad/度偏差。
- 节奏停顿（秒，嫌衔接快就调大这些）：`ready_hold_seconds`（开机到 ready 后保持，默认 1.0）、
  `hover_dwell_seconds`（到螺母上方、下探前，0.5）、`pre_hand_seconds`（每次张/合手前，0.3）、
  `point_dwell_seconds`（预录段每个点到位后，0.2）、`between_leg_seconds`（段与段之间，0.8）；
  另闭合后静置 `settle_seconds`、张手后静置 `release_seconds`（各 0.6）。
- `hand.close.left/right`：左右臂各自的 6 路闭合默认值（顺序
  [拇指侧摆,拇指弯曲,食,中,无名,小]，现场调）；当前现场值左 `[0,40,0,0,0,255]`、
  右 `[0,80,0,0,0,255]`，三颗共用。
- 需要按尺寸分手型时用 `hand.sizes`：裸列表 `{joint: [...]}`=双臂共用，
  或 `{left: [...], right: [...]}` 分臂；优先级 尺寸分臂 > 尺寸 joint > `hand.close`。
- `hand.force/speed` 是力矩和速度；`hand.open` **必填**（缺省会误发全 0 把手闭合），
  即开机张开值（当前 `[255,80,255,255,255,255]`，上使能后、任何臂运动前先发并保持 1s）。
- `vision.*`：相机话题（深度必须 `depth_registration:=true` 对齐）；
  `extrinsics/camera_info` 留空用标定默认文件。

## 6. 接入自己的视觉

检测器只回答「哪颗螺母在相机光学系的什么位置」，手眼反算由主流程统一完成。

1. 拷贝 `tools/nut_detector_example.py` 改 `detect()`；
2. 配置：

```yaml
detector:
  type: external
  external: '/绝对路径/my_detector.py:MyNutDetector'
  # 可加任意自定义键，原样通过 sub_cfg 传进构造函数
```

契约（详见 `nut_detector_example.py` 文件头）：

- `__init__(self, node, sub_cfg, K)`：node 是任务 rclpy 节点（可直接订阅彩色画面）；
  K 为彩色内参 3×3（可能为 None，可自己订阅 camera_info）。
- `detect(expected) -> list`：`expected` 如 `('l','m','s')`；每颗返回 Detection（或等价 dict），
  四种给法任选：
  - `Detection('l', p_cam=[X,Y,Z])`：你自己算好相机光学系 3D（米），框架走外参变换；
  - `Detection('l', [x,y,z], extra={'frame':'base_link'})`：**视觉模块直接给 base_link 系
    机械臂坐标（米）**，框架原样使用、跳过手眼变换；附带的 euler/quat 放 extra 只记录，
    抓取姿态仍以 `left_grasp_init` 为准。json 形式：`{"label":"l","frame":"base_link","p_base":[...]}`；
  - `Detection('l', None, u=.., v=.., z=..)`：像素 + 对齐深度，框架用 K 反投影；
  - **`Detection('l', None, u=.., v=..)`：只给画面像素，框架自动订阅对齐深度、
    在该像素邻域取样（深度空洞自动扩大半径）并反投影**——识别算法只出 2D 位置时用这个。
- 调用时机：双臂已沿 ready 轨迹离场（停在 ready 末点）、离开画面之后。检测函数内**不要发任何运动指令**。
- 同一尺寸返回多个目标会直接中止（无法决定抓哪个）；深度在螺母表面空洞/超画面会报明确错误。

联调参考：`开发资源/nut_sort/nut_detector_ref.py` 是固定位姿桩（位置 0.378,0.326,-0.348，
base_link 系；另带视觉给的欧拉/四元数仅记录）。配置里 `detector.external` 已指向它，
用以下命令只测这一颗参考点（hover 应为 [378, 326, -248]mm，down 为原 z）：

```bash
python3 tools/nut_pick_place.py --detector external --order l
```

## 7. 文件

| 文件 | 作用 |
|---|---|
| `开发资源/nut_sort/nut_task.yaml` | 任务配置：顺序、段映射、速度容差、三档手型、视觉/检测器 |
| `开发资源/nut_sort/task_poses.yaml` | 位姿库；左臂 `left_grasp_init`，按尺寸分姿态时再加 `left_grasp_l/m/s`（名字自定） |
| `recordings/{left_grasp_middle1,left_middle_back,right_middle_grasp1,right_middle_back1}/events.jsonl` | 预录关节序列（schema v3，4 段任务段；旧 *1 前文件夹保留未引用） |
| `recordings/left_trace2/events.jsonl` · `recordings/right_trace/events.jsonl` | 开机 ready 轨迹（左 `left_ready2_001` 3 点，末点 (257,390,-271)；右 `right_ready1_001` 3 点，末点 (426,-215,-175)；同文件里的旧试录段未被引用） |
| `tools/record_workpoints.py` / `replay_workpoints.py` | 序列录制 / 单段预览回放 |
| `tools/retune_waypoint.py` | 微调记录点末端 xyz **和/或姿态**（默认末点）：`--dx/--dy/--dz`（米）+ `--drx/--dry/--drz`（度，xyz 欧拉角增量），驱动 IK 多种子选最近臂型 + FK 复核，关节角与 pose 快照一起回写并自动备份；只算不动。如左臂中央释放末点降 0.6cm：`--file recordings/left_grasp_middle1/events.jsonl --sequence left_middle_grasp_001 --arm left --dz -0.006`；右臂中央重抓末点俯仰多压 5°：`...right_middle_grasp1... right_grasp_middle1_001 --arm right --drx -5` |
| `tools/capture_task_pose.py` | 位姿采集（当前实到位姿 → yaml） |
| `tools/nut_robot.py` | 配置/位姿库/双臂运动服务/灵巧手封装 |
| `tools/nut_sequences.py` | 段（Leg）加载、交接点核对、SequenceRunner 回放 |
| `tools/nut_detectors.py` | 检测器接口 + 画面点选/json/外部加载 |
| `tools/nut_detector_example.py` | 自接检测器模板（只出像素的最简示例） |
| `开发资源/nut_sort/nut_detector_ref.py` | 固定参考位姿桩（base_link 直给，联调用） |
| `tools/nut_pick_place.py` | 主流程（默认 dry-run） |
| `tools/test_nut_task.py` | 离线单测（85 项，含 ready 轨迹接入/无 ready 回退、停顿期持续 spin、关节到位补发+分臂容差+逐关节诊断、视觉笛卡尔位姿核对/补发/指令vs实际报错、速度倍率、记录段姿态源、多种子 IK+对照探针、终端输入检测器、同型号多目标 random/first/abort 策略、缺型号整轮重拍（detect_attempts/missing_retry_seconds）、识别弹窗配置校验、4 段真实任务段加载、共用 place、右 approach 按尺寸分段（l/m/s 选段/校验/交接行）、分臂闭合值、4 种检测结果形式、base 系直给、grasp_offset_xyz 腕部偏移、按尺寸姿态/偏移覆盖与回退、非法值校验） |
| `tools/test_nut_yolo.py` | 深度/内参/ROI/locate 换算 + 重取帧重试 + annotate 画框 + DetectionWindow 弹窗生命周期/按键/超时/headless 降级（22 项） |
| `tools/test_nut_yolo_live.py` | 配对取帧纯函数：积压跳帧、过期/失配拒绝、坏深度不掩盖另一检测（3 项） |

## 8. 安全

- 默认 dry-run；`--execute` 才运动。首次执行清空桌面活动范围、手持急停，低速起调。
- 运动前视觉点强制 IK 预检；服务/反馈缺失、未采 `left_grasp_init`、序列/关节名/frame
  不一致、缺螺母（require_all，且 detect_attempts 轮重拍后仍缺），都在运动前中止；
  同尺寸多目标按 detector.duplicate_policy 处理（random/first 不中止，随机/取首颗继续，
  缺省 abort 中止）。
- 回放中另一只臂发生漂移立即中止；异常后**不自动掉使能**，在途运动需现场确认。
- 外参残差约 10.6mm；相机被碰过必须重新标定，每次启动都重读外参 yaml，换文件免操作。
- 手型值先小力慢速空载验证，确认不夹线缆/盒壁。

## 9. 测试

```bash
cd tools
source /opt/ros/jazzy/setup.bash && source ../install/setup.bash
/usr/bin/python3 -m pytest test_nut_task.py test_nut_yolo.py test_nut_yolo_live.py -q
# 当前共 110 项；只用 unittest 也可逐个文件跑
```
