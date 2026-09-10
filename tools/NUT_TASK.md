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

需要已启动彩色、对齐深度和彩色 camera_info。模型在 Conda 子进程中常驻，
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
ROS 系统 Python 负责相机与运动接口，`nut_yolo_infer.py` 子进程使用 Conda 的
`nut-yolo` 环境执行 YOLO，两者不用安装到同一个 Python 环境。

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
# 确认实物坐标、抓取高度、交接点后，实际运行：
source install/setup.zsh
/usr/bin/python3 tools/nut_pick_place.py --detector yolo --execute
```

执行路径：YOLO 框中心 -> 配对深度邻域中位数 -> 内参反投影 `p_cam` ->
现有 `det_base_xyz` 外参换算 -> `run_one_nut` 左臂 hover/down -> 既有双臂序列。
缺类或同类多目标由现有业务校验中止，不擅自选最高置信度目标。
执行模式仍沿用现有行为：先使能并回 home，后检测及 IK 预检。

`nut_task.yaml` 的 detector 段提供模型、Conda python、confidence、imgsz、device、
可选原图像素 `roi` 等设置；默认仍是 manual，传 `--detector yolo` 切换。
默认 CPU 推理，也可按机器情况设置 `device: '0'` 使用 GPU。
实时模式采用当前彩色 camera_info，要求彩色/深度分辨率一致、消息时间差不超过 0.1s，
接收帧龄不超过 1s；推理结果超过 10s 则拒绝。不同尺寸不会用简单缩放冒充深度对齐。

注意：中心是检测框中心，未做孔轮廓精定位。沿用原点选工具的中心邻域深度策略，
螺母孔可能测到桌面；输出不是自动修正后的指尖接触点，需要现场核对抓取高度。
外参必须对应当前相机安装；纯预览坐标成功不等于实物抓取误差已经验证。

## 1. 执行流程与代码对应

任务开始只做一次，之后每颗螺母（顺序由 `order` 决定）循环：

| 阶段 | 内容 | 实现 |
|---|---|---|
| 0（一次） | 双臂慢速 MoveJ 到各自 home（= 首段第一个记录点），张手 | `SequenceRunner.join_to_start` |
| 1（一次） | 双臂离场后视觉检测大/中/小，像素+深度→相机系→base 系 | `nut_detectors.py` + `camera_pick_move` 标定链路 |
| 2 | 左臂 MoveJP 到螺母正上方（高 10cm）→ MoveL 竖直下探 → 按尺寸闭合 → MoveL 抬起；姿态三角度固定取 `left_grasp_init` | `nut_pick_place.run_one_nut` |
| 3 | 左臂回放 left_grasp_place_middle 段，终点张手放中央，再回放 left_middle_back 回位 | `nut_sequences.SequenceRunner.run_leg` |
| 4 | 右臂回放 right_grasp_middle 段到中央，段尾闭合重抓 | 同上，`hand_after: close` |
| 5 | 右臂回放 right_middle_back 段，段尾张手在右侧释放（当前三颗共用一条） | 同上，`hand_after: open` |

- 视觉 xyz 经外参 `p_base = R·p_cam + t` 变到 base_link；两臂 pose_states 共用同一个
  躯干 base_link，所以右臂标定的外参直接用于左臂目标。
- 视觉点（hover/down）运动前全部过**驱动实时 IK 预检**，任一不可达就在任何运动前中止。
- 序列段逐点 MoveJ，每点下发前校验反馈新鲜、实时关节名与记录一致、**另一只臂没动**
  （漂移超 `other_tolerance` 立即中止防干涉）；到位超 `reached_tolerance` 才继续。
- 手动作只在**段终点**由框架发出（`hand_after`），录轨迹时不要做手部动作。

## 2. 固定段（当前为 2026-09-10 正式录制的 4 段）

预录序列（schema v3，`record_workpoints.py` 产物，每段 ≥2 个点）：

| 序列 | 文件夹 | 点数 | 段尾手动作 | 作用 |
|---|---|---|---|---|
| `left_grasp_place_middle_001` | recordings/left_grasp_middle | 3 | **open** | 抓取区 → 桌面中央，末点释放螺母；第 0 点=左臂 home |
| `left_middle_back_001` | recordings/left_middle_back | 2 | 无（纯运动段） | 中央抬起 → 回位 |
| `right_grasp_middle_001` | recordings/right_grasp_middle | 3 | **close** | home（第 0 点）→ 桌面中央，末点重抓螺母 |
| `right_middle_back_001` | recordings/right_middle_back | 2 | **open** | 中央 → 右侧释放点；三颗螺母共用这一条 |

段间端点不必严格重合：框架以 `join_speed`（0.15 rad/s）慢速 MoveJ 直动接入下一段首点，
dry-run「段间接入距离」报告会列出每条补动的距离（无避障，>250mm 会提醒）。

唯一需要在桌面上对齐的关键点（容差建议 ≤20mm）：

```
left_grasp_place_middle 末点（左释放） ≈ right_grasp_middle 末点（右重抓）
```

dry-run「交接点核对」超 20mm 标 ⚠。2026-09-10 正式录制此处差 **69mm**
（左释放 (400,74,-350) vs 右抓 (359,19,-338)），用户决定**先按现状试真机**，
若右臂抓空再以左臂释放点为基准重录 right_grasp_middle 末点。
right_middle_back 末点 (399,-314,-72) 是**有意的高位释放点**（盒子高/空投），
距桌面 278mm 属正常，不是漏录下放点。

若以后三颗要分三个格位：把 yaml 的 `right.place` 从单段改成 `l:/m:/s:` 三段即可，
此时各 place 段**第 0 点必须就是中央重抓点**，dry-run 会逐段核对。

另需 1 个位姿（只用它的三个旋转角）：`task_poses.yaml` 里的左臂 `left_grasp_init`
已按 2026-09-10 现场给定写入，euler=(0.924, 0, -1.506) rad（52.9°, 0°, -86.3°）；
姿态需要改时重新采：

```bash
python3 tools/capture_task_pose.py --arm left --name left_grasp_init --force
```

xyz 会被视觉覆盖。

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
  `recordings/{left_grasp_middle,left_middle_back,right_grasp_middle,right_middle_back}/events.jsonl`
  （新录的时间戳文件夹可以直接 `mv` 改名；段内也可用 `file:` 单独指定任意文件，
  见 `nut_task.yaml`）。

可先用回放工具单段预览/验证（见 [REPLAY.md](REPLAY.md)）：

```bash
/usr/bin/python3 tools/replay_workpoints.py recordings/left_grasp_middle/events.jsonl \
    --to left_grasp_place_middle_001 --arm left --move-to-start   # 预览
    # --execute 才真机回放
```

## 4. dry-run 与真机

```bash
# 离线核对：打印全部段点（base_link 系 xyz/euler）、交接点距离、外参残差；不动机器人
python3 tools/nut_pick_place.py

# 用 json 假检测额外核对「相机点→base→hover/down」坐标（detector.type 改 json）
python3 tools/nut_pick_place.py --detector json

# 真机：先双臂回 home，再弹窗依次点击 大/中/小（左键选点，u 撤销，回车确认，q 中止）
python3 tools/nut_pick_place.py --execute
# 临时覆盖：
python3 tools/nut_pick_place.py --execute --order sml       # 小→中→大
python3 tools/nut_pick_place.py --execute --detector external
```

manual 点选窗口只在双臂到位后出现（避免臂挡住画面）。检测后、运动前会再打印一次
含每颗螺母 base 坐标/hover/down 的完整计划并做 IK 预检。

## 5. 配置（开发资源/nut_sort/nut_task.yaml）

- `order`：`[l,m,s]` / `[s,m,l]` / `[m,l,s]` 任意排列；命令行 `--order sml` 临时覆盖。
- `require_all`：缺螺母时 true=中止，false=只抓检测到的。
- `motion.hover_height`：螺母正上方抬高，默认 0.10m。
- `motion.grasp_z_offset`：下探终点相对视觉点 z 的微调，想让指尖更低给负值（如 -0.01）。
- `motion.sequence_speed/acce`：预录段逐点 MoveJ 速度（≤0.5）；`join_speed/acce`：
  段间接入与 retreat 回 home 的慢速（≤0.3）。首调保持默认低值。
- `motion.other_tolerance/start_tolerance/reached_tolerance`：干涉保护与到位容差（rad）。
- `hand.close.left/right`：左右臂各自的 6 路闭合默认值（顺序
  [拇指侧摆,拇指弯曲,食,中,无名,小]，现场调）；当前现场值左 `[0,40,0,0,0,255]`、
  右 `[0,80,0,0,0,255]`，三颗共用。
- 需要按尺寸分手型时用 `hand.sizes`：裸列表 `{joint: [...]}`=双臂共用，
  或 `{left: [...], right: [...]}` 分臂；优先级 尺寸分臂 > 尺寸 joint > `hand.close`。
- `hand.force/speed` 是力矩和速度；`hand.open` 张开值。
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
- 调用时机：双臂已回 home、离开画面之后。检测函数内**不要发任何运动指令**。
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
| `开发资源/nut_sort/task_poses.yaml` | 位姿库；本版只需左臂 `left_grasp_init` |
| `recordings/{left_grasp_middle,left_middle_back,right_grasp_middle,right_middle_back}/events.jsonl` | 预录关节序列（schema v3，4 段） |
| `tools/record_workpoints.py` / `replay_workpoints.py` | 序列录制 / 单段预览回放 |
| `tools/capture_task_pose.py` | 位姿采集（当前实到位姿 → yaml） |
| `tools/nut_robot.py` | 配置/位姿库/双臂运动服务/灵巧手封装 |
| `tools/nut_sequences.py` | 段（Leg）加载、交接点核对、SequenceRunner 回放 |
| `tools/nut_detectors.py` | 检测器接口 + 画面点选/json/外部加载 |
| `tools/nut_detector_example.py` | 自接检测器模板（只出像素的最简示例） |
| `开发资源/nut_sort/nut_detector_ref.py` | 固定参考位姿桩（base_link 直给，联调用） |
| `tools/nut_pick_place.py` | 主流程（默认 dry-run） |
| `tools/test_nut_task.py` | 离线单测（40 项，含 4 段真实轨迹加载、共用 place、分臂闭合值、像素自动补深度、base 系直给） |

## 8. 安全

- 默认 dry-run；`--execute` 才运动。首次执行清空桌面活动范围、手持急停，低速起调。
- 运动前视觉点强制 IK 预检；服务/反馈缺失、未采 `left_grasp_init`、序列/关节名/frame
  不一致、缺螺母（require_all）、同尺寸多目标，都在运动前中止。
- 回放中另一只臂发生漂移立即中止；异常后**不自动掉使能**，在途运动需现场确认。
- 外参残差约 10.6mm；相机被碰过必须重新标定，每次启动都重读外参 yaml，换文件免操作。
- 手型值先小力慢速空载验证，确认不夹线缆/盒壁。

## 9. 测试

```bash
cd tools
/usr/bin/python3 -m unittest -v test_nut_task.py
```
