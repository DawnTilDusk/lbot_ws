# 任务二：把螺母依次叠到铁针上（`--place-at needle`）

任务一（`NUT_TASK.md`）把螺母放进盒子；**任务二**改成把白 / 黑大 / 中 / 小四颗螺母
依次穿过玻璃盘中央的铁针、叠成一摞。

```
左臂视觉抓螺母 → 预录轨迹放到桌面中央 → 右臂重抓
                                    ↓
                        直接横移到铁针 (x0,y0) 上方
                                    ↓
                    降到「桌面 + 20cm」松手 → 抬起 → 下一颗
```

和任务一**共用同一套双臂交接**（左臂视觉抓取、左臂固定段、右臂中央重抓），
**唯一区别是右臂拿到螺母之后**：任务一回放入盒段，任务二自己算一条到铁针的路径。

铁针只在**开机双臂离场时识别一次**，之后四颗都用同一个 `(x0, y0)`。

---

## 1. 怎么跑

### 1.1 三个终端（每次都要）

**终端 A — 驱动**
```bash
source /opt/ros/jazzy/setup.bash
source /home/undefined/cleverhand/lbot_ws/install/setup.bash
ros2 launch lbot_driver lbot_start_driver.launch.py
```

**终端 B — 相机**
```bash
source /opt/ros/jazzy/setup.bash
source /home/undefined/orbbec_ws/install/setup.bash
ros2 launch orbbec_camera gemini2.launch.py \
    enable_color:=true enable_depth:=true depth_registration:=true \
    enable_point_cloud:=false enable_colored_point_cloud:=false \
    time_domain:=system
```

> `time_domain:=system` **必须带**。用默认的 `global` 会让彩色/深度两路时间戳差一个
> 恒定偏移（实测深度差 19s、彩色差 579s），框架会一直报
> 「等待彩色、对齐深度、彩色 camera_info 超时，或图像时间戳不同步」。

**终端 C — 任务**
```bash
cd /home/undefined/lbot_ws-2
source /opt/ros/jazzy/setup.bash
source /home/undefined/cleverhand/lbot_ws/install/setup.bash
source /home/undefined/orbbec_ws/install/setup.bash
```

### 1.2 跑之前先空跑

```bash
python3 tools/nut_pick_place.py --detector yolo --place-at needle
```

弹识别窗、打印完整计划、**不动机器人**。确认：

- 四颗螺母都认出来了（`白` / `大` / `中` / `小`）
- 「视觉抓取点偏移」那一段四颗的 `down +XXmm` 和你标定的一致
- 交接点核对那些 ⚠ 可以忽略（它比的是两只臂的**腕部**位姿，不是指尖，135mm 左右是正常的）

### 1.3 真机跑

```bash
# 四颗全部（order 已在 yaml 里 = [white, l, m, s]）
python3 tools/nut_pick_place.py --detector yolo --place-at needle --execute

# 只跑一颗（调试用）
python3 tools/nut_pick_place.py --detector yolo --place-at needle --order white --execute

# 交接点停下，肉眼确认左手松没松，回车继续
python3 tools/nut_pick_place.py --detector yolo --place-at needle --pause-handoff --execute

# 只走第一颗的 hover 悬停位（核对视觉点准不准）
python3 tools/nut_pick_place.py --detector yolo --place-at needle --stop-at-hover --execute

# 只走「抓起 + 上抬」（验证左臂抓得稳不稳）
python3 tools/nut_pick_place.py --detector yolo --place-at needle --stop-after-lift --execute

# 只验证开机动作（使能 → 张手 → 双臂回 ready），最常见的"臂不动"排查入口
python3 tools/nut_pick_place.py --execute --go-ready

# 提速
python3 tools/nut_pick_place.py --detector yolo --place-at needle --speed 1.5 --execute
```

任何一步都可以 `Ctrl-C`：**臂保持当前姿态，不会掉使能**。

### 1.4 全部命令行参数

| 参数 | 作用 |
|---|---|
| `--config PATH` | 换任务配置（默认 `开发资源/nut_sort/nut_task.yaml`） |
| `--order l,m,s` / `--order white,l,m,s` | 覆盖抓取顺序；单个如 `--order l` 只跑一颗 |
| `--detector yolo` | **任务二必须**（铁针识别也走它） |
| `--place-at needle` | 任务二开关；`box`（默认）= 任务一 |
| `--execute` | 真机动；不加 = dry-run |
| `--go-ready` | 只做开机动作后退出 |
| `--pause-handoff` | 左臂中央释放后暂停等回车 |
| `--stop-at-hover` | 停在第一颗螺母正上方 |
| `--stop-after-lift` | 停在左臂抓起上抬后 |
| `--show` | 强制弹识别窗 |
| `--speed N` | 整体速度倍率 |
| `--allow-ik-fail` | IK 预检失败也继续（探针证明是误判时才用） |

---

## 2. 执行流程（逐步 + 代码位置）

### 步骤 0：开机（`nut_pick_place.initial_poses`）

| # | 动作 | 说明 |
|---|---|---|
| 1 | 双臂 `set_enable` | 失败会**硬报错**（`nut_robot.RobotClient.enable`） |
| 2 | 下发手速/手力 + 双手张开 `hand.open` | 保持 1s |
| 3 | 双臂回放 `left.ready` / `right.ready` | 先慢速 MoveJ 接入 pt0，再逐点 MoveJ 到末点（离场位） |
| 4 | 首轮 YOLO 识别 | 一次拍全部四颗 |
| 5 | **识别铁针**（只此一次） | `nut_needle.detect_needle` |
| 6 | 视觉点 IK 预检 | hover / down / lift 三点，多种子 |
| 7 | 打印计划 | 段表、交接点核对、抓取偏移 |

**「双臂回 ready」是唯一一处无避障的大幅关节运动**：`join_to_start` 读当前关节角，
差多少补多少（`start_tolerance 0.05`），纯关节空间慢速 MoveJ（`join_speed 0.3`）。
从"双臂自然下垂"起步是设计内的，不是异常。

### 步骤 1：左臂视觉抓取（`nut_pick_place.run_one_nut`）

| # | 动作 | 实现 |
|---|---|---|
| 1 | 竖直上升到中转平面 | SDK MoveL；高度见下 |
| 2 | 水平横移到 hover 正上方 | `move_via_ik`，**按 25mm 分段** |
| 3 | 下降到 hover | `move_via_ik`，按 25mm 分段 |
| 4 | 下探到 down | `move_via_ik`，按 25mm 分段 |
| 5 | 合手 | `cfg.close_for('left', label)` |
| 6 | 竖直上抬 lift | `move_via_ik`，按 25mm 分段 |

抓取点换算：

```
pb     = 检测点 + grasp_by_size.<尺寸>.offset_xyz      （腕部目标）
hover  = pb + [0, 0, hover_height]                     （默认 +120mm）
down   = pb + [0, 0, z_offset]                         （按尺寸标定）
lift   = down + [0, 0, lift_height]                    （默认 +50mm）
```

**中转平面高度是自动算的**（`resolve_transit_z`）：从理想高度（桌面 -0.44 + 0.25 =
-190mm）逐级下降 25mm，取第一个**「上升 + 整条分段横移」全程臂型连续**的高度。
下限是 hover 上方 50mm。日志会打：

```
[左] 中转平面 -215.0mm —— 理想高度 -190.0mm 走不通，降 1 档到 -215.0mm（上升最大差 4.1°，横移 252mm/10 段最大差 6.7°）
```

### 步骤 2：左臂固定段 + 中央释放

```
[left] 回放段 left_middle_grasp_001     # 3 点，末点 OPEN 张手
[left] 段尾张手释放（[185, 80, 255, 255, 255, 255]），静置 0.6s
[left] 中央释放复核：补发一次张手       # 防丢包
[left] 回放段 left_middle_back_001      # 2 点，回位（纯运动段）
```

### 步骤 3：右臂中央重抓

```
[right] 回放段 right_grasp_middle1_001（按尺寸选段：l/white、_m、_s）
        pt0 home → pt1 → pt2 中央重抓点，段尾 CLOSE 合手
```

### 步骤 4：叠到铁针（`nut_needle.place_at_needle`）

从 pt2（右臂刚捏住螺母）出发，全程 `move_via_ik`（外部 IK + MoveJ），**每段运动前
先把所有段的解算完并核对，任何一段失败都在动之前中止**：

| # | 动作 | 目标 |
|---|---|---|
| 1 | 竖直上升让针 | `[当前 x, 当前 y, 释放z + transit_clearance]` |
| 2 | 水平横移到针正上方 | `[x0, y0, 释放z + transit_clearance]` |
| 3 | 竖直下降到释放点 | `[x0, y0, 释放z]`，按 25mm 分段 |
| 4 | 张手释放 | `hand.open`，静置 `release_seconds` |
| 5 | 竖直抬起让开 | 回到中转平面 |

释放点：

```
x0, y0 = 铁针关键点（keypoint: base|tip）反投影到 base_link
释放z  = 桌面(-0.44) + needle.release_height(0.20) = -240mm
最终目标 = [x0, y0, 释放z] + needle.release_offset_xyz
```

**释放姿态默认用右臂当前的抓握姿态**（从 `/pose_states` 四元数转欧拉）——
螺母本来就是平着握的，直接松手最自然，而且第一步上升不必同时大角度换向。

### 步骤 5：下一颗

四颗依次重复步骤 1~4。`redetect_each_nut: false`（当前）表示四颗都用**首轮快照**
的位置连抓；改成 `true` 则每抓一颗前双臂回 ready 重拍快照。

---

## 3. 配置（`开发资源/nut_sort/nut_task.yaml`）

### 3.1 任务二专属段

```yaml
needle:
  model: weights/needle_pose_best.pt   # 两关键点 Pose 模型（tip / base）
  keypoint: base        # 用哪个关键点定 (x0,y0)；细杆深度不可靠时优先 base
  release_height: 0.20  # 释放点高于桌面多少米（任务要求 20cm）
  transit_clearance: 0.10   # 横移平面再高出释放点多少米
  min_confidence: 0.5   # 关键点置信度下限
  release_euler_deg: null   # 释放姿态；null=用当前抓握姿态
  max_joint_diff_deg: 60.0  # 每段允许的最大单关节变化
  release_offset_xyz: [-0.075, -0.077, 0.0]   # 现场标定的系统性偏差
  fallback_depth: null  # 关键点深度无效时的固定相机系 Z（米），null=直接报错
```

### 3.2 抓取相关（任务一/二共用）

```yaml
motion:
  transit_z: null       # 中转平面绝对高度覆盖（米）；null=自动
  grasp_by_size:
    l: {offset_xyz: [-0.01, -0.03, 0.0], z_offset: 0.087, hover_height: 0.12}
    m: {offset_xyz: [-0.022, -0.053, 0.0], z_offset: 0.082, hover_height: 0.12}
    s: {offset_xyz: [-0.029, -0.035, 0.0], z_offset: 0.074, hover_height: 0.12}

hand:
  open:  [185, 80, 255, 255, 255, 255]
  close: {left: [20, 0, 0, 0, 0, 0], right: [0, 0, 0, 0, 0, 0]}
  sizes:
    s: {left: [20, 0, 40, 40, 40, 40]}
```

手部通道顺序（现场实测）：`[拇指弯曲, 拇指侧摆, 食指, 中指, 无名指, 小指]`，
**255 = 全张，0 = 全闭**（与 URDF 里关节名的排列相反）。

### 3.3 白螺母

白螺母与大黑螺母**形状完全相同**，所以不单独录段、不单独调参，而是用一层
「识别标签 → 物理尺寸档」别名（`nut_robot.SHAPE_OF`）：

```python
SIZE_LABELS  = ('l', 'm', 's')                    # 物理尺寸档
SHAPE_OF     = {'l':'l', 'm':'m', 's':'s', 'white':'l'}
ORDER_LABELS = ('l', 'm', 's', 'white')           # 允许出现在 order 里
```

`white` 在画面/日志里仍显示「白」，但它走的 `right.approach`、`right.place`、
`grasp_by_size`、`hand.sizes`、抓取姿态**全是 `l` 档的对象**（同一个 Leg 对象，不是复制）。

- 需要**单独**给白螺母调偏移/手型时，直接在对应配置里加 `white:` 键覆盖即可，例如
  `hand.sizes.white: {left: [...]}` 或 `motion.grasp_by_size.white: {...}`；不写就自动跟 `l`。
- 任务一（入盒）**不需要**白螺母时，把 `order` 里的 `white` 删掉就行。

---

## 4. 现场标定值（截至 2026-09-16）

| 项目 | 位置 | 现在 | 相对录制原值 |
|---|---|---|---|
| 抓取顺序 | `order` | `[white, l, m, s]` | — |
| 左手指取高度 l / white | `grasp_by_size.l.z_offset` | `87.0mm` | 90 → **-3mm** |
| 左手指取高度 m | `grasp_by_size.m.z_offset` | `82.0mm` | 80 → **+2mm** |
| 左手指取高度 s | `grasp_by_size.s.z_offset` | `74.0mm` | 72 → **+2mm** |
| 左手拇指弯曲（所有螺母） | `hand.close.left[0]` / `hand.sizes.s.left[0]` | `20` | 0 → **+20** |
| 左手小螺母四指 | `hand.sizes.s.left[2:6]` | `40` | 未变 |
| 右手中央重抓点 | `recordings/right_middle_grasp1/…` `right_grasp_middle1_001` pt2 | `y = -42.9mm` | -22.9 → **-2cm** |
| 叠针释放点 | `needle.release_offset_xyz` | `[-75, -77, 0]mm` | 基准 0 → **x-75 / y-77** |

### 怎么改这些值

**笛卡尔路点**（右手重抓点、入盒点、ready 点等）用 `retune_waypoint.py`，
走驱动 IK + FK 复核，**只改录制文件，不动臂**：

```bash
cd /home/undefined/lbot_ws-2
source /opt/ros/jazzy/setup.bash
source /home/undefined/cleverhand/lbot_ws/install/setup.bash

python3 tools/retune_waypoint.py \
    --file recordings/right_middle_grasp1/events.jsonl \
    --sequence right_grasp_middle1_001 \
    --arm right --point 3 --dy -0.02
```

`--point` 是 **1 起**（pt0→1、pt1→2、pt2→3）。自动备份 `events.jsonl.bak_年月日_时分秒`，
输出里会打 IK 成功数、FK 复核、位置/姿态误差、逐关节变化。

**手型 / 偏移 / 释放点**直接改 yaml（见第 3 节），改完跑一次 dry-run 核对：

```bash
python3 tools/nut_pick_place.py --detector input --place-at needle
```

（`--detector input` 会提示手动输入四颗螺母位置，然后只打印计划，不动机器人。）

---

## 5. 踩过的坑（按"卡住顺序"）

### 5.1 左臂中转横移报「需要换臂型」

**症状**
```
[中止] 白螺母 中转平面横移：第 9/10 段需要换臂型（最大单关节变化 150.7° > 60°）
```

**根因（三层）**

1. 横移原来是 `steps=1`，25cm 一趟 MoveJ 走完 —— 数值 IK 会在中途跳到另一个臂型
   分支。下降/上升段早就分段了，**只有横移漏了**。
2. `resolve_transit_z` 最初只问"这个点能不能解出逆解"。探针证明某颗螺母的目标点
   **只有"抓取区那个臂型"才解得开**（差 130~197°），于是放行了 -190mm。
3. 改成"解出来的臂型离当前不超过阈值"后还是挂 —— 端点近了，但直臂横移 25cm 会在
   **中途**穿过奇异位形。实测同一颗螺母：

   | 中转平面 | 横移最大单关节变化 |
   |---|---|
   | -190mm | **156.8°** ✗ |
   | -215mm | 6.7° ✓ |
   | -240mm | 11.8° ✓ |

**现在的实现**

- 横移按 25mm 分段（`move_via_ik(..., steps=t_steps)`）。
- `transit_path_ok()` 把「上升 + 每一段横移」**整条路径**按 `move_via_ik` 将要用的
  **同一套算法**预演一遍（`plan_via_ik` 被两边共用，预检和执行不会漂移），任何一段
  跳分支就降一档高度重来。
- 单关节差按 2π 归一化：数值 IK 会把同一个物理角度给成 `+234°` / `-126°`，
  不归一化会把 360° 环绕误判成换臂型（实测 `Left_Elbow_Pitch` 原始差 196.7°，
  真实只差 163.3°）。

**排查入口**：`tools/probe_ik_point.py`（见第 6 节），或直接看日志里那行
`中转平面 ... 横移 XXXmm/N 段最大差 X°`。

### 5.2 右臂叠针第一步报「需要换臂型 73.9°」

1. **IK 种子给错了臂**：`run_one_nut` 里 `ik_seed_bank(left_legs)` 是**左臂**录段的
   关节角，却被传给了右臂的 `place_at_needle`。数值 IK 拿另一只臂的臂型当种子会收敛
   到别的分支。→ 新增 `right_ik_seed_bank(apprs, places)`，用右臂自己录段的
   12 个真实臂型。
2. **第一步同时"上升 + 大角度换向"**：原来全程用入盒段末点姿态 `[-70,-3,76]°`，
   而右臂刚重抓完是 `[-50,0,127]°`，第一步要边升 212mm 边拧腕 ~50°。→ 默认改成
   **当前抓握姿态**，可配 `needle.release_euler_deg` 覆盖。

### 5.3 铁针识别到一半整个任务崩了：`KeyError: 'u'`

`needle_pose_best.pt` 是 **Pose 模型**，记录里没有检测框中心 `u/v`，只有 `tip`/`base`
关键点。而 `nut_yolo.annotate()` 是按普通检测框写的。→ 现在 `annotate` 支持 Pose 记录
（画两个关键点十字），并且 `DetectionWindow.result()` 把标注包了 `try/except`：
**画框出错降级成原图，绝不再拖垮识别/抓取流程。**

### 5.4 `KeyError: 'white'`

四类模型（`nut_white_best.pt`）输出 `white` 标签，而框架只认 `l/m/s`。
→ 加入 `SHAPE_OF` 别名机制（见 3.3），`nut_detectors.normalize()` 也放行 `white`。

### 5.5 左手「张手了但螺母被原样带回」

日志里 `段尾张手释放（[185,80,255,255,255,255]）` 打了，手指却没动。
话题不锁存，丢一包手就一直攥着。→ `RobotClient.hand_open()` 默认**连发 3 轮**
（间隔 0.25s）。仍复现的话用 `tools/test_hand_release.py` 单独验手（见第 6 节）。

### 5.6 图像时间戳不同步

`time_domain:=system` 必带，否则彩色/深度时间戳差一个恒定偏移。

### 5.7 驱动段错误 `lbot_create_ik_request+0x50`

`InverseKinematics` 的 `joints` 传空数组会让驱动把 `nullptr` 交给厂商 SDK，
**直接段错误、整个驱动进程死掉**。→ `ik_try` 系列绝不发空种子，有专门的回归测试。

### 5.8 驱动"服务返回成功但机构一点没动"

出现过一次：`接入 left-ready_001 起点 服务完成但反馈停在容差外（残差已不降，
最好 1.442rad > 0.1）`。残差 = 目标值本身，说明**一个关节都没动**。
`check_names` 与 `enable()` 都通过了，所以不是软件问题。
→ **重启驱动 / 给机械臂断电重启**后恢复。判断方法：`--execute --go-ready` 单跑这一步。

### 5.9 细杆深度不可靠

针很细，关键点处深度可能取到背景或底座。→ 默认 `keypoint: base`（根部，更实）；
不行就设 `fallback_depth`（固定相机系 Z，米）兜底。

### 5.10 铁针只识别一次

第一个螺母叠上去之后，针的外观就变了，每次重检只会越来越不准 —— 这是**有意的设计**。
如果发现第 2 颗开始越叠越偏，先怀疑这个。

---

## 6. 诊断工具

### `probe_ik_point.py` — 离线多种子 IK 探针（**不动臂**）

`move_via_ik` 报"需要换臂型"时，先看清这个点到底有没有解、有几个分支、差在哪个关节、
差额是不是 360° 环绕造成的假象：

```bash
cd /home/undefined/lbot_ws-2/tools
source /opt/ros/jazzy/setup.bash
source /home/undefined/cleverhand/lbot_ws/install/setup.bash

python3 probe_ik_point.py --arm left --xyz 254.7,282.4,-190 --eul 40,-6.7,-95
```

### `test_hand_release.py` — 单独测灵巧手张/合（**不动臂、不跑视觉**）

排查"张手不放 / 闭合夹不住"：

```bash
python3 tools/test_hand_release.py --arm left --hold 2
```

每组值前按一次回车。第 1 组【张开】时把螺母塞进手指 → 回车闭合 → 回车张开，
看螺母掉不掉。三种结果对应三种病根：

| 现象 | 病根 |
|---|---|
| 手指张开、螺母掉 | 手本身没问题，是丢包/时序（把 `hand_open` 的 `repeat` 调大） |
| 手指完全不动 | 命令没到手上或手在保护状态（查 `ros2 topic info /robot1/left_hand/set_l6_joint -v`；试 `--force 30,30,30,30,30,30`） |
| 手指张开、螺母还挂着 | 夹持值问题（给该尺寸单独配 `hand.sizes`） |

### `nut_yolo_live.py` — 实时识别预览（**不碰机械臂**）

单独调铁针识别、完全不动机器人：

```bash
cd /home/undefined/lbot_ws-2
python3 tools/nut_yolo_live.py --model weights/needle_pose_best.pt
```

右边面板实时打 `tip: (u,v) px` / `base: (u,v) px`。哪个点稳定、落在针上，
就把 `needle.keypoint` 设成哪个。

### 监看手部指令

```bash
source /opt/ros/jazzy/setup.bash
source /home/undefined/cleverhand/lbot_ws/install/setup.bash
ros2 topic echo /robot1/left_hand/set_l6_joint
```

张手时应该刷出 `data: [185, 80, 255, 255, 255, 255]`。刷了但手指不动 → 手侧问题；
没刷 → 软件问题。

### `retune_waypoint.py` — 微调录段路点（**不动臂**）

见 4.1。

---

## 7. 故障排查表

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 开机第 0 步就「服务完成但反馈停在容差外」，残差 ≈ 目标值 | 驱动/伺服没真动 | 重启驱动；不行给机械臂断电重启；用 `--execute --go-ready` 单验 |
| `中转平面横移：第 N/M 段需要换臂型` | 该高度下横移穿奇异位形 | 已自动降高度。若连下限都找不到，在 `motion.transit_z` 写死更低的值（如 `-0.24`） |
| `上升让针：第 1/2 段需要换臂型` | 释放姿态要换向 | 确认 `needle.release_euler_deg: null`；或把 `max_joint_diff_deg` 放宽到 90（确认路径无干涉） |
| `上升让针：多种子逆解全部失败` | 针位不对/不可达 | 看 `铁针 base: ... -> base_link [...]` 那行；换 `keypoint`；必要时 `motion.transit_z` 降高度 |
| `没检测到铁针（needle Pose 模型无输出）` | 针不在视野/太暗/模型配错 | 用 `nut_yolo_live.py --model weights/needle_pose_best.pt` 单独看 |
| `铁针关键点 base 无效或置信度不足` | 关键点越界/低置信 | 降 `needle.min_confidence` 到 0.3，或换 `keypoint: tip` |
| 左手张手但螺母没掉 | 丢包 / 夹持值 | 见 5.5 与第 6 节 |
| `重新识别后找不到 X 螺母` | 前一颗碰动了它 | 正常保护；重跑。想避免就开 `redetect_each_nut: true` |
| `缺少螺母检测结果` | 识别漏检 | `require_all: false`（当前）会跳过；调 `confidence` / 光照 |
| 检测点 z 比桌面低 20mm+ | 深度读矮了 | 用 `z_offset` 补；连续两次差 5mm 说明深度有噪声 |

判断"是不是软件问题"的快速入口：**`--execute --go-ready`**。
它只做使能 + 张手 + 双臂回 ready，不含视觉、不含任务二逻辑。这一步过不去就是硬件/驱动。

---

## 8. 文件

| 文件 | 作用 |
|---|---|
| `tools/nut_pick_place.py` | 主流程（双臂交接 + 任务一/二分支） |
| `tools/nut_needle.py` | **任务二核心**：铁针识别、释放点、右臂叠放动作 |
| `tools/nut_robot.py` | 配置解析（`TaskConfig`）、机械臂/手 ROS 客户端、`SHAPE_OF` 标签别名 |
| `tools/nut_sequences.py` | 录段加载与回放（`SequenceRunner`） |
| `tools/nut_yolo.py` | YOLO 检测器 + 识别弹窗（含 Pose 记录标注） |
| `tools/nut_yolo_infer.py` | 推理子进程（隔离 venv，`--allow-pose` 放行铁针模型） |
| `tools/nut_detectors.py` | manual / json / external / input 四种检测器 |
| `tools/retune_waypoint.py` | 路点微调（IK+FK 复核，不动臂） |
| `tools/probe_ik_point.py` | **新增**：离线多种子 IK 探针 |
| `tools/test_hand_release.py` | **新增**：单独测灵巧手张/合 |
| `开发资源/nut_sort/nut_task.yaml` | 任务配置（顺序/抓取/手型/铁针） |
| `weights/needle_pose_best.pt` | 铁针两关键点 Pose 模型 |
| `weights/nut_white_best.pt` | 四类螺母模型（large/medium/small/white） |

---

## 9. 测试

```bash
cd /home/undefined/lbot_ws-2/tools

python3 test_nut_task.py     # 99 项：配置解析/段加载/视觉点/IK 策略/检测器/白螺母别名
python3 test_nut_yolo.py     # 24 项：反投影/深度采样/标注（含铁针 Pose 记录）
python3 test_needle_live.py  #  2 项：铁针关键点序列化与渲染
```

三个都用 `unittest`，**不需要 ROS、不需要机器人、不需要相机**。

---

## 10. 安全

- 默认 dry-run，`--execute` 才动真机。
- **所有运动前先算后动**：`move_via_ik` 两阶段（先把所有段 IK 解完并核对换臂型幅度，
  全部通过才开始执行），任何一段失败都在**臂未移动**的状态下中止。
- 换臂型保护默认 60° 单关节变化，超过拒绝"抡臂"。
- 回放中另一只臂漂移超过 `other_tolerance` 立即中止（防干涉）。
- 异常后**不自动掉使能**，在途运动需现场确认。
- 手上一定是先张手再动臂；臂运动前手部动作都带静置时间。
