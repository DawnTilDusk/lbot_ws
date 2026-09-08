# 双臂工作区域轨迹记录

脚本仅订阅关节/位姿反馈，不发送运动、使能、掉使能或标零请求。掉使能不等于自由拖动或重力补偿；实物示教前确认抱闸和支撑方式，避免其他客户端同时下发运动。

## 启动

保持驱动运行，退出 Conda，在另一个 zsh 终端执行：

```zsh
cd ~/lbot_ws
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
/usr/bin/python3 tools/record_workpoints.py
```

## 新操作流程

1. 等待状态就绪。第一次直接按回车，保存起点并开始采样。
2. 手动移动机械臂。第二次直接按回车，立即停止采样并固定终点。
3. 输入名称后回车，例如 `lift`，保存为 `lift_001`。
4. 再按回车开始下一段，回车停止，再命名，例如 `place_002`。
5. `:q` 回车或 Ctrl+C 退出，不改变机械臂使能状态。

序号代表本次脚本运行中开始的第几条记录，每次启动从 001 开始。每次运行有独立时间目录，不覆盖之前记录。名称可以重复，但追加序号后的标签唯一。名称不能为空。输入 `:q` 保留为退出命令。

等待开始和命名期间不采样。命名时即使机械臂移动，也不会改变第二次回车时已经固定的终点。开始时反馈无效会拒绝；停止时反馈无效仍立即停止并允许命名保存，但标记无效，回放拒绝使用。中途无效采样也会保留。

默认 10 Hz 采样，调整示例：

```zsh
/usr/bin/python3 tools/record_workpoints.py --rate 20 --output ~/lbot_ws/recordings
```

## 文件格式

每次运行创建 `recordings/日期_时间_微秒/events.jsonl`，一行一个 JSON 对象。新格式 `schema_version: 2`：

| type | 含义 |
| --- | --- |
| metadata | 采样参数和限制 |
| waypoint，role=start | 第一次回车的起点，例如 record_001_start |
| sample | 记录期间采样，含 record_id 和全局 sample_id |
| record_stop | 第二次回车的终点快照，在命名前立即落盘 |
| waypoint，role=end | 命名后的终点，例如 record_001_end，label=lift_001 |
| end | 总数、未结束或未命名的记录编号和状态 |

终点的 `segment.from_point_id` 指向这一条的起点；`first_sample_id` 到 `last_sample_id`（含两端）标识其中的采样，无采样时为 null。`invalid_samples` 记录中途无效样本数量。正常回放只选择已命名的结束点，自动匹配对应起点。

退出时没有停止的记录、已经停止但未命名的记录仍保留原始数据，不伪装成已完成记录。点位文件持续 flush，起点、终点、命名、正常退出时 fsync。突然断电仍可能丢失未同步尾部或留下不完整末行。

## 记录范围

保存双臂关节位置/速度/力矩、末端位置/四元数、原始 header、frame_id 和本地接收年龄。关节位置 rad，末端位置 m，速度/力矩原样保存；非有限数值为 null，并标记无效。

当前驱动没有发布使能、急停、故障、温度、灵巧手反馈和工具配置，不能记录这些内部状态。四个话题不是硬件同步快照；接收超时和接收时差可检测，但驱动给缓存状态重打时间戳，不能可靠识别底层反馈卡住。坐标系原样保留，不能假定 base_link 等于模型 base_torso_root。

这是示教轨迹记录，不是零点标定、碰撞检查或安全工作区域证明。

## 离线测试

```zsh
source /opt/ros/jazzy/setup.zsh
cd ~/lbot_ws/tools
/usr/bin/python3 -m unittest -v test_record_workpoints.py test_replay_workpoints.py
```
