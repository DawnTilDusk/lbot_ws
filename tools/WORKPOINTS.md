# 手动点动作序列记录

每按一次回车记录一个双臂状态点；不自动采样，不插入中间点。

## 启动

保持驱动运行，退出 Conda，在第二个终端执行：

```zsh
cd ~/lbot_ws
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
/usr/bin/python3 tools/record_workpoints.py --arm right
```

`--arm left` 用于生成左臂回放命令；记录内容始终包含双臂。默认 right。

## 操作

1. 等待状态就绪，将机械臂摆到第一个点，按回车保存。
2. 移到下一点，按回车保存。重复直到所有必要点已记录。
3. 直接按 `q`（不用回车），停止本条动作序列，进入命名。
4. 输入名称再回车，例如 `lift`，保存为 `lift_001`。
5. 终端自动显示两条完整命令：预览命令和实际执行命令。包含绝对路径、准确的序列名称和 `--move-to-start`；执行命令另含 `--execute`。
6. 可以继续回车记录下一条序列，命名后得到 `place_002` 等；Ctrl+C 退出整个程序。

序号按本次脚本运行的已命名序列递增，每次启动从 001 开始。新运行使用独立时间目录。命名期间不记录点；空名称会要求重输。没有点时 q 不会进入命名。一个点也可保存，执行命令会用起点接近功能到达它。

脚本持续接收状态，但只有按回车时才写入点位。状态缺失、过期或数值异常会拒绝本次回车，不消耗编号。停止后、命名期间的移动不会改变已记录的点。

执行命令会实际运动，需要另外确认使能、支撑和点间路径。脚本不会自动执行打印出的命令，也不会自动使能。没有碰撞规划；没有记录你在两个点之间的手动绕行，必要时增加避障点。

## 文件

`recordings/日期_时间_微秒/events.jsonl` 为逐行 JSON，schema_version=3。

- metadata：命名空间、手动记录模式、命令所选机械臂等。
- waypoint：每次有效回车一个点，包含唯一 point_id、elapsed_seconds、双臂四话题的状态。
- sequence_stop：q 时固定本序列 point_ids。
- sequence：命名后的 sequence_id、label、base_label 和按顺序排列的 point_ids。
- end：命名序列数量、点位数量、未命名点编号。

没有 sample 事件。每个事件立即 flush 和 fsync；退出前未命名的点仍保留，但不会作为已命名序列回放。突然断电可能留下不完整末行。

## 数据限制

保留关节角(rad)、速度/力矩原始数值、末端位置(m)/四元数、话题 header 和 frame_id、本地接收年龄。四个话题不是硬件同步快照。当前驱动未提供使能、急停、故障、温度、手指反馈和工具配置。驱动重新给缓存反馈打时间戳，所以接收及时不保证硬件反馈未停滞；不能假定 base_link 与模型 base_torso_root 相同。

掉使能不等于自由拖动或重力补偿，实物示教须先确认抱闸及支撑方式。本工具不是零位标定或安全区域验证。

## 测试

```zsh
source /opt/ros/jazzy/setup.zsh
cd ~/lbot_ws/tools
/usr/bin/python3 -m unittest -v test_record_workpoints.py test_replay_workpoints.py
```
