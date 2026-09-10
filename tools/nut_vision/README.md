# 螺母 RGB-D 采集与回放工具

本目录用于现场补采、保存和离线回放。**不是已验收的自主抓取系统。**
检测预览仍有漏检、错分；没有训练完成的 YOLO 模型。几何实验版本未包含在此交接目录。

完整逐步说明见 [现场采集操作手册](docs/capture_guide.md)。手册中的克隆目录是示例；实际位置不同请相应调整。

## 安全边界

- 不直接发送机器人运动命令，但工作台会发布 `/nut_vision/detections`。
- 采集时断开控制端对该话题的自动动作消费；不能把绿色标注当成抓取安全证明。
- 回放使用与实机隔离的 ROS 域，不与实时相机、控制节点混用。
- 本工具只提供相机坐标，未进行相机到机器人外参标定。

## 环境

沿用 ROS 2 Jazzy、系统 Python 3、Orbbec Gemini 2 驱动环境。
Python 依赖为 OpenCV、NumPy、PyQt5、rclpy、cv_bridge、message_filters、
rosidl_runtime_py，以及标准 ROS 图像、相机信息、TF 等消息包。
此目录不是独立的 ament 包，无需为了运行它修改运动模块或执行 colcon build。
相机驱动必须已安装并构建。不要直接把它复制到 ROS 的 install 目录。

## 启动

先由现场人员启动一份相机驱动，确保输出以下四个话题：

```text
/camera/color/image_raw
/camera/depth/image_raw
/camera/color/camera_info
/camera/depth/camera_info
```

在团队仓库根目录执行：

```bash
bash tools/nut_vision/start_capture.sh --output "$HOME/nut_capture/team_batch01"
```

启动脚本只启动工作台，不启动相机或机器人。默认保存目录为 `$HOME/nut_capture`，
也可以用 `--output` 指定新的批次目录。启动脚本不依赖原开发者的主目录。

若 ROS 安装路径不同，可指定：

```bash
ROS_SETUP=/实际ROS安装目录/setup.bash \
  bash tools/nut_vision/start_capture.sh --output "$HOME/nut_capture/team_batch01"
```

## 现场操作

1. 确认 LIVE 和两路更新画面；ROI 使用 `0,0,1,1`。
2. 复核并保存实测尺寸。当前提供：大70/34、中50/26、小40/22 mm（对边宽/高度）。大银和大黑是否同高需现场确认。
3. 按手册导出相机节点参数到输出目录的 `camera_parameters_720p.yaml`。
4. 重新摆放后点“新场景”，选择train或validation，填写真实物体与采集条件。
5. 点“采集10对”，先回传一小组检查格式，再批量采集。
6. 不必等算法正确识别再保存；同步帧有效、原图清楚即可。

参数文件只有预先存在时才会复制到新场景。程序不会自动导出完整驱动参数。
配置中的高度可能仍为未知，现场录入后点“保存实测尺寸”；不要从算法预测反填真值。

## 保存格式

```text
输出目录/dataset/train或validation/场景ID/
  scene.json
  camera_driver_parameters.yaml  # 可选，取决于事先是否已导出
  秒_纳秒/
    color.png
    depth.npz
    metadata.json
```

这是抽样同步RGB-D样本，不是30fps录像。十对之间至少约0.5秒，实际可更长。
`depth.npz` 中数组键为 `depth`。元数据保存时间戳、内参、深度单位与哈希。
`detector_predictions` 是原型预测，不是人工标签；机器人观测不严格同步。

## 离线回放

在独立且未被实机占用的ROS域中运行，例如：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=77
/usr/bin/python3 tools/nut_vision/replay_dataset.py /实际路径/场景ID --rate 0.5 --loop
```

在同域的另一个终端启动工作台观看。回放保留历史时间戳，不发布 `/clock`，
也不回放机器人位姿或TF，不能作为实时控制输入。

## 提交范围

只包含采集/回放工具、配置和说明。没有原始数据、机器人控制代码改动、
新模型权重或几何实验模块。跨电脑启动与真实设备试采仍需现场确认。
