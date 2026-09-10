#!/usr/bin/env python3
"""自接螺母检测器模板（识别算法只输出画面像素位置即可）。

接入：
  1. 把本文件拷到任意位置（建议 开发资源/nut_sort/ 下），改 detect() 内部实现；
  2. nut_task.yaml 里 detector.type=external，detector.external 填绝对路径:类名，如
       external: '/home/lionheart/.../nut_sort/my_detector.py:MyNutDetector'
  3. --execute 时任务在【双臂回到 home、离开画面后】调用 detect(expected)。

契约：
  - 构造函数签名固定为 __init__(self, node, sub_cfg, K)
      node    : rclpy 节点（nut_pick_place），可直接 create_subscription 取相机画面；
                节点已在 spin，你只需注册订阅、在 detect 里取最新消息
      sub_cfg : nut_task.yaml 的 detector 段原始 dict（可加自己的配置键），
                sub_cfg['_vision'] 含 color/depth/camera_info 话题与内参文件路径
      K       : 彩色内参 3x3 np.ndarray（可能为 None，话题来了也可以自己订阅）
  - detect(expected) 同步阻塞，expected 如 ('l','m','s')；返回 Detection 列表
    （也接受等价 dict）。每颗螺母三种给法任选：
      Detection('l', p_cam=[X,Y,Z])              你自己算好相机光学系 3D（米）
      Detection('l', None, u=320, v=240, z=0.7)  像素 + 对齐深度（米），框架用 K 反投影
      Detection('l', None, u=320, v=240)         ★ 只给像素：真机运行时框架自动订阅
                                                 对齐深度、在该像素附近取样并反投影
    标签固定 'l'大 / 'm'中 / 's'小；看到几颗返回几颗，多检/缺检由框架按
    require_all 处理（同尺寸返回多颗会中止）。
  - 不要在 detect 里发任何运动指令；用户中止可抛 KeyboardInterrupt。
"""
import numpy as np

from nut_detectors import Detection


class MyNutDetector:
    def __init__(self, node, sub_cfg, K):
        self.node = node
        self.cfg = sub_cfg
        self.K = K
        # 订阅彩色画面（深度不用订——只返回像素时框架自动处理深度）：
        # from sensor_msgs.msg import Image
        # from rclpy.qos import qos_profile_sensor_data
        # from cv_bridge import CvBridge
        # self.bridge = CvBridge()
        # self.color = None
        # node.create_subscription(Image, sub_cfg['_vision']['color_topic'],
        #                          self._img, qos_profile_sensor_data)

    def detect(self, expected):
        """对最新一帧跑你的识别，返回看到的螺母。下面是假数据模板。"""
        out = []
        for i, label in enumerate(expected):
            # TODO: 你的识别 -> 螺母中心像素 (u, v)，并按大小分到 l/m/s
            u, v = 320 + 40 * i, 240
            out.append(Detection(label, None, u=u, v=v))
        return out
