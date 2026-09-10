#!/usr/bin/env python3
"""视觉联调用【固定参考位姿】外部检测器桩（2026-09-10 用户提供）。

参考值（视觉模块将传入的参数格式）：
  位置  : 0.378, 0.326, -0.348   m，base_link 系（机械臂坐标，非相机系）
  欧拉角: 0.924, 0.000, -1.506   rad（约 52.9, 0, -86.3 度，与四元数自洽）
  四元数: 0.325, -0.305, -0.612, 0.653（xyzw）

当前任务设计：抓取旋转角固定取采集的 left_grasp_init，视觉给的欧拉/四元数只记录、
不参与运动；位置直接作为 base_link 目标点（框架跳过手眼变换，hover/down 在其上加减 z）。

用法：
  nut_task.yaml 里 detector.external 指向本文件后：
  /usr/bin/python3 tools/nut_pick_place.py --detector external --order l   # 先只测大螺母
真要测三颗时，在 REF_POSES 里照样子补 m/s 行并返回。
"""
import numpy as np

from nut_detectors import Detection

REF_POSES = {
    'l': {
        'position': [0.378, 0.326, -0.348],
        'euler_rad': [0.924, 0.000, -1.506],
        'quat_xyzw': [0.325, -0.305, -0.612, 0.653],
    },
    # 'm': {'position': [...], 'euler_rad': [...], 'quat_xyzw': [...]},
    # 's': {...},
}


class RefNutDetector:
    def __init__(self, node, sub_cfg, K):
        self.node = node
        self.cfg = sub_cfg
        self.K = K

    def detect(self, expected):
        out = []
        for label in expected:
            ref = REF_POSES.get(label)
            if ref is None:
                continue
            out.append(Detection(
                label,
                np.array(ref['position'], float),
                extra={'frame': 'base_link',
                       'euler_rad': ref['euler_rad'],
                       'quat_xyzw': ref['quat_xyzw']}))
        return out
