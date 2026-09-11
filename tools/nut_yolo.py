#!/usr/bin/env python3
"""YOLO + 配对的彩色/对齐深度帧，返回主业务所需的 Detection(p_cam)。"""
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from nut_robot import TaskError
from nut_detectors import Detection
from camera_pick_move import sample_depth, deproject

WORKSPACE = Path(__file__).resolve().parents[1]


def infer_pixels(image, cfg):
    import cv2
    model = Path(cfg.get('model', 'weights/nut_best.pt')).expanduser()
    if not model.is_absolute():
        model = WORKSPACE / model
    python = Path(cfg.get('python', str(WORKSPACE / '.venv-yolo/bin/python'))).expanduser()
    if not model.is_file() or not python.is_file():
        raise TaskError(f'找不到 YOLO 模型或解释器：{model}，{python}')
    with tempfile.TemporaryDirectory(prefix='nut_yolo_') as folder:
        image_path = Path(folder)/'color.png'
        result_path = Path(folder)/'detections.json'
        if not cv2.imwrite(str(image_path), image):
            raise TaskError('无法写入 YOLO 输入图片')
        cmd = [str(python), str(WORKSPACE/'tools/nut_yolo_infer.py'),
               '--model', str(model), '--image', str(image_path), '--output', str(result_path),
               '--conf', str(cfg.get('confidence', .5)), '--imgsz', str(cfg.get('imgsz', 640)),
               '--device', str(cfg.get('device', 'cpu'))]
        env = os.environ.copy()
        # ROS PYTHONPATH 含系统二进制扩展，不传进 Conda。
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env['PYTHONNOUSERSITE'] = '1'
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                    timeout=float(cfg.get('inference_timeout', 30)))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TaskError(f'YOLO 推理失败：{exc}') from exc
        if result.returncode:
            raise TaskError(f'YOLO 推理失败：{result.stderr[-2000:]}')
        return json.loads(result_path.read_text())


def locate(records, depth, K, image_shape, cfg):
    """框中心邻域深度 -> 相机光学坐标（米），复用现有点选数学链路。"""
    if depth.ndim != 2 or depth.shape != tuple(image_shape[:2]):
        raise TaskError('彩色和深度尺寸不一致：需要与彩色同分辨率的对齐深度，不能直接缩放替代对齐')
    if depth.dtype not in (np.dtype('uint16'), np.dtype('float32'), np.dtype('float64')):
        raise TaskError('深度必须为 uint16 毫米或浮点米，不能使用深度伪彩图')
    K = np.asarray(K, float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise TaskError('无效相机内参')
    roi = cfg.get('roi')  # 原图像素 [xmin, ymin, xmax, ymax]
    out = []
    for record in records:
        u, v = float(record['u']), float(record['v'])
        if not np.isfinite([u,v]).all() or not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            raise TaskError('检测中心超出图像范围')
        if roi and not (roi[0] <= u < roi[2] and roi[1] <= v < roi[3]):
            continue
        ui, vi = int(round(u)), int(round(v))
        ui, vi = min(ui, depth.shape[1]-1), min(vi, depth.shape[0]-1)
        z = sample_depth(depth, ui, vi)
        if z is None:
            raise TaskError(f'{record["label"]} 中心 ({u:.1f},{v:.1f}) 深度无效，中止本次检测')
        out.append(Detection(record['label'], deproject(u, v, z, K), u=u, v=v, z=z,
                             extra=dict(confidence=record['confidence'], bbox=record['bbox'],
                                        depth_method='center_patch_median')))
    return out


class YoloDetector:
    def __init__(self, node, sub_cfg, K=None):
        self.node = node
        self.cfg = sub_cfg
        self.snapshot = None

    def detect(self, expected):
        if self.node is None:
            raise TaskError('YOLO 需要实时彩色/深度；使用 nut_yolo_preview.py，或主流程 --detector yolo')
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo
        from cv_bridge import CvBridge
        bridge = CvBridge()
        color, depth = deque(maxlen=15), deque(maxlen=15)
        camera = {}
        vision = self.cfg['_vision']
        max_age = float(self.cfg.get('max_frame_age', 1.))
        skew = float(self.cfg.get('max_skew', .1))

        def receive(msg, target, encoding):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
            if stamp <= 0:
                return
            try:
                frame = bridge.imgmsg_to_cv2(msg, encoding).copy()
                target.append((stamp, time.monotonic(), frame))
            except Exception as exc:
                self.node.get_logger().warning(f'图像解码失败：{exc}')

        def info(msg):
            camera.update(K=np.array(msg.k).reshape(3,3), width=msg.width, height=msg.height,
                          frame_id=msg.header.frame_id)

        subs = [self.node.create_subscription(Image, vision['color_topic'],
                                             lambda m: receive(m, color, 'bgr8'), qos_profile_sensor_data),
                self.node.create_subscription(Image, vision['depth_topic'],
                                             lambda m: receive(m, depth, 'passthrough'), qos_profile_sensor_data),
                self.node.create_subscription(CameraInfo, vision['color_info_topic'], info, qos_profile_sensor_data)]
        try:
            end = time.monotonic()+float(self.cfg.get('camera_timeout', 10))
            pair = None
            while time.monotonic() < end:
                rclpy.spin_once(self.node, timeout_sec=.05)
                now = time.monotonic()
                pairs = [(c,d) for c in color for d in depth
                         if now-c[1] <= max_age and now-d[1] <= max_age and abs(c[0]-d[0]) <= skew]
                if pairs and camera:
                    pair = min(pairs, key=lambda x: abs(x[0][0]-x[1][0]))
                    break
            if pair is None:
                raise TaskError('等待彩色、对齐深度、彩色 camera_info 超时，或图像时间戳不同步')
            c,d = pair
            if (camera['height'],camera['width']) != c[2].shape[:2]:
                raise TaskError('camera_info 与彩色分辨率不匹配')
            from camera_pick_move import load_extrinsics
            _, _, ext = load_extrinsics(Path(vision['extrinsics_path']))
            if ext.get('parent_frame') != 'base_link' or ext.get('child_frame') != camera['frame_id']:
                raise TaskError('外参坐标系与实时彩色相机不匹配，不能直接交给 base_link 抓取流程')
            records = infer_pixels(c[2], self.cfg)
            if time.monotonic()-min(c[1],d[1]) > float(self.cfg.get('max_result_age', 10)):
                raise TaskError('推理耗时过长，抓取快照已过期')
            detections = locate(records, d[2], camera['K'], c[2].shape, self.cfg)
            self.snapshot = dict(color=c[2], depth=d[2], K=camera['K'],
                                 color_stamp=c[0], depth_stamp=d[0], frame_id=camera['frame_id'])
            return detections
        finally:
            for sub in subs:
                self.node.destroy_subscription(sub)


def detect_once(cfg):
    """独立的相机只读入口；没有机器人服务客户端。"""
    import rclpy
    from nut_detectors import build_detector
    rclpy.init()
    node = rclpy.create_node('nut_yolo_preview')
    try:
        detector = build_detector(cfg, node)
        detections = detector.detect(cfg.order)
        return detections, detector.snapshot
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
