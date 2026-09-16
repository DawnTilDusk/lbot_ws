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

WINDOW_NAME = 'nut YOLO detect (q/ESC=abort, space/enter=continue)'


def _put_text(img, text, org, scale=.6, color=(0, 255, 255), thickness=2):
    """putText 加黑边，彩图上可读；只写 ASCII（OpenCV 不含中文字库）。"""
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def annotate(color, records, located=None, failed_label=None, status=None):
    """彩色帧上画 YOLO 检测框；located（Detection 列表）补中心深度。

    records：nut_yolo_infer 的输出字典列表（label/confidence/bbox/u/v）。
    failed_label：该尺寸框中心深度无效时画红框。返回新图像，不改原图。
    """
    import cv2
    canvas = color.copy()
    by_center = {}
    for det in (located or []):
        by_center[(int(round(det.u)), int(round(det.v)))] = det
    for r in records:
        x1, y1, x2, y2 = [int(round(float(v))) for v in r['bbox']]
        label = str(r['label'])
        bad = label == failed_label
        box_color = (0, 0, 255) if bad else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        if r.get('u') is None or r.get('v') is None:
            # Pose 模型（铁针）：没有框中心，只有 tip/base 关键点
            kps = [k for k in (r.get('keypoints') or [])
                   if k.get('u') is not None and k.get('v') is not None]
            for k in kps:
                ku, kv = int(round(float(k['u']))), int(round(float(k['v'])))
                kc = (0, 255, 255) if str(k.get('name')) == 'tip' else (255, 0, 255)
                cv2.drawMarker(canvas, (ku, kv), kc, cv2.MARKER_CROSS, 18, 2)
                _put_text(canvas, f"{k.get('name')} "
                                  f"{float(k.get('confidence') or 0.):.2f} ({ku},{kv})",
                          (ku + 8, max(18, kv - 8)), scale=.5, color=kc)
            text = f"{label} {float(r['confidence']):.2f}"
            if not kps:
                text += ' NO VALID KEYPOINT'
                box_color = (0, 0, 255)
            _put_text(canvas, text, (x1, max(18, y1 - 6)),
                      color=(0, 0, 255) if bad else (0, 255, 255))
            continue
        u, v = int(round(float(r['u']))), int(round(float(r['v'])))
        cv2.drawMarker(canvas, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        text = f"{label} {float(r['confidence']):.2f} ({u},{v})"
        det = by_center.get((u, v))
        if det is not None:
            text += f" z={det.z * 1000:.0f}mm"
        if bad:
            text += ' DEPTH INVALID'
        _put_text(canvas, text, (x1, max(18, y1 - 6)),
                  color=(0, 0, 255) if bad else (0, 255, 255))
    if status:
        _put_text(canvas, str(status), (8, 24), scale=.55, color=(255, 255, 255))
    return canvas


class DetectionWindow:
    """识别阶段的实时弹窗：等待帧期间显示实时画面，出结果后画检测框。

    开关 detector.show_window（yaml/--show）；结果停留 show_seconds 秒后自动继续
    （空格/回车立即继续，q/ESC 中止任务；show_seconds=0 则必须按键）。
    无显示环境（headless/namedWindow 失败）自动降级为无窗，不影响识别。
    """

    def __init__(self, cfg, log=print):
        self.enabled = bool(cfg.get('show_window', False))
        try:
            self.seconds = float(cfg.get('show_seconds', 2.0))
        except (TypeError, ValueError):
            self.enabled = False
            self.seconds = 2.0
        self.cv2 = None
        if not self.enabled:
            return
        if not os.environ.get('DISPLAY'):
            self.enabled = False
            log('未检测到 DISPLAY，跳过识别弹窗（其余流程正常）')
            return
        try:
            import cv2
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.cv2 = cv2
        except Exception as exc:  # 无 X/建窗失败：降级
            self.enabled = False
            log(f'识别弹窗不可用（{exc}），继续无窗流程')

    def _show(self, img, wait_ms=1):
        try:
            self.cv2.imshow(WINDOW_NAME, img)
            self.cv2.waitKey(wait_ms)
        except Exception:
            # 运行途中显示连接丢失：后续静默，不让弹窗影响识别本身
            self.enabled = False

    def live(self, frame, status='waiting synced color+depth pair ...'):
        if self.enabled:
            self._show(annotate(frame, [], status=status))

    def result(self, color, records, located=None, failed_label=None, status=None):
        if self.enabled:
            # 画框只是给人看的：标注本身出错也绝不能中断识别/抓取流程
            try:
                img = annotate(color, records, located, failed_label, status)
            except Exception as exc:  # noqa: BLE001
                img = color.copy()
                try:
                    _put_text(img, f'annotate error: {exc}', (8, 24),
                              scale=.55, color=(0, 0, 255))
                except Exception:  # noqa: BLE001
                    pass
            self._show(img)

    def dwell(self):
        """结果画面停留 show_seconds；q/ESC 中止，空格/回车立即放行。"""
        if not self.enabled:
            return
        end = time.monotonic() + max(0.0, self.seconds)
        while True:
            key = self.cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                raise TaskError('用户在识别窗口按 q/ESC，中止任务')
            if key in (32, 13, 10):
                return
            if self.seconds > 0 and time.monotonic() >= end:
                return

    def close(self):
        if self.cv2 is not None:
            try:
                self.cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass



def resolve_python(cfg):
    """Auto supports the team's workspace venv and this machine's Conda env."""
    configured = cfg.get('python', 'auto')
    if configured != 'auto':
        path = Path(configured).expanduser()
        return path if path.is_absolute() else WORKSPACE / path
    candidates = [WORKSPACE / '.venv-yolo/bin/python',
                  Path.home() / 'miniconda3/envs/nut-yolo/bin/python']
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise TaskError('未找到 YOLO Python；请在 detector.python 指定已安装 ultralytics 的解释器')


def infer_pixels(image, cfg):
    import cv2
    model = Path(cfg.get('model', 'weights/nut_best.pt')).expanduser()
    if not model.is_absolute():
        model = WORKSPACE / model
    python = resolve_python(cfg)
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
        if cfg.get('allow_pose'):
            # 铁针是 Pose 模型；nut_yolo_infer 默认拒绝，必须显式放行
            cmd.append('--allow-pose')
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


def resolve_records(cfg, wait_pair, run_infer, log=print):
    """「取一对最新同步帧 -> YOLO 推理」循环，快照过期或推理失败就重取重识别。

    wait_pair(timeout) -> pair=(color_tuple, depth_tuple) 或 None（取帧超时）；
      帧元组第 2 项是本机接收 monotonic，帧结构 (消息戳, monotonic, frame)。
    run_infer(color_frame) -> records（推理子进程失败可抛 TaskError，按重试处理）。
    每次重试前由 wait_pair 负责清空缓存，保证拿到的是本次推理后的新帧。
    重试次数 detector.inference_retries（缺省 3）；快照年龄上限 max_result_age 秒。
    """
    retries = max(1, int(cfg.get('inference_retries', 3)))
    max_result_age = float(cfg.get('max_result_age', 10))
    last_err = None
    for attempt in range(1, retries + 1):
        pair = wait_pair(float(cfg.get('camera_timeout', 10)))
        if pair is None:
            raise TaskError('等待彩色、对齐深度、彩色 camera_info 超时，或图像时间戳不同步')
        c, d = pair
        try:
            records = run_infer(c[2])
        except TaskError as exc:  # 推理子进程失败/超时：重新发识别指令
            last_err = exc
            print(f'  第 {attempt}/{retries} 次识别失败（{exc}），重新取帧识别...')
            continue
        age = time.monotonic() - min(c[1], d[1])
        if age <= max_result_age:
            return records, pair
        print(f'  第 {attempt}/{retries} 次推理完成但快照已过期'
              f'（帧龄 {age:.1f}s > {max_result_age:g}s，CPU 冷启动模型常见），'
              f'丢弃旧帧重新取帧识别...')
    detail = str(last_err) if last_err is not None else f'快照连续过期（>{max_result_age:g}s）'
    raise TaskError(f'YOLO 连续 {retries} 次识别未成功，中止（最后原因：{detail}）。'
                    f'可调大 detector.max_result_age / inference_timeout，或检查 CPU 负载')


class YoloDetector:
    def __init__(self, node, sub_cfg, K=None):
        self.node = node
        self.cfg = sub_cfg
        self.snapshot = None

    def detect(self, expected, raw=False):
        """expected 给定时只返回这些标签；raw=True 时不过 locate()，
        直接返回 (records, color, depth, camera) —— 给 pose 模型用：
        关键点记录没有 u/v，走 locate() 会 KeyError，选点逻辑交给调用方。
        """
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
        window = DetectionWindow(self.cfg)

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
            from camera_pick_move import load_extrinsics
            _, _, ext = load_extrinsics(Path(vision['extrinsics_path']))

            def wait_pair(timeout):
                # 清空缓存：上一次推理耗时可能很长，只接受本轮开始后收到的新帧
                color.clear(); depth.clear()
                end = time.monotonic()+timeout
                while time.monotonic() < end:
                    rclpy.spin_once(self.node, timeout_sec=.05)
                    if color:
                        window.live(color[-1][2])  # 等待/重试期间持续刷新实时画面
                    now = time.monotonic()
                    pairs = [(c,d) for c in color for d in depth
                             if now-c[1] <= max_age and now-d[1] <= max_age and abs(c[0]-d[0]) <= skew]
                    if pairs and camera:
                        return min(pairs, key=lambda x: abs(x[0][0]-x[1][0]))
                return None

            def run_infer(img):
                # 推理（CPU 冷启动可能十几秒）前给一帧带提示的画面，免得窗口像卡死
                window.result(img, [], status='YOLO inferring (CPU cold start may take ~10s) ...')
                return infer_pixels(img, self.cfg)

            records, (c, d) = resolve_records(self.cfg, wait_pair, run_infer)
            if (camera['height'],camera['width']) != c[2].shape[:2]:
                raise TaskError('camera_info 与彩色分辨率不匹配')
            if ext.get('parent_frame') != 'base_link' or ext.get('child_frame') != camera['frame_id']:
                raise TaskError('外参坐标系与实时彩色相机不匹配，不能直接交给 base_link 抓取流程')
            if raw:
                # pose 模型（铁针）：不做中心反投影，原样交给调用方
                self.snapshot = dict(color=c[2], depth=d[2], K=camera['K'],
                                     color_stamp=c[0], depth_stamp=d[0],
                                     frame_id=camera['frame_id'])
                window.result(c[2], records, status=f'{len(records)} record(s) (raw)')
                window.dwell()
                return records, c[2], d[2], camera
            window.result(c[2], records,
                          status=f'{len(records)} box(es), sampling aligned depth ...')
            try:
                detections = locate(records, d[2], camera['K'], c[2].shape, self.cfg)
            except TaskError as exc:
                failed_label = next((str(r['label']) for r in records
                                     if str(exc).startswith(f"{r['label']} ")), None)
                window.result(c[2], records, failed_label=failed_label,
                              status='depth invalid at marked box center (aborting)')
                window.dwell()
                raise
            # detect(expected) 约定：只返回 caller 要的尺寸
            # （与 JsonDetector / InputDetector 一致）。四类模型的 white 等
            # 非抓取标签在此丢弃，避免流进抓取流程触发 KeyError。
            if expected:
                dropped = sorted({str(d.label) for d in detections
                                   if d.label not in expected})
                detections = [d for d in detections if d.label in expected]
                if dropped:
                    self.node.get_logger().warning(
                        '已忽略非抓取目标（不在 %s 中）：%s'
                        % ('/'.join(map(str, expected)), '、'.join(dropped)))
            n = len(detections)
            window.result(c[2], records, located=detections,
                          status=f'{n} nut(s) localized — auto in {window.seconds:g}s, space=go, q=abort')
            window.dwell()
            self.snapshot = dict(color=c[2], depth=d[2], K=camera['K'],
                                 color_stamp=c[0], depth_stamp=d[0], frame_id=camera['frame_id'])
            return detections
        finally:
            window.close()
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
