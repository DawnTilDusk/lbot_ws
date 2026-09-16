#!/usr/bin/env python3
"""螺母检测器接口与内置实现。

检测只负责回答「哪颗螺母（label）在相机光学系的什么位置 p_cam」，
手眼反算（内参反投影 + 外参变换到 base）由主流程统一完成。

返回元素统一为 Detection：
  label : 'l' / 'm' / 's'
  p_cam : [X, Y, Z] 米，camera_color_optical_frame（X 右 / Y 下 / Z 前）

内置：
  ManualClickDetector 画面点选（按提示依次点大/中/小），框架联调与现场兜底
  JsonDetector        从 json 文件读结果，离线 dry-run / 回归测试
  external            动态加载你自己的类（文件路径或可 import 模块 :类名）

接入自己的视觉见 NUT_TASK.md「接入自己的视觉」。
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nut_robot import SIZE_NAMES_CN, TaskError


class DetectorAbort(TaskError):
    """用户在检测窗口中止（q/ESC）。"""


@dataclass
class Detection:
    label: str
    p_cam: np.ndarray
    u: float = None
    v: float = None
    z: float = None
    extra: dict = field(default_factory=dict)


class DetectorBase:
    """自接视觉的最小接口。

    构造：__init__(self, node, sub_cfg: dict, K)
      node    : rclpy 节点（可自建订阅/服务）；json 等离线检测器可为 None
      sub_cfg : nut_task.yaml 里 detector 段的原始 dict
      K       : 标定内参 3x3（np.ndarray），无内参为 None
    方法：detect(expected: tuple[str, ...]) -> list[Detection]
      expected : 本次需要的尺寸，如 ('l','m','s')
    """

    def __init__(self, node=None, sub_cfg=None, K=None):
        self.node = node
        self.cfg = sub_cfg or {}
        self.K = K

    def detect(self, expected):
        raise NotImplementedError


BASE_FRAMES = ('base', 'base_link')


def normalize(label, p_cam=None, u=None, v=None, z=None, K=None, frame=None,
              extra=None):
    """把自接检测器的一条结果（p_cam 或 像素+深度）规范成 Detection。

    frame='base_link' 时位置视为已在 base 系（视觉模块直接给机械臂坐标），
    主流程跳过手眼变换；orientation 等附加信息放 extra 仅记录用。
    """
    label = str(label).strip().lower()
    if label not in ('l', 'm', 's'):
        raise TaskError(f'检测结果 label 只能是 l/m/s，收到 {label!r}')
    merged = dict(extra or {})
    if frame is not None:
        merged['frame'] = str(frame)
    if p_cam is not None:
        p = np.asarray(p_cam, float).reshape(3)
    else:
        if u is None or v is None or z is None or K is None:
            raise TaskError(f'{label}: 需提供 p_cam，或同时提供 u/v/z 与内参 K')
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
    if not np.all(np.isfinite(p)):
        raise TaskError(f'{label}: 坐标非法 {p}')
    return Detection(label=label, p_cam=p, u=u, v=v, z=z, extra=merged)


# ---------------- 画面点选检测器 --------------------------------------------

class ManualClickDetector(DetectorBase):
    """订阅彩色+对齐深度，按 expected 顺序依次点选，回车完成。

    键位：左键=选当前尺寸的螺母；u=撤销上一颗；回车/空格=完成；q/ESC=中止。
    """

    def detect(self, expected):
        import cv2
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo
        from cv_bridge import CvBridge
        from camera_pick_move import sample_depth, deproject, aligned_depth_and_K, load_camera_K

        vision = self.cfg['_vision']
        K_file = load_camera_K(vision['camera_info_path'])
        node = self.node
        bridge = CvBridge()
        state = {'color': None, 'depth': None,
                 'K_color': K_file, 'K_depth': None}

        def img(m):
            try:
                state['color'] = bridge.imgmsg_to_cv2(m, 'bgr8')
            except Exception:
                pass

        def dep(m):
            try:
                state['depth'] = bridge.imgmsg_to_cv2(m, 'passthrough')
            except Exception:
                pass

        def kci(m):
            if state['K_color'] is None:
                state['K_color'] = np.array(m.k, float).reshape(3, 3)

        def kd(m):
            state['K_depth'] = np.array(m.k, float).reshape(3, 3)

        subs = [
            node.create_subscription(Image, vision['color_topic'], img, qos_profile_sensor_data),
            node.create_subscription(Image, vision['depth_topic'], dep, qos_profile_sensor_data),
            node.create_subscription(CameraInfo, vision['color_info_topic'], kci, qos_profile_sensor_data),
            node.create_subscription(CameraInfo, vision['depth_info_topic'], kd, qos_profile_sensor_data),
        ]  # noqa: F841（保活）

        picked = []  # list[Detection]
        click = {'xy': None}

        # 复用 camera_pick_move.aligned_depth_and_K：它读 node.color/depth/K_*
        node.color = None
        node.depth = None
        node.K_color = state['K_color']
        node.K_depth = None

        click['err'] = None

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click['xy'] = (x, y)

        win = 'nut manual detector (click per prompt)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        PW, PH = 960, 540
        cv2.resizeWindow(win, PW, PH)
        cv2.setMouseCallback(win, on_mouse)

        def current_label():
            return expected[len(picked)] if len(picked) < len(expected) else None

        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.02)
                node.color, node.depth = state['color'], state['depth']
                node.K_color, node.K_depth = state['K_color'], state['K_depth']
                if state['color'] is None:
                    canvas = np.zeros((PH, PW, 3), np.uint8)
                    cv2.putText(canvas, 'WAITING FOR COLOR IMAGE...', (180, PH // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(20) & 0xFF
                    if key in (ord('q'), 27):
                        raise DetectorAbort('点选检测被中止')
                    continue

                frame = state['color'].copy()
                H0, W0 = frame.shape[:2]

                if click['xy'] is not None and current_label() is not None:
                    x, y = click['xy']
                    click['xy'] = None
                    u = int(round(x * W0 / PW))
                    v = int(round(y * H0 / PH))
                    depth_a, K = aligned_depth_and_K(node)
                    if depth_a is None:
                        click['err'] = '该点无深度：DEPTH WAITING'
                    elif K is None:
                        click['err'] = '该点无内参：NO CAMERA INFO'
                    else:
                        z = sample_depth(depth_a, u, v)
                        if z is None:
                            click['err'] = '该点深度无效/空洞，换个位置点'
                        else:
                            p_cam = deproject(u, v, z, K)
                            picked.append(Detection(label=current_label(), p_cam=p_cam,
                                                    u=u, v=v, z=z))
                            click['err'] = None

                # 画已选点
                for det in picked:
                    x = int(det.u * PW / W0)
                    y = int(det.v * PH / H0)
                    cv2.drawMarker(frame, (det.u, det.v), (0, 255, 0),
                                   cv2.MARKER_CROSS, 30, 2)
                    cv2.putText(frame, SIZE_NAMES_CN[det.label], (det.u + 12, det.v - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2, cv2.LINE_AA)

                lines = []
                for i, lab in enumerate(expected):
                    mark = '[x]' if i < len(picked) else '[ ]'
                    lines.append(f'{mark} {lab} ({SIZE_NAMES_CN[lab]}螺母)')
                lines.append('')
                if current_label() is not None:
                    lines.append(f'>>> 请点击【{SIZE_NAMES_CN[current_label()]}】螺母')
                    lines.append('左键=选取  u=撤销上一颗  q=中止')
                else:
                    lines.append('三颗已选完：回车/空格=确认  u=重选')
                if click.get('err'):
                    lines.append(click['err'])
                for i, ln in enumerate(lines):
                    cv2.putText(frame, ln, (14, 34 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(frame, ln, (14, 34 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(win, cv2.resize(frame, (PW, PH)))
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), 27):
                    raise DetectorAbort('点选检测被中止')
                if key in (ord('u'), ord('U')) and picked:
                    picked.pop()
                if key in (13, 10, 32) and len(picked) == len(expected):
                    return picked
        finally:
            cv2.destroyWindow(win)


# ---------------- JSON 离线检测器 -------------------------------------------

class JsonDetector(DetectorBase):
    """每次 detect 重新读 json：{"detections": [{"label":"l","p_cam":[x,y,z]}, ...]}

    也支持 {"label":"l","u":..,"v":..,"z":..}（此时用内参反投影）。
    """

    def detect(self, expected):
        path = Path(self.cfg['json_path'])
        if not path.exists():
            raise TaskError(f'json 检测器找不到文件 {path}')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        items = data.get('detections', data if isinstance(data, list) else [])
        out = []
        for it in items:
            frame = it.get('frame')
            if frame in BASE_FRAMES and it.get('p_base') is not None:
                det = normalize(it['label'], it.get('p_base'), frame='base_link',
                                extra={k: v for k, v in it.items()
                                       if k not in ('label', 'p_base', 'frame')})
            else:
                det = normalize(it['label'], it.get('p_cam'), it.get('u'),
                                it.get('v'), it.get('z'), self.K)
            if det.label in expected:
                out.append(det)
        return out


class InputDetector(DetectorBase):
    """临时联调用：不走相机，终端手动输入每颗螺母的 base_link 系位置（米）。

    每行 3 个数（空格或逗号分隔）；直接回车跳过该颗（require_all=true 时
    主流程的缺螺母校验会中止）。姿态仍取 left_grasp_init，与输入无关。
    """

    def __init__(self, node=None, sub_cfg=None, K=None, reader=input, printer=print):
        super().__init__(node, sub_cfg, K)
        self._read = reader
        self._print = printer

    @staticmethod
    def parse_line(text):
        nums = str(text).replace(',', ' ').split()
        if len(nums) != 3:
            raise TaskError(f'需要 3 个数（x y z，米），收到 {text!r}')
        try:
            return [float(v) for v in nums]
        except ValueError:
            raise TaskError(f'位置里有不是数字的内容：{text!r}')

    def detect(self, expected):
        self._print('手动输入螺母位置（base_link 系、单位米；空格/逗号分隔，回车跳过该颗）')
        out = []
        for label in expected:
            raw = self._read(f'  {SIZE_NAMES_CN[label]}螺母({label}) x,y,z: ')
            raw = (raw or '').strip()
            if not raw:
                continue
            p = self.parse_line(raw)
            out.append(normalize(label, p, frame='base_link',
                                 extra={'source': 'terminal_input'}))
        if not out:
            raise TaskError('一颗位置都没输入')
        return out


# ---------------- 像素结果自动补深度包装 -------------------------------------

class DepthPixelDetector(DetectorBase):
    """包装外部检测器：识别算法只给画面像素 (u,v) 时，框架订阅对齐深度自动取 z 并反投影。

    内层检测器每条结果给 p_cam（直接 3D）或 u/v（可带可不带 z）：
      - p_cam 直接透传；
      - u/v + z 用 K 反投影；
      - 只有 u/v：在对齐深度图该像素附近取样（复用 camera_pick_move.sample_depth 的
        邻域/半径退避策略），深度空洞则报错中止。
    也接受等价 dict（键 label/p_cam/u/v/z）。
    """

    DEPTH_WAIT_SECONDS = 3.0

    def __init__(self, inner, node, sub_cfg, K):
        super().__init__(node, sub_cfg, K)
        self.inner = inner
        vision = sub_cfg['_vision']
        self._K = K

        import rclpy  # noqa: F401（包装器只在真机节点路径构造）
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo
        from cv_bridge import CvBridge
        bridge = CvBridge()

        def dep(m):
            try:
                self._depth = bridge.imgmsg_to_cv2(m, 'passthrough')
            except Exception:
                pass

        def kci(m):
            if self._K is None:
                self._K = np.array(m.k, float).reshape(3, 3)

        node.create_subscription(Image, vision['depth_topic'], dep,
                                 qos_profile_sensor_data)
        # 深度已与彩色对齐，内参用彩色 K；文件没给就等话题
        node.create_subscription(CameraInfo, vision['color_info_topic'], kci,
                                 qos_profile_sensor_data)
        self._depth = None

    def _fields(self, it):
        if isinstance(it, Detection):
            frame = it.extra.get('frame')
            p = it.p_cam
            if frame in BASE_FRAMES:
                p = it.extra.get('p_base', it.p_cam)
            return (it.label, p, it.u, it.v, it.z, frame,
                    {k: v for k, v in it.extra.items() if k != 'frame'})
        if isinstance(it, dict) and 'label' in it:
            frame = it.get('frame')
            p = it.get('p_base') if frame in BASE_FRAMES else it.get('p_cam')
            extra = {k: v for k, v in it.items()
                     if k not in ('label', 'p_cam', 'p_base', 'frame',
                                  'u', 'v', 'z')}
            return (it['label'], p, it.get('u'), it.get('v'), it.get('z'),
                    frame, extra)
        raise TaskError(f'检测器结果必须是 Detection 或 dict，收到 {type(it)}：{it!r}')

    def detect(self, expected):
        import rclpy
        from camera_pick_move import sample_depth
        out = []
        for it in self.inner.detect(expected):
            label, p_xyz, u, v, z, frame, extra = self._fields(it)
            label = str(label).strip().lower()
            if p_xyz is not None:
                out.append(normalize(label, p_cam=p_xyz, frame=frame, extra=extra))
                continue
            if u is None or v is None:
                raise TaskError(f'{label}: 检测器结果必须给 p_cam，或给画面像素 u/v')
            ui, vi = int(round(u)), int(round(v))
            if z is None:
                t0 = time.monotonic()
                while self._depth is None and time.monotonic() - t0 < self.DEPTH_WAIT_SECONDS:
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                if self._depth is None:
                    raise TaskError(
                        f'{label}: {self.DEPTH_WAIT_SECONDS:.0f}s 内没收到对齐深度，'
                        f'相机是否 depth_registration:=true？')
                h, w = self._depth.shape[:2]
                if not (0 <= ui < w and 0 <= vi < h):
                    raise TaskError(f'{label}: 像素({ui},{vi}) 超出 {w}x{h} 画面')
                z = sample_depth(self._depth, ui, vi)
                if z is None:
                    raise TaskError(f'{label}: 像素({ui},{vi}) 附近深度无效/空洞，'
                                    f'识别框中心是否落在螺母表面？')
            if self._K is None:
                raise TaskError(f'{label}: 缺内参 K（标定文件与 camera_info 话题都没有）')
            out.append(normalize(label, u=ui, v=vi, z=z, K=self._K))
        return out


# ---------------- 外部检测器加载 ---------------------------------------------

def load_external_detector(spec, node, sub_cfg, K):
    """spec 形如 '/path/to/file.py:ClassName' 或 'pkg.module:ClassName'。"""
    if ':' not in spec:
        raise TaskError("detector.external 格式应为 '路径或模块:类名'")
    mod_part, cls_name = spec.rsplit(':', 1)
    # 让外部文件能 import 同目录的 nut_detectors.Detection
    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    if mod_part.endswith('.py'):
        mod_path = Path(mod_part)
        if not mod_path.is_absolute():
            # 先按【工作区根目录】解析（仓库内相对路径与当前目录无关），
            # 不存在再退回当前工作目录。换电脑 / 换 cwd 都不会失效。
            ws_cand = Path(__file__).resolve().parents[1] / mod_path
            mod_path = ws_cand if ws_cand.exists() else (Path.cwd() / mod_path)
        if not mod_path.exists():
            raise TaskError(f'外部检测器文件不存在 {mod_path}')
        import importlib.util
        spec_obj = importlib.util.spec_from_file_location(
            f'nut_ext_detector_{mod_path.stem}', mod_path)
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[spec_obj.name] = module
        spec_obj.loader.exec_module(module)
    else:
        import importlib
        module = importlib.import_module(mod_part)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise TaskError(f'{mod_part} 里没有类 {cls_name}')
    return cls(node, sub_cfg, K)


def build_detector(cfg, node=None, K=None):
    """按 TaskConfig 构造检测器。

    cfg     : TaskConfig
    node    : rclpy 节点（manual 必须；json 可 None）
    K       : 内参 3x3 或 None
    """
    kind = cfg.detector_type
    sub = dict(cfg.detector_raw)
    sub['json_path'] = str(cfg.detector_json_path)
    sub['_vision'] = {
        'color_topic': cfg.color_topic,
        'color_info_topic': cfg.color_info_topic,
        'depth_topic': cfg.depth_topic,
        'depth_info_topic': cfg.depth_info_topic,
        'camera_info_path': cfg.camera_info_path,
    }
    if kind == 'yolo':
        from nut_yolo import YoloDetector
        sub['_vision']['extrinsics_path'] = str(cfg.extrinsics_path)
        return YoloDetector(node, sub, K)
    if kind == 'manual':
        if node is None:
            raise TaskError('manual 检测器需要 rclpy 节点')
        return ManualClickDetector(node, sub, K)
    if kind == 'json':
        return JsonDetector(node, sub, K)
    if kind == 'input':
        return InputDetector(node, sub, K)
    if kind == 'external':
        if not cfg.detector_external:
            raise TaskError("detector.type=external 但 detector.external 为空")
        if node is None:
            return load_external_detector(cfg.detector_external, node, sub, K)
        # 真机：包一层，允许外部算法只返回画面像素 u/v，深度由框架补
        inner = load_external_detector(cfg.detector_external, node, sub, K)
        return DepthPixelDetector(inner, node, sub, K)
    raise TaskError(f'未知检测器类型 {kind!r}（manual/json/external/yolo）')
