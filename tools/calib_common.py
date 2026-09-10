#!/usr/bin/env python3
"""手眼标定共享工具：坐标系路径、板配置、旋转/齐次矩阵换算、样本落盘。

被 calib_charuco_board.py / calib_handeye_sample.py / calib_handeye_solve.py /
calib_handeye_publish_tf.py / calib_handeye_verify.py 复用。

坐标约定（Eye-to-Hand）：
  B = base_link（机械臂基座，pose_states 所在帧）
  E = 机械臂末端 frame（pose_states 表达的那个 frame）
  M = ChArUco 标定板 frame
  C = camera_color_optical_frame（相机光学帧）
采样得到 B_T_E（机器人位姿）与 C_T_M（solvePnP 估计的板位姿）。
"""
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # 运行时若缺 scipy 给出清晰报错
    Rotation = None

import yaml


# ---- 路径 ------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
WORKSPACE = TOOLS_DIR.parent
CALIB_DIR = WORKSPACE / '开发资源' / 'calibration'
RECORDINGS_DIR = WORKSPACE / 'recordings'

BOARD_CONFIG = CALIB_DIR / 'charuco_board.yaml'
EXTRINSICS_PATH = CALIB_DIR / 'gemini2_extrinsics.yaml'
COLOR_CAMERA_INFO_PATH = CALIB_DIR / 'gemini2_color_camera_info.yaml'


# ---- 旋转 / 齐次矩阵 --------------------------------------------------------

def _require_scipy():
    if Rotation is None:
        raise RuntimeError('需要 scipy：pip install scipy 或 source 完整 ROS2 环境')


def quat_xyzw_to_mat(q):
    """四元数 [x, y, z, w] -> 3x3 旋转矩阵。"""
    _require_scipy()
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q).as_matrix()


def mat_to_quat_xyzw(R):
    """3x3 旋转矩阵 -> 四元数 [x, y, z, w]。"""
    _require_scipy()
    return Rotation.from_matrix(R).as_quat()


def quat_xyzw_to_euler_deg(q):
    """四元数 [x,y,z,w] -> 欧拉角 [roll,pitch,yaw]（度，xyz 顺序）。"""
    _require_scipy()
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_euler('xyz', degrees=True)


def rvec_to_mat(rvec):
    """Rodrigues 旋转向量 (3,) 或 (3,1) -> 3x3。"""
    _require_scipy()
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    return Rotation.from_rotvec(rvec).as_matrix()


def mat_to_rvec(R):
    _require_scipy()
    return Rotation.from_matrix(R).as_rotvec()


def make_T(R, t):
    """由 3x3 旋转与 3 维平移构造 4x4 齐次变换。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def T_from_pose(position, quat_xyzw):
    return make_T(quat_xyzw_to_mat(quat_xyzw), position)


def T_to_pose(T):
    """4x4 -> (position(3,), quat_xyzw(4,))。"""
    return T[:3, 3].copy(), mat_to_quat_xyzw(T[:3, :3])


def T_from_rvec_tvec(rvec, tvec):
    return make_T(rvec_to_mat(rvec), np.asarray(tvec).reshape(3))


def T_to_rvec_tvec(T):
    return mat_to_rvec(T[:3, :3]), T[:3, 3].copy()


def invert_T(T):
    return np.linalg.inv(T)


def rotation_angle_between(R1, R2):
    """两个旋转矩阵之间的旋转角（度）。"""
    _require_scipy()
    R = np.asarray(R1) @ np.asarray(R2).T
    return Rotation.from_matrix(R).magnitude() * 180.0 / math.pi


def rotation_angle_of(R):
    _require_scipy()
    return Rotation.from_matrix(R).magnitude() * 180.0 / math.pi


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


# ---- ChArUco 板配置 ---------------------------------------------------------

DEFAULT_DICT_NAME = 'DICT_5X5_100'


def get_aruco_dictionary(name=DEFAULT_DICT_NAME):
    """OpenCV 4.6 旧版 aruco API。"""
    import cv2
    from cv2 import aruco
    dict_id = getattr(aruco, name, None)
    if dict_id is None:
        raise ValueError(f'未知 ArUco 字典名：{name}')
    return aruco.Dictionary_get(dict_id)


def create_board(cfg):
    """按配置 dict 构造 ChArUco 板（旧版 API）。返回 board 对象。"""
    from cv2 import aruco
    dictionary = get_aruco_dictionary(cfg['dictionary'])
    return aruco.CharucoBoard_create(
        int(cfg['squares_x']), int(cfg['squares_y']),
        float(cfg['square_length_m']), float(cfg['marker_length_m']),
        dictionary)


def load_board_config(path=BOARD_CONFIG):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for key in ('dictionary', 'squares_x', 'squares_y', 'square_length_m', 'marker_length_m'):
        if key not in cfg:
            raise ValueError(f'板配置缺少字段：{key}（{path}）')
    return cfg


def save_board_config(cfg, path=BOARD_CONFIG):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


# ---- 样本 JSONL -------------------------------------------------------------

def clean_json(value):
    """NaN/Inf -> None，保证 strict JSON（对齐 record_workpoints）。"""
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [clean_json(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_jsonl_event(fh, event):
    fh.write(json.dumps(clean_json(event), ensure_ascii=False, allow_nan=False) + '\n')
    fh.flush()
    os.fsync(fh.fileno())


def read_jsonl(path):
    events = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
