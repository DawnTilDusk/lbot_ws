#!/usr/bin/env python3
"""双臂序列版抓放任务的配置/位姿/运动与灵巧手封装。

被 nut_pick_place.py / nut_sequences.py / capture_task_pose.py 复用：
  - TaskConfig: 读 nut_task.yaml（双臂 legs 模型）
  - PoseStore:  task_poses.yaml 读写（本版只需要左臂 grasp_orientation 姿态）
  - RobotClient: 单臂客户端 —— SetEnable / MoveJP / MoveL / MoveJ / IK + 状态订阅 + L6 手
"""
import datetime
import time
from pathlib import Path

import numpy as np
import yaml

SIZE_LABELS = ('l', 'm', 's')
SIZE_NAMES_CN = {'l': '大', 'm': '中', 's': '小'}

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = WORKSPACE / '开发资源' / 'nut_sort' / 'nut_task.yaml'


class TaskError(RuntimeError):
    """任务级错误；抛出即中止，不自动掉使能。"""


def _hand6(value, where):
    arr = [int(x) for x in value]
    if len(arr) != 6 or any(not 0 <= x <= 255 for x in arr):
        raise TaskError(f'{where} 必须是 6 个 0~255 的整数，当前 {value}')
    return arr


def resolve_ws(path_str):
    """配置里的 trace 路径：绝对路径原样；否则相对【工作区根目录】解析。"""
    p = Path(path_str)
    return p if p.is_absolute() else WORKSPACE / p


# ---------------- 配置：一条「腿」（预录序列描述） ---------------------------

def _parse_leg_spec(arm, raw, default_trace):
    """配置 dict -> (trace 文件路径, sequence 名, hand_after, retreat)。"""
    if not isinstance(raw, dict) or raw.get('sequence') is None:
        raise TaskError(f'{arm} 段配置必须是带 sequence 名的映射，当前 {raw}')
    hand_after = raw.get('hand_after')
    if hand_after not in (None, 'open', 'close'):
        raise TaskError(f'{arm} 段 {raw["sequence"]} 的 hand_after 只能是 open/close/空')
    retreat = bool(raw.get('retreat', False))
    file_ = raw.get('file') or default_trace
    if not file_:
        raise TaskError(f'{arm} 段 {raw["sequence"]} 没给 trace 文件（段 file: 或该臂 trace:）')
    return resolve_ws(file_), str(raw['sequence']), hand_after, retreat


class TaskConfig:
    def __init__(self, yaml_path=DEFAULT_CONFIG, arm_override=None, order_override=None,
                 detector_override=None):
        self.path = Path(yaml_path)
        if not self.path.exists():
            raise TaskError(f'任务配置文件不存在：{self.path}')
        with open(self.path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        self.dir = self.path.parent
        d = lambda k, default=None: cfg.get(k, default)

        self.namespace = '/' + str(d('namespace', '/robot1')).strip('/')
        self.arm_for_capture = arm_override or d('arm', 'right')
        self.check_other = bool(d('check_other_arm', True))

        order = order_override or d('order', ['l', 'm', 's'])
        order = [str(x).strip().lower() for x in order]
        if not order or any(x not in SIZE_LABELS for x in order):
            raise TaskError(f'order 只允许 l/m/s，当前 {order}')
        if len(set(order)) != len(order):
            raise TaskError(f'order 有重复尺寸：{order}')
        self.order = tuple(order)
        self.require_all = bool(d('require_all', True))

        # ---- 左臂 ----
        left = d('left', {}) or {}
        self.left_grasp_pose_name = left.get('grasp_orientation', 'left_grasp_init')
        left_trace = left.get('trace', '')
        if not isinstance(left.get('legs'), list) or not left['legs']:
            raise TaskError('left.legs 必须是非空列表（视觉抓起后依次回放的段）')
        self.left_legs = [_parse_leg_spec('left', x, left_trace) for x in left['legs']]
        if any(leg[2] == 'close' for leg in self.left_legs):
            raise TaskError('左臂序列段 hand_after 不应为 close（抓握发生在视觉下探后，由流程发）')
        # home = 第一段第一个点；中央释放必须在某段尾张手
        if not any(leg[2] == 'open' for leg in self.left_legs):
            raise TaskError('左臂 legs 没有任何段 hand_after=open（无法在中央释放螺母）')

        # ---- 右臂 ----
        right = d('right', {}) or {}
        right_trace = right.get('trace', '')
        appr = right.get('approach')
        if not appr:
            raise TaskError('right.approach 必填（home->中央，段尾 close 重抓）')
        self.right_approach = _parse_leg_spec('right', appr, right_trace)
        if self.right_approach[2] != 'close':
            raise TaskError('右臂 approach 段必须 hand_after: close（到中央后重抓）')
        place = right.get('place', {}) or {}
        self.right_place = {}
        if isinstance(place, dict) and 'sequence' in place:
            # 三颗螺母共用一条 place 段（中央 -> 同一个盒位）
            leg = _parse_leg_spec('right', place, right_trace)
            if leg[2] != 'open':
                raise TaskError('右臂 place 段必须 hand_after: open（入盒释放）')
            self.place_shared = True
            self.right_place = {k: leg for k in SIZE_LABELS}
        else:
            self.place_shared = False
            for k in SIZE_LABELS:
                raw = place.get(k)
                if raw is None:
                    raise TaskError(f'right.place.{k} 未配置（该尺寸放哪个序列段？'
                                    f'三颗同盒也可直接把 place 写成单段）')
                leg = _parse_leg_spec('right', raw, right_trace)
                if leg[2] != 'open':
                    raise TaskError(f'右臂 place.{k} 段必须 hand_after: open（入盒释放）')
                self.right_place[k] = leg
        # 右臂 home = approach 第一段第一个点

        # ---- 运动参数 ----
        m = d('motion', {}) or {}
        self.speed = float(m.get('speed', 0.3))
        self.acce = float(m.get('acce', 0.3))
        self.linear_speed = float(m.get('linear_speed', 0.2))
        self.linear_acce = float(m.get('linear_acce', 0.2))
        self.hover_height = float(m.get('hover_height', 0.10))
        self.grasp_z_offset = float(m.get('grasp_z_offset', 0.0))
        self.join_speed = float(m.get('join_speed', 0.15))
        self.join_acce = float(m.get('join_acce', 0.15))
        self.sequence_speed = float(m.get('sequence_speed', 0.3))
        self.sequence_acce = float(m.get('sequence_acce', 0.5))
        for key, val, limit in (('join_speed', self.join_speed, 0.3),
                                ('join_acce', self.join_acce, 0.3),
                                ('sequence_speed', self.sequence_speed, 0.5)):
            if not 0 < val <= limit:
                raise TaskError(f'motion.{key}={val} 必须在 (0,{limit}]')
        self.state_timeout = float(m.get('state_timeout', 0.5))
        self.settle_seconds = float(m.get('settle_seconds', 0.6))
        self.release_seconds = float(m.get('release_seconds', 0.4))
        self.move_timeout = float(m.get('move_timeout', 60.0))
        self.start_tolerance = float(m.get('start_tolerance', 0.05))
        self.reached_tolerance = float(m.get('reached_tolerance', 0.03))
        self.other_tolerance = float(m.get('other_tolerance', 0.05))

        # ---- 手参数 ----
        h = d('hand', {}) or {}
        self.hand_speed = _hand6(h.get('speed', [80] * 6), 'hand.speed')
        self.hand_force = _hand6(h.get('force', [100] * 6), 'hand.force')
        self.hand_open_vals = _hand6(h.get('open', [0] * 6), 'hand.open')
        # 闭合值两级配置：hand.close.{left,right} 给各臂默认值；
        # hand.sizes.{l,m,s} 可用 left/right/joint 按尺寸覆盖
        #   sizes.l: {left: [...], right: [...]}  或裸列表（=双臂共用 joint）
        close_cfg = h.get('close', {}) or {}
        self._hand_close_arm = {}
        for arm in ('left', 'right'):
            if close_cfg.get(arm) is not None:
                self._hand_close_arm[arm] = _hand6(close_cfg[arm], f'hand.close.{arm}')
        sizes = h.get('sizes', {}) or {}
        self._hand_close = {}
        for k in SIZE_LABELS:
            raw = sizes.get(k)
            if raw is None:
                entry = {}
            elif isinstance(raw, list):
                entry = {'joint': _hand6(raw, f'hand.sizes.{k}')}
            elif isinstance(raw, dict):
                entry = {}
                for key in ('joint', 'left', 'right'):
                    if raw.get(key) is not None:
                        entry[key] = _hand6(raw[key], f'hand.sizes.{k}.{key}')
            else:
                raise TaskError(f'hand.sizes.{k} 必须是列表或映射')
            self._hand_close[k] = entry

        # ---- 视觉 ----
        v = d('vision', {}) or {}
        from calib_common import EXTRINSICS_PATH, COLOR_CAMERA_INFO_PATH
        self.color_topic = v.get('color_topic', '/camera/color/image_raw')
        self.color_info_topic = v.get('color_info_topic', '/camera/color/camera_info')
        self.depth_topic = v.get('depth_topic', '/camera/depth/image_raw')
        self.depth_info_topic = v.get('depth_info_topic', '/camera/depth/camera_info')
        self.extrinsics_path = resolve_ws(v['extrinsics']) if v.get('extrinsics') else EXTRINSICS_PATH
        self.camera_info_path = resolve_ws(v['camera_info']) if v.get('camera_info') else COLOR_CAMERA_INFO_PATH
        if not self.extrinsics_path.is_absolute():
            self.extrinsics_path = self.dir / self.extrinsics_path
        if not self.camera_info_path.is_absolute():
            self.camera_info_path = self.dir / self.camera_info_path

        # ---- 检测器 ----
        det = d('detector', {}) or {}
        self.detector_type = detector_override or det.get('type', 'manual')
        jp = det.get('json_path') or 'detections.json'
        jp = Path(jp)
        self.detector_json_path = jp if jp.is_absolute() else self.dir / jp
        self.detector_external = det.get('external', '') or ''
        self.detector_raw = det

    def close_for(self, arm, label):
        """该臂抓 label 螺母时的 6 路闭合值：尺寸级 arm > 尺寸级 joint > 臂默认。"""
        entry = self._hand_close.get(label, {})
        if arm in entry:
            return entry[arm]
        if 'joint' in entry:
            return entry['joint']
        if arm in self._hand_close_arm:
            return self._hand_close_arm[arm]
        raise TaskError(f'没配 {arm} 臂对 {label} 螺母的闭合值'
                        f'（需要 hand.close.{arm} 或 hand.sizes.{label}）')

    @property
    def poses_path(self):
        """位姿文件（本版主要存 left_grasp_init）。"""
        return self.dir / 'task_poses.yaml'


# ---------------- 位姿库 -----------------------------------------------------

class PoseStore:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {'left': {}, 'right': {}}
        if self.path.exists():
            with open(self.path, encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
            for arm in ('left', 'right'):
                if isinstance(loaded.get(arm), dict):
                    self.data[arm] = loaded[arm]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('# 由 capture_task_pose.py 写入；position=m, euler=度, joints=rad\n')
            yaml.safe_dump(self.data, f, allow_unicode=True, sort_keys=False)

    def names(self, arm):
        return sorted(self.data.get(arm, {}).keys())

    def get(self, arm, name):
        try:
            pose = self.data[arm][name]
        except KeyError:
            raise TaskError(
                f"位姿 {name!r} 未标定（--arm {arm}）。已有：{self.names(arm) or '无'}；"
                f"用 capture_task_pose.py --arm {arm} --name {name} 采集")
        for key in ('position_m', 'euler_deg'):
            if key not in pose or len(pose[key]) != 3:
                raise TaskError(f'位姿 {name!r} 缺字段 {key}，请重新采集')
        return pose

    def put(self, arm, name, position, euler_deg, joints, frame_id='base_link'):
        self.data.setdefault(arm, {})[name] = {
            'position_m': [float(x) for x in position],
            'euler_deg': [float(x) for x in euler_deg],
            'joints_rad': [float(x) for x in joints] if joints is not None else None,
            'frame_id': frame_id,
            'recorded_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }

    def delete(self, arm, name):
        if name not in self.data.get(arm, {}):
            raise TaskError(f'{arm} 臂没有位姿 {name!r}；已有：{self.names(arm) or "无"}')
        del self.data[arm][name]


def pose_euler_rad(pose):
    return np.radians(np.asarray(pose['euler_deg'], float))


# ---------------- 单臂运动 / 手 客户端 ---------------------------------------

class RobotClient:
    def __init__(self, node, namespace, arm):
        self.node = node
        self.ns = namespace
        self.arm = arm
        base = f'{namespace}/{arm}_arm'
        from lbot_arm_interfaces.srv import MoveJP, MoveL, MoveJ, SetEnable, InverseKinematics
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import UInt8MultiArray
        self._MoveJP, self._MoveL, self._MoveJ = MoveJP, MoveL, MoveJ
        self.enable_cli = node.create_client(SetEnable, f'{base}/set_enable')
        self.move_cli = node.create_client(MoveJP, f'{base}/move_pose')
        self.linear_cli = node.create_client(MoveL, f'{base}/move_linear')
        self.joint_cli = node.create_client(MoveJ, f'{base}/move_joint')
        self.ik_cli = node.create_client(InverseKinematics, f'{base}/inverse_kinematics')
        self.joints = None
        self.joint_names = None
        self.joints_ts = -1e18
        self.pose = None
        node.create_subscription(JointState, f'{base}/joint_states', self._joints, 10)
        node.create_subscription(PoseStamped, f'{base}/pose_states', self._pose, 10)
        hbase = f'{namespace}/{arm}_hand'
        self.hand_pubs = {
            'joint': node.create_publisher(UInt8MultiArray, f'{hbase}/set_l6_joint', 10),
            'speed': node.create_publisher(UInt8MultiArray, f'{hbase}/set_l6_speed', 10),
            'force': node.create_publisher(UInt8MultiArray, f'{hbase}/set_l6_force', 10),
        }

    def _joints(self, m):
        if len(m.position) == 7:
            self.joints = [float(x) for x in m.position]
            self.joint_names = list(m.name)
            self.joints_ts = time.monotonic()

    def _pose(self, m):
        p, q = m.pose.position, m.pose.orientation
        self.pose = (np.array([p.x, p.y, p.z]),
                     np.array([q.x, q.y, q.z, q.w]))

    def feedback_fresh(self, timeout):
        return (self.joints is not None and self.joint_names is not None
                and time.monotonic() - self.joints_ts < timeout)

    def _spin_call(self, cli, req, timeout):
        import rclpy
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if fut.done():
                return fut.result()
        return None

    def wait_services(self, timeout=5.0):
        for tag, cli in (('set_enable', self.enable_cli), ('move_pose', self.move_cli),
                         ('move_linear', self.linear_cli), ('move_joint', self.joint_cli),
                         ('inverse_kinematics', self.ik_cli)):
            if not cli.wait_for_service(timeout_sec=timeout):
                raise TaskError(f'{self.arm}_arm/{tag} 服务不可用，驱动是否启动？')

    def wait_state(self, timeout=3.0):
        import rclpy
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.joints is not None and self.pose is not None:
                return True
        return False

    def enable(self):
        from lbot_arm_interfaces.srv import SetEnable
        res = self._spin_call(self.enable_cli, SetEnable.Request(enable=True), timeout=10)
        if not (res is not None and res.success):
            raise TaskError(f'{self.arm} 臂上使能失败')

    def ik_check(self, position, euler_rad):
        req = self.ik_cli.srv_type.Request()
        req.position.x, req.position.y, req.position.z = [float(v) for v in position]
        req.euler.x, req.euler.y, req.euler.z = [float(v) for v in euler_rad]
        req.joints = list(self.joints) if self.joints is not None else []
        res = self._spin_call(self.ik_cli, req, timeout=5)
        return res is not None and bool(res.success)

    def move_pose(self, position, euler_rad, speed, acce, timeout=None):
        req = self._MoveJP.Request()
        req.position.x, req.position.y, req.position.z = [float(v) for v in position]
        req.euler.x, req.euler.y, req.euler.z = [float(v) for v in euler_rad]
        req.speed, req.acce, req.block = float(speed), float(acce), True
        self._do(self.move_cli, req, f'{self.arm} move_pose', timeout)

    def move_linear(self, position, euler_rad, speed, acce, timeout=None):
        req = self._MoveL.Request()
        req.position.x, req.position.y, req.position.z = [float(v) for v in position]
        req.euler.x, req.euler.y, req.euler.z = [float(v) for v in euler_rad]
        req.speed, req.acce, req.block = float(speed), float(acce), True
        self._do(self.linear_cli, req, f'{self.arm} move_linear', timeout)

    def move_joint(self, joints, speed, acce, timeout=None):
        req = self._MoveJ.Request()
        req.joints = [float(v) for v in joints]
        req.speed, req.acce, req.block = float(speed), float(acce), True
        self._do(self.joint_cli, req, f'{self.arm} move_joint', timeout)

    def _do(self, cli, req, tag, timeout):
        res = self._spin_call(cli, req, timeout or 60.0)
        if res is None or not res.success:
            raise TaskError(f'{tag} 失败/超时')

    # ---- 灵巧手 ----
    def _publish_hand(self, kind, values):
        from std_msgs.msg import UInt8MultiArray
        msg = UInt8MultiArray()
        msg.data = list(values)
        for _ in range(3):  # 话题不锁存，连发防丢包
            self.hand_pubs[kind].publish(msg)
            time.sleep(0.03)

    def hand_setup(self, speed, force):
        self._publish_hand('speed', speed)
        self._publish_hand('force', force)

    def hand_open(self, values):
        self._publish_hand('joint', values)

    def hand_close(self, values, settle=0.0):
        self._publish_hand('joint', values)
        if settle > 0:
            time.sleep(settle)

    def joint_diff(self, target):
        if self.joints is None:
            return None
        return max(abs(a - b) for a, b in zip(self.joints, target))
