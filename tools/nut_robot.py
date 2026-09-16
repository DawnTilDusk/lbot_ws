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


def _parse_grasp_orientation(raw, where):
    """grasp_orientation 的值 -> (位姿库名|None, 记录段映射|None)。

    字符串=task_poses.yaml 里的位姿名（只取欧拉角）；
    映射={file?, sequence, point?}=预录段某点的末端姿态。
    """
    if isinstance(raw, str) and raw:
        return raw, None
    if isinstance(raw, dict) and raw.get('sequence'):
        point = int(raw.get('point', 0))
        if point < 0:
            raise TaskError(f'{where}.point 不能为负')
        return None, {'file': raw.get('file'), 'sequence': str(raw['sequence']), 'point': point}
    raise TaskError(f'{where} 必须是位姿名字符串，或 {{file?, sequence, point?}} 映射，'
                    f'当前 {raw!r}')


def _parse_grasp_offset(raw, where):
    """检测点->腕部目标的 base_link 平移（米）；全局与按尺寸共用同一校验。"""
    try:
        off = [float(x) for x in raw]
    except (TypeError, ValueError):
        raise TaskError(f'{where} 必须是 3 个数（米），当前 {raw}')
    if len(off) != 3 or not all(np.isfinite(off)):
        raise TaskError(f'{where} 必须是 3 个有限数（米），当前 {raw}')
    if np.linalg.norm(off) > 0.3:
        raise TaskError(f'{where} 模长 {np.linalg.norm(off):.2f}m 过大'
                        f'（上限 0.3m），确认单位是米而不是毫米')
    return np.array(off)


def _parse_ready_spec(arm, raw, default_trace):
    """left/right.ready -> leg 规格；不配返回 None。ready 是开机纯运动段，不许带手动作。"""
    if raw is None:
        return None
    spec = _parse_leg_spec(arm, raw, default_trace)
    if spec[2] is not None:
        raise TaskError(f'{arm}.ready（{spec[1]}）是开机纯运动段，不许配 hand_after'
                        f'（初始张手由框架在回放前统一发）')
    if spec[3]:
        raise TaskError(f'{arm}.ready（{spec[1]}）不支持 retreat（末点就是要停住的 ready 位）')
    return spec


class TaskConfig:
    def __init__(self, yaml_path=DEFAULT_CONFIG, arm_override=None, order_override=None,
                 detector_override=None, speed_scale=1.0):
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
        self.left_trace = left.get('trace', '')
        # 视觉抓取姿态两种来源：
        #   字符串  -> task_poses.yaml 里的位姿名（只取三欧拉角）
        #   映射    -> {sequence: 名, file?: 可选(默认 left.trace), point?: 点序号(默认0)}
        #              取预录段该点的末端姿态（如 left_grasp_middle pt0 的旋转角）
        self.left_grasp_pose_name, self.grasp_orientation_rec = _parse_grasp_orientation(
            left.get('grasp_orientation', 'left_grasp_init'), 'left.grasp_orientation')
        # 按尺寸覆盖抓取姿态（l/m/s 可缺省 -> 回退全局姿态）
        self.grasp_orientation_by_size = {}
        go_size = left.get('grasp_orientation_by_size', {}) or {}
        if not isinstance(go_size, dict):
            raise TaskError('left.grasp_orientation_by_size 必须是 l/m/s 映射')
        for k, v in go_size.items():
            if k not in SIZE_LABELS:
                raise TaskError(f'left.grasp_orientation_by_size 只允许 l/m/s 键，当前 {k!r}')
            self.grasp_orientation_by_size[k] = _parse_grasp_orientation(
                v, f'left.grasp_orientation_by_size.{k}')
        left_trace = left.get('trace', '')
        if not isinstance(left.get('legs'), list) or not left['legs']:
            raise TaskError('left.legs 必须是非空列表（视觉抓起后依次回放的段）')
        self.left_legs = [_parse_leg_spec('left', x, left_trace) for x in left['legs']]
        if any(leg[2] == 'close' for leg in self.left_legs):
            raise TaskError('左臂序列段 hand_after 不应为 close（抓握发生在视觉下探后，由流程发）')
        # home = 第一段第一个点；中央释放必须在某段尾张手
        if not any(leg[2] == 'open' for leg in self.left_legs):
            raise TaskError('左臂 legs 没有任何段 hand_after=open（无法在中央释放螺母）')
        # 可选开机 ready 段：上使能/张手后逐点回放到其末点（离场位），再开始检测
        self.left_ready = _parse_ready_spec('left', left.get('ready'), left_trace)

        # ---- 右臂 ----
        right = d('right', {}) or {}
        right_trace = right.get('trace', '')
        appr = right.get('approach')
        if not appr:
            raise TaskError('right.approach 必填（home->中央，段尾 close 重抓）')

        def _parse_approach(raw, where):
            leg = _parse_leg_spec('right', raw, right_trace)
            if leg[2] != 'close':
                raise TaskError(f'右臂 {where} 必须 hand_after: close（到中央后重抓）')
            return leg

        if isinstance(appr, dict) and 'sequence' in appr:
            # 三颗共用一条 approach 段（单段写法，向后兼容）
            leg = _parse_approach(appr, 'approach 段')
            self.approach_shared = True
            self.right_approaches = {k: leg for k in SIZE_LABELS}
            self.right_approach = leg
        elif isinstance(appr, dict):
            # 按尺寸分别给中央重抓段（中小螺母拇指高度不同时各调各的末点）
            self.approach_shared = False
            self.right_approaches = {}
            for k in SIZE_LABELS:
                raw = appr.get(k)
                if raw is None:
                    raise TaskError(f'right.approach.{k} 未配置（该尺寸中央重放走哪条序列段？'
                                    f'三颗相同也可直接把 approach 写成单段）')
                self.right_approaches[k] = _parse_approach(raw, f'approach.{k} 段')
            self.right_approach = self.right_approaches['l']  # 代表段：home/ready 接入
        else:
            raise TaskError('right.approach 必须是段映射（含 sequence）或 l/m/s 三段映射')
        self.right_ready = _parse_ready_spec('right', right.get('ready'), right_trace)
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
        # 抓稳后竖直抬起高度（相对下探终点 down 向上，独立于接近悬停 hover_height）
        try:
            self.lift_height = float(m.get('lift_height', 0.05))
        except (TypeError, ValueError):
            raise TaskError(f"motion.lift_height 必须是数（米），当前 {m.get('lift_height')!r}")
        if not np.isfinite(self.lift_height) or not 0 <= self.lift_height <= 0.30:
            raise TaskError(f'motion.lift_height={self.lift_height} 必须是 0~0.30 米之间的有限数')
        # 检测点 -> 腕部目标的 base_link 系平移（米）。视觉/点选给的是螺母位置，
        # 而 pose_states/腕部（法兰）与指尖抓取中心有约 15cm 前后偏差：
        # 默认把腕部目标向机体方向（base_link -X）退 15cm，指尖才正好到螺母。
        # 三种检测器（manual/input/yolo/...）在变到 base_link 之后统一施加。
        self.grasp_offset_xyz = _parse_grasp_offset(
            m.get('grasp_offset_xyz', [-0.15, 0.0, 0.0]), 'motion.grasp_offset_xyz')
        # 按尺寸覆盖（motion.grasp_by_size.<l/m/s>）：offset_xyz/z_offset/hover_height/
        # lift_height 四个字段都可省，省的回退上面的全局值；大中小螺母几何不同时分别微调。
        self.grasp_by_size = {}
        gb = m.get('grasp_by_size', {}) or {}
        if not isinstance(gb, dict):
            raise TaskError('motion.grasp_by_size 必须是 l/m/s 映射')
        for k, v in gb.items():
            if k not in SIZE_LABELS:
                raise TaskError(f'motion.grasp_by_size 只允许 l/m/s 键，当前 {k!r}')
            if not isinstance(v, dict):
                raise TaskError(f'motion.grasp_by_size.{k} 必须是映射'
                                f'（offset_xyz/z_offset/hover_height/lift_height 任选）')
            entry = {}
            if 'offset_xyz' in v:
                entry['offset_xyz'] = _parse_grasp_offset(
                    v['offset_xyz'], f'motion.grasp_by_size.{k}.offset_xyz')
            for fk in ('z_offset', 'hover_height', 'lift_height'):
                if fk in v:
                    try:
                        fv = float(v[fk])
                    except (TypeError, ValueError):
                        raise TaskError(f'motion.grasp_by_size.{k}.{fk} 必须是数（米），'
                                        f'当前 {v[fk]!r}')
                    if not np.isfinite(fv):
                        raise TaskError(f'motion.grasp_by_size.{k}.{fk} 必须是有限数（米）')
                    if fk == 'lift_height' and not 0 <= fv <= 0.30:
                        raise TaskError(
                            f'motion.grasp_by_size.{k}.lift_height={fv} 必须在 0~0.30 米')
                    entry[fk] = fv
            self.grasp_by_size[k] = entry
        # 竖直几何必须单调：接近悬停 hover 在抓取点 down 上方（z 越大越高）。
        # 若 z_offset >= hover_height，下探反向上走、抓后回 hover 反而下杵桌面。
        for k in SIZE_LABELS:
            zoff, hov = self.grasp_z_offset_for(k), self.hover_height_for(k)
            if zoff >= hov:
                where = ('motion.grasp_by_size.' + k) if k in self.grasp_by_size else 'motion'
                raise TaskError(
                    f'{where}: 抓取点 z_offset={zoff:.3f} 必须小于悬停 hover_height='
                    f'{hov:.3f}（当前 hover 不在 down 上方，会导致接近时上抬、'
                    f'抓后回 hover 下杵桌面）。把 hover_height 调到 z_offset+期望余量'
                    f'（如 +0.05），或减小 z_offset')
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
        # 节奏停顿（秒）：让各环节衔接可观察、可急停
        for key, default in (('ready_hold_seconds', 1.0),    # 开机到位后保持
                             ('point_dwell_seconds', 0.2),   # 每个记录点到位后停顿
                             ('between_leg_seconds', 0.8),   # 段与段之间
                             ('pre_hand_seconds', 0.3),      # 张/合手指令前停顿
                             ('hover_dwell_seconds', 0.5)):  # 视觉段到 hover 后/抬起后
            val = float(m.get(key, default))
            if val < 0:
                raise TaskError(f'motion.{key} 不能为负')
            setattr(self, key, val)
        self.move_timeout = float(m.get('move_timeout', 60.0))
        self.start_tolerance = float(m.get('start_tolerance', 0.05))
        self.reached_tolerance = float(m.get('reached_tolerance', 0.03))
        # 分臂到位容差：某臂持物/受力姿态伺服稳态偏差可能系统性地略大于全局值
        # （现场右臂持螺母伸展时肩滚转稳态停在 0.033rad，重发同目标无效）
        self.reached_tolerance_left = m.get('reached_tolerance_left')
        self.reached_tolerance_right = m.get('reached_tolerance_right')
        for side, val in (('left', self.reached_tolerance_left),
                          ('right', self.reached_tolerance_right)):
            if val is None:
                continue
            val = float(val)
            if not 0.0 < val <= 0.1:
                raise TaskError(f'motion.reached_tolerance_{side} 应在 (0, 0.1] rad')
            setattr(self, f'reached_tolerance_{side}', val)
        self.other_tolerance = float(m.get('other_tolerance', 0.05))
        # 服务返回后等反馈进入 reached_tolerance 的最长时间（伺服收敛/重力下沉有滞后）
        self.reached_wait_seconds = float(m.get('reached_wait_seconds', 3.0))
        if self.reached_wait_seconds < 0.5:
            raise TaskError('motion.reached_wait_seconds 不应小于 0.5s')
        # 服务完成但反馈未到位时，同目标 MoveJ 最多补发次数（每次只发同一目标，逐次收敛）
        self.reached_reissue_count = int(m.get('reached_reissue_count', 2))
        if not 0 <= self.reached_reissue_count <= 5:
            raise TaskError('motion.reached_reissue_count 应在 0~5 之间')
        # 视觉 MoveJP/MoveL 的笛卡尔到位容差（block 服务会提前返回，必须核对末端实际位姿）
        self.pose_pos_tolerance = float(m.get('pose_pos_tolerance', 0.01))
        self.pose_ori_tolerance = float(m.get('pose_ori_tolerance', 0.05))
        if not 0.0 < self.pose_pos_tolerance <= 0.05:
            raise TaskError('motion.pose_pos_tolerance 应在 (0, 0.05] m')
        if not 0.0 < self.pose_ori_tolerance <= 0.2:
            raise TaskError('motion.pose_ori_tolerance 应在 (0, 0.2] rad')

        # 命令行 --speed 整体倍率：在 yaml 校验之后统一缩放全部运动速度/加速度
        # （视觉 MoveJP/MoveL、段间接入 MoveJ、序列逐点 MoveJ）。
        self.speed_scale = float(speed_scale)
        if self.speed_scale <= 0:
            raise TaskError(f'--speed 倍率必须为正，当前 {speed_scale}')
        if self.speed_scale != 1.0:
            for key in ('speed', 'acce', 'linear_speed', 'linear_acce',
                        'join_speed', 'join_acce', 'sequence_speed', 'sequence_acce'):
                setattr(self, key, getattr(self, key) * self.speed_scale)

        # ---- 手参数 ----
        h = d('hand', {}) or {}
        self.hand_speed = _hand6(h.get('speed', [80] * 6), 'hand.speed')
        self.hand_force = _hand6(h.get('force', [100] * 6), 'hand.force')
        if h.get('open') is None:
            raise TaskError('hand.open 必填（开机张开手型；缺省会误发全 0 把手闭合）')
        self.hand_open_vals = _hand6(h['open'], 'hand.open')
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
        # 同型号多目标（如两颗中螺母）时的策略：abort=中止（安全缺省）/
        # random=随机抓一颗 / first=取检测结果列表里的第一颗
        self.duplicate_policy = str(det.get('duplicate_policy', 'abort')).lower()
        if self.duplicate_policy not in ('abort', 'random', 'first'):
            raise TaskError('detector.duplicate_policy 只能是 abort/random/first，'
                            f'当前 {det.get("duplicate_policy")!r}')
        # 整轮检测后缺型号时重新拍快照识别：最多 detect_attempts 轮（含首轮），
        # 轮间隔 missing_retry_seconds 秒。YOLO 单帧偶发漏检时不必整任务中止。
        raw_attempts = det.get('detect_attempts', 3)
        if isinstance(raw_attempts, float) and not float(raw_attempts).is_integer():
            raise TaskError(f'detector.detect_attempts 必须是整数轮数，当前 {raw_attempts!r}')
        try:
            self.detect_attempts = int(raw_attempts)
        except (TypeError, ValueError):
            raise TaskError(f'detector.detect_attempts 必须是整数，当前 {raw_attempts!r}')
        if not 1 <= self.detect_attempts <= 10:
            raise TaskError(f'detector.detect_attempts 必须在 1~10，当前 {self.detect_attempts}')
        self.missing_retry_seconds = float(det.get('missing_retry_seconds', 1.0))
        if self.missing_retry_seconds < 0:
            raise TaskError('detector.missing_retry_seconds 不能为负')
        # YOLO 识别阶段的实时弹窗（等待帧实时画面 + 检测框/深度结果）
        raw_show = det.get('show_window', False)
        if not isinstance(raw_show, bool):
            raise TaskError(f'detector.show_window 必须是 true/false，当前 {raw_show!r}')
        self.show_window = raw_show
        try:
            self.show_seconds = float(det.get('show_seconds', 2.0))
        except (TypeError, ValueError):
            raise TaskError(f'detector.show_seconds 必须是秒数，当前 {det.get("show_seconds")!r}')
        if not 0 <= self.show_seconds <= 30:
            raise TaskError('detector.show_seconds 必须在 0~30 秒（0=必须按键才继续）')
        # 每颗螺母抓取前重新拍快照识别：抓前一颗可能碰动其余螺母，旧位置不能再用。
        # 重拍前双臂自动回 ready 离场位（与开机首次识别同位姿）。
        raw_redetect = det.get('redetect_each_nut', True)
        if not isinstance(raw_redetect, bool):
            raise TaskError(f'detector.redetect_each_nut 必须是 true/false，当前 {raw_redetect!r}')
        self.redetect_each_nut = raw_redetect

    def grasp_orientation_for(self, label):
        """label 尺寸的视觉抓取姿态源 (位姿库名|None, 记录段映射|None)：覆盖 > 全局。"""
        ov = self.grasp_orientation_by_size.get(label)
        return ov if ov is not None else (self.left_grasp_pose_name, self.grasp_orientation_rec)

    def grasp_offset_for(self, label):
        """label 尺寸的检测点->腕部目标平移：grasp_by_size 覆盖 > 全局。"""
        return self.grasp_by_size.get(label, {}).get('offset_xyz', self.grasp_offset_xyz)

    def grasp_z_offset_for(self, label):
        return self.grasp_by_size.get(label, {}).get('z_offset', self.grasp_z_offset)

    def hover_height_for(self, label):
        return self.grasp_by_size.get(label, {}).get('hover_height', self.hover_height)

    def lift_height_for(self, label):
        """抓稳后相对 down 竖直上抬的高度：grasp_by_size 覆盖 > 全局（缺省 0.05m）。"""
        return self.grasp_by_size.get(label, {}).get('lift_height', self.lift_height)

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

    def ik_try(self, position, euler_rad, seed, timeout=8.0):
        """单次驱动逆解。seed 必须是长度 7 的关节角序列。
        返回 'ok' / 'fail'（服务明确返回不可解）/ 'timeout'（服务无响应）。

        ⚠ 绝不发空 joints（seed=None）：驱动源码在 request->joints.empty() 时把
        initial_ptr = nullptr 传给 lbot_inverse_kinematics，而厂商 SDK 的
        lbot_create_ik_request 会解引用它 —— 实测直接段错误，崩溃地址固定为
        liblbot_api_cpp.so[add0] == lbot_create_ik_request+0x50，整个驱动进程死掉。
        srv 注释号称空数组=驱动自读当前角，但实现不支持，必须给真种子。
        """
        status, _ = self.ik_try_full(position, euler_rad, seed, timeout)
        return status

    def ik_try_full(self, position, euler_rad, seed, timeout=8.0):
        """同 ik_try，但成功时把解出的关节角一并返回。
        返回 ('ok' | 'fail' | 'timeout', joints | None)。
        """
        if seed is None:
            raise TaskError('ik 系列不接受空种子：会把 nullptr 交给 SDK 并段错误'
                            '（lbot_create_ik_request+0x50）')
        req = self.ik_cli.srv_type.Request()
        req.position.x, req.position.y, req.position.z = [float(v) for v in position]
        req.euler.x, req.euler.y, req.euler.z = [float(v) for v in euler_rad]
        req.joints = [float(x) for x in seed]
        res = self._spin_call(self.ik_cli, req, timeout=timeout)
        if res is None:
            return 'timeout', None
        if bool(res.success):
            return 'ok', [float(v) for v in res.joints]
        return 'fail', None

    def ik_solve(self, position, euler_rad, extra_seeds=None):
        """多种子逆解并把关节角带回来：返回 (joints, 成功种子下标)，全失败 (None, None)。

        与 ik_check 同策略（当前关节角 -> 记录段臂型种子）。存在的理由：
        lbot_move_pose 内部自己做 IK 且不接受初值，解落在另一个臂型分支时会失败
        （实测中螺母中转横移：外部用记录臂型种子可解，MoveJP 却被拒）。
        拿到关节角后即可用 MoveJ 执行同一目标。
        """
        seeds = [self.joints] + [s for s in (extra_seeds or []) if s is not None]
        for idx, seed in enumerate(seeds):
            status, joints = self.ik_try_full(position, euler_rad, seed)
            if status == 'ok' and len(joints or []) == 7:
                return joints, idx
        return None, None

    def ik_check(self, position, euler_rad, extra_seeds=None):
        """可达性预检。驱动数值逆解以 joints 为初始种子，臂型离目标远时会单纯因种子
        收敛失败（而非真不可达）。故依次尝试：当前关节角 -> 调用方给的记录段臂型种子。
        返回 (是否可达, 成功种子序号/None)。

        ⚠ 种子里绝不能带 None（空 joints）：会让驱动段错误，见 ik_try 的说明。
        """
        seeds = [self.joints] + [s for s in (extra_seeds or []) if s is not None]
        for idx, seed in enumerate(seeds):
            if self.ik_try(position, euler_rad, seed) == 'ok':
                return True, idx
        return False, None

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
        diffs = self.joint_diffs(target)
        return None if diffs is None else max(diffs)

    def joint_diffs(self, target):
        """逐关节 |当前-目标|（rad），无反馈为 None。"""
        if self.joints is None:
            return None
        return [abs(a - b) for a, b in zip(self.joints, target)]

    def pose_errors(self, position, euler_rad):
        """末端反馈相对笛卡尔目标的 (位置误差 m, 姿态误差 rad, 当前 xyz)，无反馈 None。"""
        if self.pose is None:
            return None
        from scipy.spatial.transform import Rotation as Rot
        p_act, q_act = self.pose
        pos_err = float(np.linalg.norm(p_act - np.asarray(position, float)))
        q_tgt = Rot.from_euler('xyz', np.asarray(euler_rad, float)).as_quat()
        dot = float(abs(np.dot(q_act / np.linalg.norm(q_act), q_tgt)))
        ori_err = float(2.0 * np.arccos(min(1.0, max(-1.0, dot))))
        return pos_err, ori_err, np.asarray(p_act, float)
