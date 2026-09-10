#!/usr/bin/env python3
"""预录关节序列「段（leg）」的加载与回放。

段 = record_workpoints 录的 schema v3 手动序列（没有中间采样、没有 hand 事件，
只有逐点关节角）。加载复用 replay_workpoints.plan 做安全校验（关节名/坐标系一致、
另一只臂录制期间静止、序列名唯一等）。

回放安全策略（对齐 replay_workpoints.execute）：
  - 段起点接入：当前关节角 vs 段起点差 > start_tolerance 才慢速 MoveJ 补动；
  - 逐点 MoveJ，每点下发前校验反馈新鲜、另一只臂没动（漂移 <= other_tolerance）；
  - 每点到位后核对反馈进入 reached_tolerance；
  - 手动作只发在【段终点】：hand_after=open/close（close 按当前螺母尺寸给值）；
  - retreat=true：手动作后 MoveJ 回该臂 home（第一段第一个点），给另一只臂让路。
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nut_robot import TaskError

try:
    import replay_workpoints as _rp
except ImportError:  # 离线测试可能没 source ROS
    _rp = None


@dataclass
class Leg:
    arm: str                 # 'left' / 'right'
    file: Path
    target: str              # 序列 label 或 sequence_id
    hand_after: object       # None / 'open' / 'close'
    retreat: bool
    joints: np.ndarray       # (N,7) 每个记录点的关节角 rad
    names: list              # 记录中的 7 关节名
    frame: str               # 记录的 frame_id
    namespace: str
    poses: list              # [{xyz:(3,), eul_deg:(3,)}] 记录本臂 cartesian，dry-run 展示用

    @property
    def end_pose(self):
        return self.poses[-1]

    @property
    def start_pose(self):
        return self.poses[0]


def _extract_poses(route, arm_key):
    """从记录事件里读本臂每个点的 xyz/euler(deg)（base_link 系）。"""
    from scipy.spatial.transform import Rotation as Rot
    out = []
    for ev in route:
        m = ev['states'][f'{arm_key}_arm/pose_states']['message']['pose']
        p, q = m['position'], m['orientation']
        eul = Rot.from_quat([q['x'], q['y'], q['z'], q['w']]).as_euler('xyz', degrees=True)
        out.append({'xyz': np.array([p['x'], p['y'], p['z']]), 'eul_deg': eul})
    return out


def load_leg(arm, spec_tuple, check_other=True):
    """配置元组 (file,target,hand_after,retreat) -> Leg。"""
    if _rp is None:
        raise TaskError('无法 import replay_workpoints，请确认在 tools 目录且已 source 环境')
    file_, target, hand_after, retreat = spec_tuple
    file_ = Path(file_)
    if not file_.exists():
        raise TaskError(f'{arm} 段 {target} 的 trace 文件不存在：{file_}')
    try:
        p = _rp.plan(file_, target, start=None, arm=f'{arm}_arm',
                     check_other=bool(check_other))
    except ValueError as exc:
        raise TaskError(f'{arm} 段 {target} 加载失败：{exc}') from exc
    if not p.get('manual'):
        raise TaskError(f'{arm} 段 {target} 不是手动序列（schema v3）；本框架只支持逐点记录的序列')
    route = p['route']
    if len(route) < 2:
        raise TaskError(f'{arm} 段 {target} 只有 {len(route)} 个点，至少 2 点')
    joints = np.array([_rp.joints(ev, f'{arm}_arm')[0] for ev in route])
    return Leg(arm=arm, file=file_, target=target, hand_after=hand_after, retreat=retreat,
               joints=joints, names=list(p['names']), frame=p['frame'],
               namespace=p['namespace'], poses=_extract_poses(route, arm))


# ---------------- 回放器 ------------------------------------------------------

class SequenceRunner:
    """持有左右两个 RobotClient，按 Leg 回放。"""

    def __init__(self, left, right, cfg):
        self.clients = {'left': left, 'right': right}
        self.cfg = cfg
        self.homes = {}   # arm -> joints(7,)

    def set_home(self, arm, joints):
        self.homes[arm] = np.asarray(joints, float)

    def _other(self, arm):
        return self.clients['right'] if arm == 'left' else self.clients['left']

    def _wait_reached(self, robot, q, timeout=2.0):
        import rclpy
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(robot.node, timeout_sec=0.02)
            if robot.feedback_fresh(self.cfg.state_timeout):
                diff = robot.joint_diff(q)
                if diff is not None and diff <= self.cfg.reached_tolerance:
                    return True
        return False

    def join_to_start(self, leg, tag=''):
        """慢速 MoveJ 接入段起点（偏差 <= start_tolerance 则不动）。"""
        robot = self.clients[leg.arm]
        q0 = leg.joints[0]
        diff = robot.joint_diff(q0)
        if diff is None:
            raise TaskError(f'{leg.arm} 臂无关节反馈，无法接入 {leg.target}')
        if diff <= self.cfg.start_tolerance:
            print(f'    [{leg.arm}] 已在段 {leg.target} 起点（Δ={diff:.3f}），直接回放')
            return
        print(f'    [{leg.arm}] MoveJ 接入 {leg.target} 起点（最大关节差 {diff:.3f} rad，'
              f'速度 {self.cfg.join_speed}）{tag}')
        robot.move_joint(q0, self.cfg.join_speed, self.cfg.join_acce, self.cfg.move_timeout)
        if not self._wait_reached(robot, q0):
            raise TaskError(f'{leg.arm} 接入 {leg.target} 起点后反馈未进入容差')

    def run_leg(self, leg, label=None):
        """join -> 逐点 MoveJ -> 段尾手动作 -> 可选 retreat。label 决定 close 用哪档手型。"""
        cfg = self.cfg
        robot = self.clients[leg.arm]
        other = self._other(leg.arm)

        if robot.joint_names != leg.names:
            raise TaskError(
                f'{leg.arm} 实时关节名顺序与记录 {leg.target} 不一致：'
                f'实时 {robot.joint_names} vs 记录 {leg.names}')
        self.join_to_start(leg)

        other_ref = other.joints
        if other_ref is None:
            raise TaskError(f'{leg.arm} 回放 {leg.target} 时另一只臂无反馈，无法做干涉保护')
        other_ref = list(other_ref)

        for i in range(1, len(leg.joints)):
            q = leg.joints[i]
            if not robot.feedback_fresh(cfg.state_timeout):
                raise TaskError(f'{leg.arm} 回放 {leg.target} 第{i}点前反馈过期/无效')
            other_now = other.joints
            if other_now is None:
                raise TaskError(f'{leg.arm} 回放 {leg.target} 第{i}点前另一只臂反馈丢失')
            drift = max(abs(a - b) for a, b in zip(other_ref, other_now))
            if drift > cfg.other_tolerance:
                raise TaskError(
                    f'{leg.arm} 回放 {leg.target} 第{i}点时另一只臂漂移 {drift:.3f} rad '
                    f'> {cfg.other_tolerance}，疑似被带动/拖动，中止防干涉')
            robot.move_joint(q, cfg.sequence_speed, cfg.sequence_acce, cfg.move_timeout)
            if not self._wait_reached(robot, q):
                raise TaskError(
                    f'{leg.arm} {leg.target} 第{i}点服务完成但反馈未到位（>{cfg.reached_tolerance}）')

        if leg.hand_after == 'close':
            if label is None:
                raise TaskError(f'{leg.arm} 段 {leg.target} hand_after=close 但没给螺母尺寸 label')
            vals = cfg.close_for(leg.arm, label)
            print(f'    [{leg.arm}] 段尾闭合手（{label}，{vals}），'
                  f'静置 {cfg.settle_seconds:.1f}s')
            robot.hand_close(vals, settle=cfg.settle_seconds)
        elif leg.hand_after == 'open':
            print(f'    [{leg.arm}] 段尾张手释放，静置 {cfg.release_seconds:.1f}s')
            robot.hand_open(cfg.hand_open_vals)
            time.sleep(cfg.release_seconds)

        if leg.retreat:
            home = self.homes.get(leg.arm)
            if home is None:
                raise TaskError(f'{leg.arm} 没注册 home，无法 retreat')
            print(f'    [{leg.arm}] retreat 回 home')
            robot.move_joint(home, cfg.join_speed, cfg.join_acce, cfg.move_timeout)
            if not self._wait_reached(robot, home):
                raise TaskError(f'{leg.arm} retreat 回 home 后反馈未进入容差')
