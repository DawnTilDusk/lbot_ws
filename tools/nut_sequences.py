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

    def _tol(self, robot):
        """该臂到位容差：可用 reached_tolerance_left/right 覆盖全局值。"""
        return getattr(self.cfg, f'reached_tolerance_{robot.arm}', None) \
            or self.cfg.reached_tolerance

    def _wait_reached(self, robot, q, timeout=None):
        """等反馈进入 reached_tolerance。返回 (ok, best, best_diffs, still_moving)：
        窗口末尾残差仍在明显减小说明运动还在执行（block 提前返回），调用方应继续等
        而不是补发——运动中重发同目标会打断轨迹。"""
        import rclpy
        tol = self._tol(robot)
        timeout = self.cfg.reached_wait_seconds if timeout is None else timeout
        t0 = time.monotonic()
        best, best_diffs = None, None
        samples = []
        while time.monotonic() - t0 < timeout:
            now = time.monotonic()
            rclpy.spin_once(robot.node, timeout_sec=0.02)
            if robot.feedback_fresh(self.cfg.state_timeout):
                diffs = robot.joint_diffs(q)
                if diffs is not None:
                    diff = max(diffs)
                    samples.append((now, diff))
                    if best is None or diff < best:
                        best, best_diffs = diff, diffs
                    if diff <= tol:
                        return True, best, best_diffs, False
        still = self._residual_moving(
            [(t, d, 0.0) for t, d in samples], 0.005, 1.0)
        return False, best, best_diffs, still

    def _diff_report(self, robot, q, diffs):
        """把超差关节列出来（rad/度），方便定位是哪个关节没到。"""
        tol = self._tol(robot)
        names = robot.joint_names or [f'j{i}' for i in range(len(q))]
        over = sorted(((d, n) for n, d in zip(names, diffs)
                       if d > tol), reverse=True)
        return '；'.join(f'{n} {d:.3f}rad/{np.degrees(d):.1f}°' for d, n in over[:3])

    def _goto_point(self, robot, q, where, speed=None, acce=None):
        """MoveJ 到目标关节角 q，核对反馈到位。"""
        self._goto_point_core(robot, q, where, speed=speed, acce=acce)

    def _goto_point_core(self, robot, q, where, speed=None, acce=None):
        """单次 MoveJ + 到位核对 + 补发（无分段）。"""
        speed = self.cfg.sequence_speed if speed is None else speed
        acce = self.cfg.sequence_acce if acce is None else acce
        retries = int(getattr(self.cfg, 'reached_reissue_count', 2))
        tol = self._tol(robot)
        best, diffs, report = None, None, '反馈缺失'
        for attempt in range(retries + 1):
            if attempt == 0:
                robot.move_joint(q, speed, acce, self.cfg.move_timeout)
            # 残差仍在减小说明运动还没完，继续等，最多顺延 2 个等待窗，不打断运动
            for ext in range(3):
                ok, best, diffs, still = self._wait_reached(robot, q)
                if ok:
                    if attempt > 0:
                        print(f'    [{robot.arm}] {where} 第{attempt}次补发后到位'
                              f'（最好 {best:.3f}rad）')
                    return
                if not still:
                    break
                print(f'    [{robot.arm}] {where} 关节仍在收敛中'
                      f'（当前 {best:.3f}rad），继续等待，不打断运动...')
            report = self._diff_report(robot, q, diffs) if diffs else report
            if attempt < retries:
                best_txt = f'{best:.3f}' if best is not None else '无有效反馈'
                print(f'    [{robot.arm}] {where} 服务完成但反馈停在容差外'
                      f'（残差已不降，最好 {best_txt}rad > {tol}；{report}），'
                      f'补发同目标（第{attempt + 1}/{retries}次）...')
                robot.move_joint(q, speed, acce, self.cfg.move_timeout)
        best_txt = f'{best:.3f}' if best is not None else '无有效反馈'
        raise TaskError(
            f'{robot.arm} {where} 服务完成但反馈未到位（补发 {retries} 次后最好 {best_txt}rad '
            f'> {tol}rad）：{report}。'
            f'若多次重发残差几乎不变，说明是该姿态伺服稳态偏差（非收敛滞后），'
            f'可在 motion 下为该臂配 reached_tolerance_{robot.arm}（全局 reached_tolerance 0.03，'
            f'建议放到 0.04~0.05rad），或检查夹持负载/关节受力')

    def _wait_pose_reached(self, robot, position, eul):
        """MoveJP/MoveL 后核对末端实际位姿。block 服务会提前返回但运动可能仍在执行，
        所以：
          - 进入容差后还要连续保持 hold 秒才算到（防过冲瞬间误判）；
          - 窗口结束时若残差相比 0.5s 前仍在明显减小，判 still_moving=True，
            调用方应继续等而【不是补发】——运动中重发同目标会打断轨迹造成抖动。
        返回 (ok, best_pos_err, best_ori_err, best_xyz, still_moving)。"""
        import rclpy
        bp, bo, bx = None, None, None
        samples = []          # (t, pos_err, ori_err)
        in_tol_since = None
        hold = 0.15
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.cfg.reached_wait_seconds:
            now = time.monotonic()
            rclpy.spin_once(robot.node, timeout_sec=0.02)
            errs = robot.pose_errors(position, eul)
            if errs is not None:
                pe, oe, px = errs
                samples.append((now, pe, oe))
                if bp is None or pe < bp:
                    bp, bo, bx = pe, oe, px
                if pe <= self.cfg.pose_pos_tolerance and oe <= self.cfg.pose_ori_tolerance:
                    if in_tol_since is None:
                        in_tol_since = now
                    elif now - in_tol_since >= hold:
                        return True, bp, bo, bx, False
                else:
                    in_tol_since = None
        still = self._residual_moving(samples, 0.005, 0.02)
        return False, bp, bo, bx, still

    @staticmethod
    def _residual_moving(samples, slack_pos, slack_ori, lookback=0.5):
        """窗口末尾残差是否仍在明显减小（位置或姿态改善超过 slack）。"""
        if len(samples) < 2:
            return False
        t_last = samples[-1][0]
        old = next((s for s in samples if t_last - s[0] >= lookback), samples[0])
        _, pe0, oe0 = old
        _, pe1, oe1 = samples[-1]
        return (pe0 - pe1) > slack_pos or (oe0 - oe1) > slack_ori

    def _goto_pose(self, robot, position, eul, where, linear=False):
        """视觉笛卡尔运动（MoveJP/MoveL）到位核对：服务返回后等末端反馈进入
        pose_pos/ori_tolerance，超差补发同目标（与 _goto_point 同策略），
        每次都打印指令位与实际位，便于发现“相同坐标落点不一致”。"""
        retries = int(self.cfg.reached_reissue_count)
        if linear:
            send = lambda: robot.move_linear(
                position, eul, self.cfg.linear_speed, self.cfg.linear_acce,
                self.cfg.move_timeout)
            kind = 'MoveL'
        else:
            send = lambda: robot.move_pose(
                position, eul, self.cfg.speed, self.cfg.acce, self.cfg.move_timeout)
            kind = 'MoveJP'
        best = (None, None, None)
        for attempt in range(retries + 1):
            if attempt == 0:
                send()
            # block 服务常提前返回：运动未结束时绝不能补发（会打断轨迹造成抖动），
            # 残差仍在减小就继续等，最多顺延 2 个等待窗；只有残差平台化才重发同目标。
            for ext in range(3):
                ok, bp, bo, bx, still = self._wait_pose_reached(robot, position, eul)
                best = (bp, bo, bx)
                if bp is None:
                    raise TaskError(
                        f'{robot.arm} {where} {kind} 后一直无末端位姿反馈，无法核对到位')
                if ok:
                    if attempt > 0:
                        print(f'    [{robot.arm}] {where} 第{attempt}次补发后到位'
                              f'（实际 {np.round(bx * 1000, 1)}mm，位置差 {bp * 1000:.1f}mm）')
                    return
                if not still:
                    break
                print(f'    [{robot.arm}] {where} {kind} 末端仍在收敛中'
                      f'（当前位置差 {bp * 1000:.1f}mm），继续等待，不打断运动...')
            if attempt < retries:
                print(f'    [{robot.arm}] {where} {kind} 完成但末端停在容差外'
                      f'（残差已不降：位置{bp * 1000:.1f}mm>{self.cfg.pose_pos_tolerance * 1000:.0f}mm '
                      f'姿态{np.degrees(bo):.1f}°>{np.degrees(self.cfg.pose_ori_tolerance):.1f}°；'
                      f'实际 {np.round(bx * 1000, 1)}mm，指令 {np.round(np.asarray(position) * 1000, 1)}mm），'
                      f'补发同目标（第{attempt + 1}/{retries}次）...')
                send()
        bp, bo, bx = best
        raise TaskError(
            f'{robot.arm} {where} {kind} 补发 {retries} 次后末端仍未到位：'
            f'实际 {np.round(bx * 1000, 1)}mm vs 指令 {np.round(np.asarray(position) * 1000, 1)}mm，'
            f'位置差 {bp * 1000:.1f}mm（容差 {self.cfg.pose_pos_tolerance * 1000:.0f}mm），'
            f'姿态差 {np.degrees(bo):.1f}°（容差 {np.degrees(self.cfg.pose_ori_tolerance):.1f}°）。'
            f'block 服务提前返回且该姿态伺服有稳态偏差，已重发同目标仍补不齐；'
            f'检查该姿态受力/驱动规划，或适当放宽 motion.pose_pos_tolerance')

    def check_names(self, leg):
        """实时关节名顺序必须与记录一致，否则 MoveJ 目标会错位。"""
        robot = self.clients[leg.arm]
        if robot.joint_names != leg.names:
            raise TaskError(
                f'{leg.arm} 实时关节名顺序与记录 {leg.target} 不一致：'
                f'实时 {robot.joint_names} vs 记录 {leg.names}')

    def _dwell(self, seconds, tag=''):
        """停顿时持续 spin：否则反馈回调不执行，超过 state_timeout 会被误判为反馈过期。"""
        if seconds > 0:
            print(f'      停顿 {seconds:.1f}s {tag}')
            import rclpy
            node = self.clients['left'].node  # 左右 client 共用同一个任务节点
            t0 = time.monotonic()
            while True:
                remain = seconds - (time.monotonic() - t0)
                if remain <= 0:
                    break
                rclpy.spin_once(node, timeout_sec=min(0.05, remain))

    def join_to_start(self, leg, tag=''):
        """慢速 MoveJ 接入段起点（偏差 <= start_tolerance 则不动）。
        若臂已在段末点附近（上轮已跑完），则跳过整段。"""
        self.check_names(leg)
        robot = self.clients[leg.arm]
        q0 = leg.joints[0]
        diff0 = robot.joint_diff(q0)
        if diff0 is None:
            raise TaskError(f'{leg.arm} 臂无关节反馈，无法接入 {leg.target}')
        if diff0 <= self.cfg.start_tolerance:
            print(f'    [{leg.arm}] 已在段 {leg.target} 起点（Δ={diff0:.3f}），直接回放')
            return 0  # 从 pt0 开始
        # 已在末点？跳过整段
        q_last = leg.joints[-1]
        diff_last = robot.joint_diff(q_last)
        if diff_last is not None and diff_last <= self.cfg.start_tolerance:
            print(f'    [{leg.arm}] 已在段 {leg.target} 末点（Δ={diff_last:.3f}），跳过整段')
            return len(leg.joints) - 1  # 跳过所有点
        print(f'    [{leg.arm}] MoveJ 接入 {leg.target} 起点（最大关节差 {diff0:.3f} rad，'
              f'速度 {self.cfg.join_speed}）{tag}')
        self._goto_point(robot, q0, f'接入 {leg.target} 起点',
                         speed=self.cfg.join_speed, acce=self.cfg.join_acce)
        self._dwell(self.cfg.point_dwell_seconds, '接入完成')
        return 0

    def run_leg(self, leg, label=None):
        """join -> 逐点 MoveJ -> 段尾手动作 -> 可选 retreat。label 决定 close 用哪档手型。"""
        cfg = self.cfg
        robot = self.clients[leg.arm]
        other = self._other(leg.arm)

        self.check_names(leg)
        start_idx = self.join_to_start(leg)
        if start_idx >= len(leg.joints) - 1:
            # 已在末点，跳过整段回放（段尾手动作/retreat 仍执行）
            print(f'    [{leg.arm}] {leg.target} 已到位，跳过逐点回放')
        self._dwell(cfg.between_leg_seconds, f'进入段 {leg.target}')

        other_ref = other.joints
        if other_ref is None:
            raise TaskError(f'{leg.arm} 回放 {leg.target} 时另一只臂无反馈，无法做干涉保护')
        other_ref = list(other_ref)

        for i in range(start_idx + 1, len(leg.joints)):
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
            self._goto_point(robot, q, f'回放 {leg.target} 第{i}点')
            self._dwell(cfg.point_dwell_seconds, f'{leg.target} pt{i} 到位')

        if leg.hand_after:
            self._dwell(cfg.pre_hand_seconds, '手部动作前')
        if leg.hand_after == 'close':
            if label is None:
                raise TaskError(f'{leg.arm} 段 {leg.target} hand_after=close 但没给螺母尺寸 label')
            vals = cfg.close_for(leg.arm, label)
            print(f'    [{leg.arm}] 段尾闭合手（{label}，{vals}），'
                  f'静置 {cfg.settle_seconds:.1f}s')
            robot.hand_close(vals, settle=cfg.settle_seconds)
        elif leg.hand_after == 'open':
            print(f'    [{leg.arm}] 段尾张手释放（{cfg.hand_open_vals}），'
                  f'静置 {cfg.release_seconds:.1f}s')
            robot.hand_open(cfg.hand_open_vals)
            self._dwell(cfg.release_seconds, '张手释放')

        if leg.retreat:
            self._dwell(cfg.between_leg_seconds, 'retreat 前')
            home = self.homes.get(leg.arm)
            if home is None:
                raise TaskError(f'{leg.arm} 没注册 home，无法 retreat')
            print(f'    [{leg.arm}] retreat 回 home')
            self._goto_point(robot, home, 'retreat 回 home',
                             speed=cfg.join_speed, acce=cfg.join_acce)
