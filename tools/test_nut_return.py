#!/usr/bin/env python3
"""nut_return 安全回退离线测试（不连机器人/相机；用真实预录 ready 段加载）：

  cd tools
  /usr/bin/python3 -m unittest -v test_nut_return.py
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

import nut_return
from nut_robot import DEFAULT_CONFIG, TaskConfig, TaskError
from nut_sequences import load_leg
from nut_return import (ARM_CN, _run_arm_return, _slerp, build_reapproach_path,
                        max_joint_diff, nearest_index, pick_transit_z,
                        plan_arm_return, plan_reapproach, polyline_samples,
                        render_arm_plan, render_reapproach, reverse_path)

WORKSPACE = Path(__file__).resolve().parent.parent
LEFT_TRACE = WORKSPACE / 'recordings/left_trace/events.jsonl'
RIGHT_TRACE = WORKSPACE / 'recordings/right_trace/events.jsonl'
LEFT_READY = WORKSPACE / 'recordings/20260915_175547_556799/events.jsonl'
RIGHT_READY = WORKSPACE / 'recordings/20260915_134046_684005/events.jsonl'

CONFIG_YAML = f"""
namespace: /robot1
arm: right
order: [l, m, s]
require_all: true
check_other_arm: true
left:
  grasp_orientation: left_grasp_init
  trace: {LEFT_TRACE}
  ready: {{file: {LEFT_READY}, sequence: left-ready_001}}
  legs:
    - {{sequence: left_ready1_001}}
    - {{sequence: left_place1_002, hand_after: open, retreat: true}}
right:
  trace: {RIGHT_TRACE}
  ready: {{file: {RIGHT_READY}, sequence: right-ready_001}}
  approach: {{sequence: right_ready1_001, hand_after: close}}
  place:
    l: {{sequence: right_place1_002, hand_after: open, retreat: true}}
    m: {{sequence: right_place2_003, hand_after: open, retreat: true}}
    s: {{sequence: right_place3_004, hand_after: open, retreat: true}}
motion:
  speed: 0.3
  acce: 0.3
  linear_speed: 0.2
  linear_acce: 0.2
  hover_height: 0.10
  grasp_z_offset: 0.0
  join_speed: 0.15
  join_acce: 0.15
  sequence_speed: 0.3
  sequence_acce: 0.5
  point_dwell_seconds: 0.0
hand:
  speed: [80, 80, 80, 80, 80, 80]
  force: [100, 100, 100, 100, 100, 100]
  open: [0, 0, 0, 0, 0, 0]
vision: {{}}
detector: {{type: json, json_path: detections.json}}
"""


def _leg_joints(n=3):
    """合成 (n,7) 关节序列，相邻点差 0.2 rad 便于区分下标。"""
    return np.array([[0.2 * i + 0.01 * j for j in range(7)] for i in range(n)], float)


class _Arm:
    def __init__(self, arm, joints, names=None):
        self.arm = arm
        self.joints = [float(x) for x in joints]
        self.joint_names = list(names) if names else [f'j{i}' for i in range(7)]

    def feedback_fresh(self, timeout):
        return True

    def joint_diffs(self, target):
        return [abs(a - b) for a, b in zip(self.joints, target)]

    def joint_diff(self, target):
        return max(self.joint_diffs(target))


class _Runner:
    """替身 runner：只记录下发目标，不做任何反馈等待。"""

    def __init__(self, on_goto=None):
        self.calls = []
        self.dwells = []
        self.on_goto = on_goto

    def _goto_point(self, robot, q, where, speed=None, acce=None):
        self.calls.append(([float(x) for x in q], where, speed, acce))
        robot.joints = [float(x) for x in q]
        if self.on_goto:
            self.on_goto(len(self.calls))

    def _dwell(self, seconds, tag=''):
        self.dwells.append((seconds, tag))


def _cfg_stub(**kw):
    base = dict(state_timeout=0.5, other_tolerance=0.05, start_tolerance=0.05,
                point_dwell_seconds=0.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


class ReversePathTest(unittest.TestCase):
    def test_reverses_order_and_copies(self):
        q = _leg_joints(3)
        r = reverse_path(q)
        np.testing.assert_allclose(r, q[::-1])
        self.assertIsNot(r, q)
        r[0, 0] = 99.0                      # 不修改输入
        self.assertNotEqual(q[-1, 0], 99.0)

    def test_rejects_bad_shape(self):
        with self.assertRaises(TaskError):
            reverse_path(np.zeros((3, 6)))
        with self.assertRaises(TaskError):
            reverse_path(np.zeros((1, 7)))   # 少于 2 点
        with self.assertRaises(TaskError):
            reverse_path(np.full((2, 7), np.nan))

    def test_max_joint_diff_and_nearest(self):
        path = reverse_path(_leg_joints(3))
        self.assertAlmostEqual(max_joint_diff(path[0], path[0]), 0.0)
        idx, diff = nearest_index(path, _leg_joints(3)[1])
        self.assertEqual(idx, 1)            # pt1 在逆向链的第 1 位
        self.assertAlmostEqual(diff, 0.0)


class PlanArmReturnTest(unittest.TestCase):
    def setUp(self):
        self.leg = _leg_joints(3)

    def test_already_home(self):
        plan = plan_arm_return(self.leg, self.leg[0], 0.10)
        self.assertEqual(plan['action'], 'already_home')
        self.assertAlmostEqual(plan['home_diff'], 0.0)

    def test_from_ready_end_reverses_whole_path(self):
        plan = plan_arm_return(self.leg, self.leg[-1], 0.10)
        self.assertEqual(plan['action'], 'reverse')
        self.assertEqual(plan['start_index'], 0)
        np.testing.assert_allclose(plan['target'], self.leg[0])

    def test_resumes_mid_path(self):
        plan = plan_arm_return(self.leg, self.leg[1], 0.10)
        self.assertEqual(plan['action'], 'reverse')
        self.assertEqual(plan['start_index'], 1)

    def test_refuses_when_off_path(self):
        away = self.leg[1] + 0.5
        plan = plan_arm_return(self.leg, away, 0.10)
        self.assertEqual(plan['action'], 'refuse')
        self.assertIn('不在 ready 轨迹容差', plan['reason'])

    def test_blind_join_when_allowed(self):
        away = self.leg[1] + 0.5
        plan = plan_arm_return(self.leg, away, 0.10, allow_blind_join=True)
        self.assertEqual(plan['action'], 'blind_join')
        self.assertEqual(plan['start_index'], 0)

    def test_refuses_without_feedback(self):
        plan = plan_arm_return(self.leg, None, 0.10)
        self.assertEqual(plan['action'], 'refuse')
        plan = plan_arm_return(self.leg, [0.0] * 6, 0.10)
        self.assertEqual(plan['action'], 'refuse')


class RenderPlanTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        yaml_path = Path(tmp.name) / 'nut_task.yaml'
        yaml_path.write_text(CONFIG_YAML, encoding='utf-8')
        self.cfg = TaskConfig(yaml_path)
        self.leg = load_leg('left', self.cfg.left_ready, check_other=True)

    def test_reverse_lines(self):
        plan = plan_arm_return(self.leg.joints, self.leg.joints[-1], 0.10)
        text = '\n'.join(render_arm_plan('left', self.leg, plan, 0.10, 0.2, 0.2,
                                         self.leg.names))
        self.assertIn('逆向回放链', text)
        self.assertIn('从逆向链第 0 步继续', text)
        self.assertIn('pt0', text)

    def test_already_home_and_refuse_lines(self):
        plan = plan_arm_return(self.leg.joints, self.leg.joints[-1] + 0.5, 0.10)
        text = '\n'.join(render_arm_plan('left', self.leg, plan, 0.10, 0.2, 0.2,
                                         self.leg.names))
        self.assertIn('不在 ready 轨迹上', text)
        self.assertIn('相对逆向链最近点', text)      # 列出是哪几个关节差多少
        self.assertIn('Left_Shoulder_Pitch_Joint', text)
        plan = plan_arm_return(self.leg.joints, self.leg.joints[0], 0.10)
        text = '\n'.join(render_arm_plan('left', self.leg, plan, 0.10, 0.2, 0.2))
        self.assertIn('已在安全起始位', text)

    def test_blind_join_lines(self):
        plan = plan_arm_return(self.leg.joints, self.leg.joints[-1] + 0.5, 0.10,
                               allow_blind_join=True)
        text = '\n'.join(render_arm_plan('left', self.leg, plan, 0.10, 0.2, 0.2))
        self.assertIn('--blind-join', text)
        self.assertIn('无避障盲动', text)


class SlerpAndPathTest(unittest.TestCase):
    def test_slerp_endpoints_and_midpoint(self):
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = Rot.from_euler('xyz', [0.0, 0.0, np.pi / 2]).as_quat()
        np.testing.assert_allclose(_slerp(q0, q1, 0.0), q0, atol=1e-9)
        np.testing.assert_allclose(_slerp(q0, q1, 1.0), q1, atol=1e-9)
        mid = _slerp(q0, q1, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(mid)), 1.0, places=9)
        self.assertAlmostEqual(float(Rot.from_quat(mid).as_euler('xyz')[2]),
                               np.pi / 4, places=6)
        # 反号四元数表示同一姿态：仍走短弧
        np.testing.assert_allclose(_slerp(q0, -q1, 1.0), q1, atol=1e-9)

    def test_build_path_below_and_above_transit(self):
        goal = np.array([0.2, 0.5, -0.22])
        below = build_reapproach_path(np.array([0.35, -0.02, -0.35]), goal, -0.19)
        self.assertEqual(len(below), 4)                     # 上 -> 横移 -> 下
        self.assertAlmostEqual(below[1][2], -0.19)
        self.assertAlmostEqual(below[2][2], -0.19)
        np.testing.assert_allclose(below[-1], goal)
        above = build_reapproach_path(np.array([0.26, 0.40, -0.15]), goal, -0.19)
        self.assertEqual(len(above), 3)                     # 已在中转平面上方：不上升
        self.assertAlmostEqual(above[1][2], -0.19)          # 横移仍降到中转平面
        same_xy = build_reapproach_path(np.array([0.2, 0.5, -0.35]), goal, -0.19)
        self.assertEqual(len(same_xy), 3)                   # 同列：只有上/下

    def test_polyline_samples_arc_length_and_goal(self):
        pts = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]),
               np.array([0.3, 0.0, 0.1])]
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = Rot.from_euler('xyz', [0.0, 0.0, 1.0]).as_quat()
        s = polyline_samples(pts, q0, q1, step=0.025)
        self.assertTrue(len(s) >= 4)
        np.testing.assert_allclose(s[-1][0], pts[-1], atol=1e-9)   # 终点精确
        np.testing.assert_allclose(s[-1][1], q1, atol=1e-9)        # 姿态到位
        # 竖直段 100mm/25mm=4 步，横移段 300mm/50mm=6 步
        self.assertEqual(len(s), 10)
        # 姿态沿弧长单调推进：转角逐步增大
        angles = [float(Rot.from_quat(q).as_euler('xyz')[2]) for _, q in s]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(angles, angles[1:])))
        self.assertAlmostEqual(angles[-1], 1.0, places=6)

    def test_polyline_samples_degenerate(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        self.assertEqual(polyline_samples([np.zeros(3), np.zeros(3)], q, q), [])

    def test_pick_transit_z_first_hit_and_lowering(self):
        def ok_first(_p):
            return True, 'seed'
        z, tried, who = pick_transit_z(ok_first, 0.2, 0.5, -0.19, -0.24, 0.025)
        self.assertAlmostEqual(z, -0.19)
        self.assertEqual(tried, 1)

        calls = {'n': 0}

        def ok_third(_p):
            calls['n'] += 1
            return (calls['n'] >= 3), 'seed'

        z, tried, who = pick_transit_z(ok_third, 0.2, 0.5, -0.19, -0.24, 0.025)
        self.assertAlmostEqual(z, -0.24)
        self.assertEqual(tried, 3)

        z, tried, who = pick_transit_z(lambda p: (False, None), 0.2, 0.5, -0.19, -0.24, 0.025)
        self.assertIsNone(z)
        self.assertEqual(tried, 3)


class _FakeRobot:
    """plan_reapproach 用的假臂：可注入 ik_solve / ik_check / fk_solve 行为。"""

    def __init__(self, joints, pose, ik_solve=None, ik_check=None, fk=None,
                 arm='left'):
        self.arm = arm
        self.joints = [float(v) for v in joints]
        self.pose = pose
        self._solve = ik_solve
        self._check = ik_check
        self._fk = fk

    def ik_solve(self, position, euler_rad, extra_seeds=None, prefer_current=True):
        extra = list(extra_seeds or [])
        if self._solve is not None:
            return self._solve(position, extra)
        for s in extra:
            if s is not None and len(s) == 7:
                return [float(v) for v in s], 0
        return [0.0] * 7, 0

    def ik_check(self, position, euler_rad, extra_seeds=None):
        if self._check is not None:
            return self._check(position, list(extra_seeds or []))
        return True, 0

    def fk_solve(self, joints, timeout=8.0):
        if self._fk is not None:
            return self._fk(joints)
        return np.array([0.25, 0.40, -0.15]), np.zeros(3)


class PlanReapproachTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.yaml_path = Path(self.tmp.name) / 'nut_task.yaml'
        self.yaml_path.write_text(CONFIG_YAML, encoding='utf-8')
        self.cfg = TaskConfig(self.yaml_path)
        self.leg = load_leg('left', self.cfg.left_ready, check_other=True)
        self.goal = np.asarray(self.leg.poses[-1]['xyz'], float)
        self.ready_end = np.asarray(self.leg.joints[-1], float)

    def _pose(self):
        q = Rot.from_euler('xyz', np.radians(self.leg.poses[-1]['eul_deg'])).as_quat()
        return (self.goal.copy(), q)

    def test_route_a_when_reachable_and_ends_at_ready_end(self):
        robot = _FakeRobot(self.ready_end + 0.05, self._pose())
        rp = plan_reapproach(robot, self.leg, self.cfg,
                             seeds=[self.leg.joints[0], self.leg.joints[-1]],
                             seed_names=['pt0', '末点'])
        self.assertEqual(rp['route'], 'cartesian')
        np.testing.assert_allclose(rp['joints'][-1], self.ready_end, atol=1e-9)
        self.assertLess(rp['branch_gap'], 0.2)
        self.assertIsNotNone(rp['fk']['z_min'])
        self.assertGreaterEqual(rp['transit_z'], self.goal[2] - 1e-9)
        self.assertEqual(len(rp['joints']), len(rp['samples']))
        for xyz, _q in rp['samples']:
            self.assertGreater(xyz[2], self.cfg.return_table_z)

    def test_transit_lowering_uses_goal_column_ik(self):
        tried = {'n': 0}

        def check(_pt, _seeds):
            tried['n'] += 1
            return tried['n'] >= 2, 0

        robot = _FakeRobot(self.ready_end + 0.05, self._pose(), ik_check=check)
        rp = plan_reapproach(robot, self.leg, self.cfg)
        self.assertEqual(rp['route'], 'cartesian')
        self.assertEqual(rp['tried'], 2)
        self.assertLess(rp['transit_z'], self.cfg.return_table_z
                        + self.cfg.return_transit_clearance + 1e-9)

    def test_falls_back_to_joint_route_when_transit_unreachable(self):
        """路线A 定中转平面就失败 -> 改走关节空间整段，且必然落在记录臂型上。"""
        robot = _FakeRobot(self.ready_end + 0.5, self._pose(),
                           ik_check=lambda p, s: (False, None))
        rp = plan_reapproach(robot, self.leg, self.cfg)
        self.assertEqual(rp['route'], 'joint')
        self.assertIn('逆解全部失败', rp['route_note'])
        np.testing.assert_allclose(rp['joints'][-1], self.ready_end, atol=1e-9)
        self.assertEqual(rp['branch_gap'], 0.0)
        self.assertGreater(rp['lerp_steps'], 0)
        self.assertGreater(rp['fk']['z_min'],
                           self.cfg.return_table_z + self.cfg.return_joint_min_clear - 1e-9)
        lerp = np.asarray(rp['joints'][-rp['lerp_steps']:], float)
        if len(lerp) > 1:
            steps = np.degrees(np.max(np.abs(np.diff(lerp, axis=0))))
            self.assertLessEqual(float(steps), self.cfg.return_joint_step_deg + 1e-6)

    def test_both_routes_fail_aborts(self):
        robot = _FakeRobot(self.ready_end + 0.05, self._pose(),
                           ik_solve=lambda p, e: (None, None),
                           ik_check=lambda p, s: (False, None))
        with self.assertRaises(TaskError) as ctx:
            plan_reapproach(robot, self.leg, self.cfg)
        self.assertIn('两条路线都不可行', str(ctx.exception))

    def test_fk_gate_rejects_low_sweep(self):
        """关节整段 FK 校验发现工具低于桌面安全面 -> 中止。"""
        robot = _FakeRobot(self.ready_end + 0.05, self._pose(),
                           ik_check=lambda p, s: (False, None),
                           fk=lambda q: (np.array([0.25, 0.40, -0.40]), None))
        with self.assertRaises(TaskError) as ctx:
            plan_reapproach(robot, self.leg, self.cfg)
        self.assertIn('正解校验不通过', str(ctx.exception))

    def test_fk_gate_rejects_wild_step(self):
        """FK 校验发现"一步窜很远"的近奇异位形 -> 中止。"""
        seq = {'n': 0}

        def fk(_q):
            seq['n'] += 1
            z = -0.15 if seq['n'] % 2 else 0.20
            return (np.array([0.25, 0.40, z]), None)

        robot = _FakeRobot(self.ready_end + 0.5, self._pose(),
                           ik_check=lambda p, s: (False, None), fk=fk)
        with self.assertRaises(TaskError) as ctx:
            plan_reapproach(robot, self.leg, self.cfg)
        self.assertIn('近奇异', str(ctx.exception))

    def test_missing_pose_aborts(self):
        robot = _FakeRobot(self.ready_end, None)
        with self.assertRaises(TaskError) as ctx:
            plan_reapproach(robot, self.leg, self.cfg)
        self.assertIn('臂未移动', str(ctx.exception))

    def test_render_reapproach_text_both_routes(self):
        robot = _FakeRobot(self.ready_end + 0.05, self._pose())
        rp_a = plan_reapproach(robot, self.leg, self.cfg)
        text = '\n'.join(render_reapproach('left', rp_a, self.cfg))
        self.assertIn('安全再接近', text)
        self.assertIn('中转平面', text)
        self.assertIn('路线A', text)
        self.assertIn('驱动正解复核', text)
        robot_b = _FakeRobot(self.ready_end + 0.05, self._pose(),
                             ik_check=lambda p, s: (False, None))
        rp_b = plan_reapproach(robot_b, self.leg, self.cfg)
        text = '\n'.join(render_reapproach('left', rp_b, self.cfg))
        self.assertIn('路线B', text)
        self.assertIn('关节空间整段', text)
        self.assertIn('驱动正解校验', text)


class ReturnConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.yaml_path = Path(self.tmp.name) / 'nut_task.yaml'
        self.yaml_path.write_text(CONFIG_YAML, encoding='utf-8')
        self.cfg = TaskConfig(self.yaml_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _reload(self, return_value):
        y = yaml.safe_load(CONFIG_YAML)
        if return_value is not None:
            y['return'] = return_value
        self.yaml_path.write_text(yaml.safe_dump(y, allow_unicode=True), encoding='utf-8')
        return TaskConfig(self.yaml_path)

    def test_defaults(self):
        self.assertEqual(self.cfg.return_order, ('left', 'right'))
        self.assertAlmostEqual(self.cfg.return_speed, 0.2)
        self.assertAlmostEqual(self.cfg.return_acce, 0.2)
        self.assertAlmostEqual(self.cfg.return_tolerance, 0.10)
        self.assertTrue(self.cfg.return_open_hand)
        self.assertFalse(self.cfg.return_allow_blind_join)

    def test_overrides(self):
        cfg = self._reload({'order': ['right'], 'speed': 0.1, 'acce': 0.1,
                            'tolerance': 0.2, 'open_hand': False,
                            'allow_blind_join': True})
        self.assertEqual(cfg.return_order, ('right',))
        self.assertAlmostEqual(cfg.return_tolerance, 0.2)
        self.assertFalse(cfg.return_open_hand)
        self.assertTrue(cfg.return_allow_blind_join)

    def test_bad_values_rejected(self):
        for bad in ({'speed': 0.5}, {'acce': 0}, {'tolerance': 0.4},
                    {'tolerance': 0}, {'order': []}, {'order': ['left', 'up']},
                    {'order': ['left', 'left']}, {'open_hand': 'yes'},
                    {'allow_blind_join': 1}, 'nope'):
            with self.assertRaises(TaskError, msg=f'{bad!r} 应被拒绝'):
                self._reload(bad)


class RealReadyLegsTest(unittest.TestCase):
    """工作区真实配置：默认 return 段 + 真实 ready 段能直接规划。"""

    def setUp(self):
        self.cfg = TaskConfig(DEFAULT_CONFIG)

    def test_ready_configured(self):
        self.assertIsNotNone(self.cfg.left_ready)
        self.assertIsNotNone(self.cfg.right_ready)

    def test_plan_from_ready_end_for_both_arms(self):
        for arm in ('left', 'right'):
            spec = self.cfg.left_ready if arm == 'left' else self.cfg.right_ready
            leg = load_leg(arm, spec, check_other=self.cfg.check_other)
            self.assertGreaterEqual(len(leg.joints), 2)
            rev = reverse_path(leg.joints)
            self.assertEqual(rev.shape, (len(leg.joints), 7))
            np.testing.assert_allclose(rev[-1], leg.joints[0])
            plan = plan_arm_return(leg.joints, leg.joints[-1],
                                   self.cfg.return_tolerance)
            self.assertEqual(plan['action'], 'reverse')
            self.assertEqual(plan['start_index'], 0)
            plan = plan_arm_return(leg.joints, leg.joints[0],
                                   self.cfg.return_tolerance)
            self.assertEqual(plan['action'], 'already_home')


class RunArmReturnTest(unittest.TestCase):
    def setUp(self):
        self.leg = _leg_joints(3)
        self.plan = plan_arm_return(self.leg, self.leg[-1], 0.10)
        self.cfg = _cfg_stub()

    def test_replays_to_target(self):
        rc = _Arm('left', self.leg[-1])
        other = _Arm('right', np.zeros(7))
        runner = _Runner()
        _run_arm_return(runner, rc, other, self.plan, self.cfg, 0.2, 0.2)
        self.assertEqual(len(runner.calls), 2)          # 3 点段 -> 逆向 2 步
        np.testing.assert_allclose(rc.joints, self.leg[0])
        self.assertEqual(runner.calls[0][2:], (0.2, 0.2))
        self.assertTrue(all(c[1].startswith('逆向回退第') for c in runner.calls))

    def test_skips_points_already_within_tolerance(self):
        # 从逆向链第 1 步开始，而臂已停在该点容差内：该点跳过，只补最后一步
        plan = plan_arm_return(self.leg, self.leg[1], 0.10)
        self.assertEqual(plan['start_index'], 1)
        rc = _Arm('left', self.leg[1])
        other = _Arm('right', np.zeros(7))
        runner = _Runner()
        _run_arm_return(runner, rc, other, plan, self.cfg, 0.2, 0.2)
        self.assertEqual(len(runner.calls), 1)
        np.testing.assert_allclose(rc.joints, self.leg[0])

    def test_already_home_never_moves(self):
        plan = plan_arm_return(self.leg, self.leg[0], 0.10)
        self.assertEqual(plan['action'], 'already_home')
        self.assertEqual(plan['start_index'], 2)      # 已在逆向链末点
        self.assertAlmostEqual(plan['home_diff'], 0.0)
        # 上层只对 reverse/blind_join 进入回放循环，already_home 一步都不发
        self.assertNotIn(plan['action'], ('reverse', 'blind_join'))

    def test_other_arm_drift_aborts(self):
        rc = _Arm('left', self.leg[-1])
        other = _Arm('right', np.zeros(7))

        def drift(_count):
            other.joints = [0.5] + [0.0] * 6

        runner = _Runner(on_goto=drift)
        with self.assertRaises(TaskError) as ctx:
            _run_arm_return(runner, rc, other, self.plan, self.cfg, 0.2, 0.2)
        self.assertIn('另一只臂漂移', str(ctx.exception))

    def test_missing_other_feedback_aborts(self):
        rc = _Arm('left', self.leg[-1])
        other = _Arm('right', np.zeros(7))
        other.joints = None
        with self.assertRaises(TaskError):
            _run_arm_return(_Runner(), rc, other, self.plan, self.cfg, 0.2, 0.2)


class ArmNameTableTest(unittest.TestCase):
    def test_cn_names(self):
        self.assertEqual(ARM_CN['left'], '左')
        self.assertEqual(ARM_CN['right'], '右')


class RunOrchestrationTest(unittest.TestCase):
    """用假 rclpy + 假 RobotClient 跑 _run 全链路：使能/张手/逐臂回放/拒绝策略。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.yaml_path = Path(self.tmp.name) / 'nut_task.yaml'
        self.yaml_path.write_text(CONFIG_YAML, encoding='utf-8')
        self.cfg = TaskConfig(self.yaml_path)
        self.legs = {}
        for arm in ('left', 'right'):
            spec = self.cfg.left_ready if arm == 'left' else self.cfg.right_ready
            self.legs[arm] = load_leg(arm, spec, check_other=True)

    def _run(self, starts, execute=True, arm='both', keep_hand=False, ik_fail=False,
             solve_fail=False, fk_low=False, fk_wild=False):
        import nut_robot
        legs, instances = self.legs, {}

        class FakeClient:
            def __init__(self, node, namespace, a):
                self.arm = a
                self.node = node
                self.joint_names = list(legs[a].names)
                self.joints = [float(x) for x in starts[a]]
                self.pose = (np.asarray(legs[a].poses[-1]['xyz'], float),
                             np.array([0.0, 0.0, 0.0, 1.0]))
                self.enabled = False
                self.hand_opens = 0
                self.moves = []
                instances[a] = self

            def wait_services(self, timeout=5.0):
                return True

            def wait_state(self, timeout=3.0):
                return True

            def enable(self):
                self.enabled = True

            def hand_setup(self, speed, force):
                pass

            def hand_open(self, values):
                self.hand_opens += 1

            def feedback_fresh(self, timeout):
                return True

            def joint_diffs(self, target):
                return [abs(a - b) for a, b in zip(self.joints, target)]

            def joint_diff(self, target):
                return max(self.joint_diffs(target))

            def ik_check(self, position, euler_rad, extra_seeds=None):
                return (not ik_fail), 0


            def ik_solve(self, position, euler_rad, extra_seeds=None,
                         prefer_current=True):
                if solve_fail:
                    return None, None
                for s in (extra_seeds or []):
                    if s is not None and len(s) == 7:
                        return [float(v) for v in s], 0
                return [0.0] * 7, 0

            def fk_solve(self, joints, timeout=8.0):
                self.fk_calls = getattr(self, 'fk_calls', 0) + 1
                if fk_wild and self.fk_calls % 2 == 0:
                    return np.array([0.3, 0.0, 0.2]), np.zeros(3)
                z = -0.40 if fk_low else -0.15
                return np.array([0.3, 0.0, z]), np.zeros(3)

            def move_joint(self, q, speed=0.0, acce=0.0, timeout=None):
                self.moves.append([float(x) for x in q])
                self.joints = [float(x) for x in q]

        fake_rclpy = types.ModuleType('rclpy')
        fake_rclpy.init = lambda *a, **k: None
        fake_rclpy.ok = lambda: True
        fake_rclpy.shutdown = lambda: None
        fake_rclpy.create_node = lambda name: types.SimpleNamespace(
            destroy_node=lambda: None)
        fake_rclpy.spin_once = lambda node, timeout_sec=0.0: None
        prev = sys.modules.get('rclpy')
        sys.modules['rclpy'] = fake_rclpy
        args = types.SimpleNamespace(config=self.yaml_path, arm=arm, execute=execute,
                                     blind_join=False, keep_hand=keep_hand, speed=1.0)
        self.last_instances = instances
        try:
            with mock.patch.object(nut_robot, 'RobotClient', FakeClient):
                code = nut_return._run(args)
        finally:
            if prev is None:
                sys.modules.pop('rclpy', None)
            else:
                sys.modules['rclpy'] = prev
        return code, instances

    def test_dry_run_never_moves_or_enables(self):
        starts = {a: self.legs[a].joints[-1] for a in ('left', 'right')}
        code, inst = self._run(starts, execute=False)
        self.assertEqual(code, 0)
        for arm in ('left', 'right'):
            self.assertEqual(inst[arm].moves, [])
            self.assertFalse(inst[arm].enabled)

    def test_execute_reverses_both_arms_to_pt0(self):
        starts = {a: self.legs[a].joints[-1] for a in ('left', 'right')}
        code, inst = self._run(starts, execute=True)
        self.assertEqual(code, 0)
        for arm in ('left', 'right'):
            self.assertTrue(inst[arm].enabled)
            self.assertEqual(inst[arm].hand_opens, 1)
            self.assertEqual(len(inst[arm].moves), 1)          # 2 点 ready 段 -> 1 步
            np.testing.assert_allclose(inst[arm].joints, self.legs[arm].joints[0])

    def test_execute_already_home_does_nothing(self):
        starts = {a: self.legs[a].joints[0] for a in ('left', 'right')}
        code, inst = self._run(starts, execute=True)
        self.assertEqual(code, 0)
        for arm in ('left', 'right'):
            self.assertEqual(inst[arm].moves, [])
            self.assertFalse(inst[arm].enabled)                # 无需运动就不上使能

    def test_off_path_arm_reapproaches_then_reverses(self):
        """不在 ready 轨迹上（抓取中间态）：先安全再接近，再逆向回 pt0。"""
        starts = {a: self.legs[a].joints[-1] for a in ('left', 'right')}
        starts['left'] = np.asarray(starts['left'], float) + 0.5
        code, inst = self._run(starts, execute=True)
        self.assertEqual(code, 0)
        # 左臂：再接近(>=1 步) + 逆向回 pt0
        self.assertGreaterEqual(len(inst['left'].moves), 2)
        np.testing.assert_allclose(inst['left'].joints, self.legs['left'].joints[0],
                                   atol=1e-9)
        self.assertTrue(inst['left'].enabled)
        # 右臂仍在 ready 轨迹上：直接逆向 1 步
        self.assertEqual(len(inst['right'].moves), 1)

    def test_reapproach_both_routes_fail_aborts_before_any_motion(self):
        starts = {a: self.legs[a].joints[-1] for a in ('left', 'right')}
        starts['left'] = np.asarray(starts['left'], float) + 0.5
        with self.assertRaises(TaskError) as ctx:
            self._run(starts, execute=True, ik_fail=True, solve_fail=True)
        self.assertIn('两条路线都不可行', str(ctx.exception))
        for arm in ('left', 'right'):
            self.assertEqual(self.last_instances[arm].moves, [])
            self.assertFalse(self.last_instances[arm].enabled)

    def test_reapproach_fk_gate_aborts_before_any_motion(self):
        starts = {a: self.legs[a].joints[-1] for a in ('left', 'right')}
        starts['left'] = np.asarray(starts['left'], float) + 0.5
        with self.assertRaises(TaskError) as ctx:
            self._run(starts, execute=True, ik_fail=True, fk_low=True)
        self.assertIn('正解校验不通过', str(ctx.exception))
        for arm in ('left', 'right'):
            self.assertEqual(self.last_instances[arm].moves, [])
            self.assertFalse(self.last_instances[arm].enabled)

    def test_off_path_falls_back_to_joint_route_and_reaches_pt0(self):
        """路线A不可行（中转平面逆解不了）时自动改走关节整段，仍回到 pt0。"""
        starts = {a: self.legs[a].joints[-1] for a in ('left', 'right')}
        starts['left'] = np.asarray(starts['left'], float) + 0.5
        code, inst = self._run(starts, execute=True, ik_fail=True)
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(inst['left'].moves), 2)
        np.testing.assert_allclose(inst['left'].joints, self.legs['left'].joints[0],
                                   atol=1e-9)
        self.assertTrue(inst['left'].enabled)

    def test_single_arm_run_ignores_other_arm_position(self):
        starts = {a: self.legs[a].joints[-1] for a in ('left', 'right')}
        starts['left'] = np.asarray(starts['left'], float) + 0.5
        code, inst = self._run(starts, execute=True, arm='right')
        self.assertEqual(code, 0)
        self.assertEqual(len(inst['right'].moves), 1)
        np.testing.assert_allclose(inst['right'].joints, self.legs['right'].joints[0])
        self.assertEqual(inst['left'].moves, [])
        self.assertFalse(inst['left'].enabled)


if __name__ == '__main__':
    sys.exit(unittest.main())
