#!/usr/bin/env python3
"""双臂交接版 nut_pick_place 离线测试（不连机器人/相机；用真实预录 trace 加载段）：

  cd tools
  /usr/bin/python3 -m unittest -v test_nut_task.py
"""
import argparse
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import yaml

from nut_robot import (DEFAULT_CONFIG, PoseStore, SIZE_LABELS, TaskConfig,
                       TaskError)
from nut_detectors import (DepthPixelDetector, Detection, InputDetector,
                           JsonDetector, normalize)
from nut_sequences import Leg, SequenceRunner, load_leg
from nut_pick_place import (cam_to_base, det_base_xyz, detect_until_complete,
                            grasp_points, handoff_check,
                            ik_diagnose, join_gap_rows, load_all_legs, parse_order,
                            resolve_grasp_euler, resume_ready_for_detect,
                            right_ik_seed_bank, validate_detections)
from nut_needle import grip_euler_rad, needle_options, release_euler

WORKSPACE = Path(__file__).resolve().parent.parent
LEFT_TRACE = WORKSPACE / 'recordings/left_trace/events.jsonl'
RIGHT_TRACE = WORKSPACE / 'recordings/right_trace/events.jsonl'

# 段名用现场已录的真实序列；路径写绝对路径，夹具可放任意临时目录
CONFIG_YAML = f"""
namespace: /robot1
arm: right
order: [l, m, s]
require_all: true
check_other_arm: true
left:
  grasp_orientation: left_grasp_init
  trace: {LEFT_TRACE}
  legs:
    - {{sequence: left_ready1_001}}
    - {{sequence: left_place1_002, hand_after: open, retreat: true}}
right:
  trace: {RIGHT_TRACE}
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
hand:
  speed: [80, 80, 80, 80, 80, 80]
  force: [100, 100, 100, 100, 100, 100]
  open: [0, 0, 0, 0, 0, 0]
  sizes:
    l: {{joint: [180, 180, 180, 180, 180, 180]}}
    m: {{joint: [150, 150, 150, 150, 150, 150]}}
    s: {{joint: [120, 120, 120, 120, 120, 120]}}
vision: {{}}
detector: {{type: json, json_path: detections.json}}
"""


def _fake_leg(xyz_end, xyz_start=(0.4, 0.05, -0.36), hand_after='open'):
    """合成只带 cartesian 端点的 Leg，供交接点距离检查用。"""
    def ps(xyz):
        return {'xyz': np.array(xyz, float), 'eul_deg': np.zeros(3)}
    return Leg(arm='left', file=Path('x.jsonl'), target='t', hand_after=hand_after,
               retreat=False, joints=np.zeros((2, 7)), names=[], frame='base_link',
               namespace='/robot1', poses=[ps(xyz_start), ps(xyz_end)])


class ConfigAndLegsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.yaml_path = Path(self.tmp.name) / 'nut_task.yaml'
        self.yaml_path.write_text(CONFIG_YAML, encoding='utf-8')
        self.cfg = TaskConfig(self.yaml_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _reload(self, mutate):
        y = yaml.safe_load(CONFIG_YAML)
        mutate(y)
        p = Path(self.tmp.name) / 'bad.yaml'
        p.write_text(yaml.safe_dump(y, allow_unicode=True), encoding='utf-8')
        return TaskConfig(p)

    def test_real_default_config_loads(self):
        cfg = TaskConfig(DEFAULT_CONFIG)
        self.assertEqual(cfg.namespace, '/robot1')
        self.assertEqual(cfg.order, ('white', 'l', 'm', 's'))
        self.assertEqual(cfg.left_grasp_pose_name, 'left_grasp_init')
        self.assertFalse(cfg.require_all)
        # 白螺母与大黑螺母同形状：段/抓取偏移/手型全部复用 l 档，只有检测标签不同
        self.assertIs(cfg.right_approaches['white'], cfg.right_approaches['l'])
        self.assertIs(cfg.right_place['white'], cfg.right_place['l'])
        np.testing.assert_allclose(cfg.grasp_offset_for('white'), cfg.grasp_offset_for('l'))
        self.assertEqual(cfg.grasp_z_offset_for('white'), cfg.grasp_z_offset_for('l'))
        self.assertEqual(cfg.hover_height_for('white'), cfg.hover_height_for('l'))
        self.assertEqual(cfg.close_for('left', 'white'), cfg.close_for('left', 'l'))
        self.assertEqual(cfg.close_for('right', 'white'), cfg.close_for('right', 'l'))

    def test_leg_specs_parsed(self):
        self.assertEqual(len(self.cfg.left_legs), 2)
        f0, t0, h0, r0 = self.cfg.left_legs[0]
        self.assertEqual((t0, h0, r0), ('left_ready1_001', None, False))
        f1, t1, h1, r1 = self.cfg.left_legs[1]
        self.assertEqual((t1, h1, r1), ('left_place1_002', 'open', True))
        self.assertEqual(f1, LEFT_TRACE)
        self.assertTrue(self.cfg.approach_shared)
        self.assertEqual(self.cfg.right_approach[1:], ('right_ready1_001', 'close', False))
        for k in SIZE_LABELS:
            self.assertIs(self.cfg.right_approaches[k], self.cfg.right_approach)
            self.assertEqual(self.cfg.right_place[k][2], 'open')
            self.assertTrue(self.cfg.right_place[k][3])

    def test_approach_by_size_parses_and_validates(self):
        def mapping(y):
            y['right']['approach'] = {
                'l': {'sequence': 'right_ready1_001', 'hand_after': 'close'},
                'm': {'sequence': 'right_appr_m_002', 'hand_after': 'close'},
                's': {'sequence': 'right_appr_s_003', 'hand_after': 'close'}}
        cfg = self._reload(mapping)
        self.assertFalse(cfg.approach_shared)
        self.assertEqual([cfg.right_approaches[k][1] for k in SIZE_LABELS],
                         ['right_ready1_001', 'right_appr_m_002', 'right_appr_s_003'])
        self.assertIs(cfg.right_approach, cfg.right_approaches['l'])  # 代表段=home

        def missing_s(y):
            y['right']['approach'] = {
                'l': {'sequence': 'right_ready1_001', 'hand_after': 'close'},
                'm': {'sequence': 'right_appr_m_002', 'hand_after': 'close'}}
        with self.assertRaises(TaskError):
            self._reload(missing_s)

        def not_close(y):
            y['right']['approach'] = {
                k: {'sequence': 'right_ready1_001', 'hand_after': 'open'}
                for k in SIZE_LABELS}
        with self.assertRaises(TaskError):
            self._reload(not_close)

        def bad_type(y):
            y['right']['approach'] = ['right_ready1_001']
        with self.assertRaises(TaskError):
            self._reload(bad_type)

    def test_ready_optional_when_unconfigured(self):
        self.assertIsNone(self.cfg.left_ready)
        self.assertIsNone(self.cfg.right_ready)

    def test_load_all_legs_real_traces(self):
        left_legs, apprs, places, release, ready_left, ready_right = load_all_legs(self.cfg)
        self.assertIsNone(ready_left)
        self.assertIsNone(ready_right)
        self.assertEqual([l.target for l in left_legs],
                         ['left_ready1_001', 'left_place1_002'])
        self.assertIs(release, left_legs[1])
        self.assertEqual(apprs['l'].target, 'right_ready1_001')
        self.assertEqual(apprs['l'].hand_after, 'close')
        self.assertEqual([places[k].target for k in SIZE_LABELS],
                         ['right_place1_002', 'right_place2_003', 'right_place3_004'])
        for leg in left_legs + list(apprs.values()) + list(places.values()):
            self.assertEqual(leg.joints.shape[1], 7)
            self.assertGreaterEqual(len(leg.joints), 2)
            self.assertEqual(leg.namespace, '/robot1')
            self.assertEqual(leg.frame, 'base_link')
            self.assertEqual(len(leg.names), 7)

    def test_left_close_leg_rejected(self):
        def m(y):
            y['left']['legs'][0]['hand_after'] = 'close'
        with self.assertRaises(TaskError):
            self._reload(m)

    def test_left_without_open_rejected(self):
        def m(y):
            y['left']['legs'][1].pop('hand_after')
        with self.assertRaises(TaskError):
            self._reload(m)

    def test_approach_must_close(self):
        def m(y):
            y['right']['approach'].pop('hand_after')
        with self.assertRaises(TaskError):
            self._reload(m)

    def test_place_must_open(self):
        def m(y):
            y['right']['place']['l'].pop('hand_after')
        with self.assertRaises(TaskError):
            self._reload(m)

    def test_missing_place_size_rejected(self):
        def m(y):
            del y['right']['place']['s']
        with self.assertRaises(TaskError):
            self._reload(m)

    def test_speed_limits(self):
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(join_speed=0.4))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(sequence_speed=0.6))

    def test_bad_hand_values(self):
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['hand'].update(force=[100] * 5))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['hand']['sizes']['l'].update(joint=[300] * 6))

    def test_order_override_and_validation(self):
        cfg = TaskConfig(self.yaml_path, order_override=('s', 'm', 'l'))
        self.assertEqual(cfg.order, ('s', 'm', 'l'))
        with self.assertRaises(TaskError):
            TaskConfig(self.yaml_path, order_override=('l', 'l', 's'))
        with self.assertRaises(TaskError):
            TaskConfig(self.yaml_path, order_override=('l', 'x'))

    def test_real_config_arm_close_defaults(self):
        cfg = TaskConfig(DEFAULT_CONFIG)
        for k in SIZE_LABELS:
            # 左臂拇指弯曲 20（2026-09-16 现场：抓取前拇指多弯 20）；
            # s 另走 hand.sizes.s 覆盖（小螺母专用手型：四指弯曲 40）
            expect = [20, 0, 40, 40, 40, 40] if k == 's' else [20, 0, 0, 0, 0, 0]
            self.assertEqual(cfg.close_for('left', k), expect)
            # 右臂未改，仍是六路全闭
            self.assertEqual(cfg.close_for('right', k), [0, 0, 0, 0, 0, 0])

    def test_real_config_open_and_pacing(self):
        cfg = TaskConfig(DEFAULT_CONFIG)
        self.assertEqual(cfg.hand_open_vals, [185, 80, 255, 255, 255, 255])
        for attr in ('ready_hold_seconds', 'point_dwell_seconds',
                     'between_leg_seconds', 'pre_hand_seconds', 'hover_dwell_seconds'):
            self.assertGreater(getattr(cfg, attr), 0)

    def test_grasp_orientation_recording_form_rejected_when_bad(self):
        def m1(y):
            y['left']['grasp_orientation'] = {'point': 0}  # 缺 sequence
        with self.assertRaises(TaskError):
            self._reload(m1)
        def m2(y):
            y['left']['grasp_orientation'] = {'sequence': 'left_ready1_001', 'point': -1}
        with self.assertRaises(TaskError):
            self._reload(m2)

    def test_speed_scale(self):
        base = TaskConfig(self.yaml_path)
        fast = TaskConfig(self.yaml_path, speed_scale=2.0)
        for attr in ('speed', 'acce', 'linear_speed', 'linear_acce',
                     'join_speed', 'join_acce', 'sequence_speed', 'sequence_acce'):
            self.assertAlmostEqual(getattr(fast, attr), 2.0 * getattr(base, attr))
        slow = TaskConfig(self.yaml_path, speed_scale=0.5)
        self.assertAlmostEqual(slow.sequence_speed, 0.15)
        # 非运动参数不缩放
        self.assertEqual(fast.hover_height, base.hover_height)
        self.assertEqual(fast.reached_tolerance, base.reached_tolerance)
        with self.assertRaises(TaskError):
            TaskConfig(self.yaml_path, speed_scale=0)
        with self.assertRaises(TaskError):
            TaskConfig(self.yaml_path, speed_scale=-1)

    def test_reached_wait_default_and_validation(self):
        # 缺省 3.0s；真实 yaml 显式 3.0；<0.5 拒绝
        self.assertEqual(TaskConfig(self.yaml_path).reached_wait_seconds, 3.0)
        self.assertEqual(TaskConfig(DEFAULT_CONFIG).reached_wait_seconds, 3.0)
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(reached_wait_seconds=0.4))

    def test_per_arm_reached_tolerance(self):
        # 夹具/缺省无分臂覆盖；真实 yaml 右臂 0.05、左臂 0.1
        bare = TaskConfig(self.yaml_path)
        self.assertIsNone(bare.reached_tolerance_left)
        self.assertIsNone(bare.reached_tolerance_right)
        real = TaskConfig(DEFAULT_CONFIG)
        self.assertAlmostEqual(real.reached_tolerance_left, 0.1)
        self.assertAlmostEqual(real.reached_tolerance_right, 0.05)
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(reached_tolerance_right=0.11))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(reached_tolerance_left=0.0))
        set_left = self._reload(lambda y: y['motion'].update(reached_tolerance_left=0.04))
        self.assertAlmostEqual(set_left.reached_tolerance_left, 0.04)

    def test_robot_pose_errors_math(self):
        # 真实 RobotClient.pose_errors：位置欧氏误差 + 姿态四元数夹角
        from scipy.spatial.transform import Rotation as Rot
        from nut_robot import RobotClient
        rc = RobotClient.__new__(RobotClient)
        eul = np.array([1.1, -0.1, -1.55])
        q = Rot.from_euler('xyz', eul).as_quat()
        rc.pose = (np.array([0.4, 0.1, -0.35]), q)
        pe, oe, px = rc.pose_errors([0.4, 0.1, -0.37], eul)
        self.assertAlmostEqual(pe, 0.02, places=6)
        self.assertAlmostEqual(oe, 0.0, places=6)
        np.testing.assert_allclose(px, [0.4, 0.1, -0.35])
        rc.pose = (np.array([0.4, 0.1, -0.35]),
                   Rot.from_euler('xyz', eul + [0.0, 0.03, 0.0]).as_quat())
        _, oe2, _ = rc.pose_errors([0.4, 0.1, -0.35], eul)
        self.assertAlmostEqual(oe2, 0.03, places=5)
        rc.pose = None
        self.assertIsNone(rc.pose_errors([0.4, 0.1, -0.35], eul))

    def test_pose_tolerance_config(self):
        cfg = TaskConfig(self.yaml_path)
        self.assertAlmostEqual(cfg.pose_pos_tolerance, 0.01)
        self.assertAlmostEqual(cfg.pose_ori_tolerance, 0.05)
        self.assertAlmostEqual(TaskConfig(DEFAULT_CONFIG).pose_pos_tolerance, 0.01)
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(pose_pos_tolerance=0.06))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(pose_ori_tolerance=0.0))

    def test_duplicate_policy_config(self):
        # 缺省 abort（安全）；真实 yaml 已设 random
        self.assertEqual(TaskConfig(self.yaml_path).duplicate_policy, 'abort')
        self.assertEqual(TaskConfig(DEFAULT_CONFIG).duplicate_policy, 'random')
        cfg = self._reload(lambda y: y['detector'].update(duplicate_policy='first'))
        self.assertEqual(cfg.duplicate_policy, 'first')
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['detector'].update(duplicate_policy='nearest'))

    def test_grasp_offset_default_and_validation(self):
        # 缺省向机体方向（base_link -X）退 15cm，补偿腕-指尖前后偏差
        np.testing.assert_allclose(TaskConfig(self.yaml_path).grasp_offset_xyz,
                                   [-0.15, 0, 0])
        np.testing.assert_allclose(TaskConfig(DEFAULT_CONFIG).grasp_offset_xyz,
                                   [-0.08, 0.02, 0])
        cfg = self._reload(lambda y: y['motion'].update(
            grasp_offset_xyz=[0, -0.02, 0.01]))
        np.testing.assert_allclose(cfg.grasp_offset_xyz, [0, -0.02, 0.01])
        zero = self._reload(lambda y: y['motion'].update(grasp_offset_xyz=[0, 0, 0]))
        np.testing.assert_allclose(zero.grasp_offset_xyz, [0, 0, 0])
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(grasp_offset_xyz=[-0.15, 0]))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(grasp_offset_xyz=[-150, 0, 0]))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(grasp_offset_xyz='x'))

    def test_reissue_count_default_and_validation(self):
        # 缺省补发 2 次（含首下共 3 次下发）；范围 0~5
        self.assertEqual(TaskConfig(self.yaml_path).reached_reissue_count, 2)
        self.assertEqual(TaskConfig(DEFAULT_CONFIG).reached_reissue_count, 2)
        cfg0 = self._reload(lambda y: y['motion'].update(reached_reissue_count=0))
        self.assertEqual(cfg0.reached_reissue_count, 0)
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(reached_reissue_count=6))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(reached_reissue_count=-1))

    def test_negative_pacing_rejected(self):
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(between_leg_seconds=-1))

    def test_real_config_ready_specs(self):
        cfg = TaskConfig(DEFAULT_CONFIG)
        self.assertIsNotNone(cfg.left_ready)
        self.assertIsNotNone(cfg.right_ready)
        self.assertEqual(cfg.left_ready[0].parent.name, '20260915_175547_556799')
        self.assertEqual(cfg.left_ready[1], 'left-ready_001')
        self.assertIsNone(cfg.left_ready[2])
        self.assertEqual(cfg.right_ready[0].parent.name, '20260915_134046_684005')
        self.assertEqual(cfg.right_ready[1], 'right-ready_001')
        _, _, _, _, rl, rr = load_all_legs(cfg)
        self.assertEqual(rl.target, 'left-ready_001')
        self.assertEqual(rr.target, 'right-ready_001')

    def test_ready_hand_action_rejected(self):
        def m(y):
            y['left']['ready'] = {'sequence': 'left_ready1_001', 'hand_after': 'open'}
        with self.assertRaises(TaskError):
            self._reload(m)
        def m2(y):
            y['right']['ready'] = {'sequence': 'right_ready1_001', 'retreat': True}
        with self.assertRaises(TaskError):
            self._reload(m2)

    def test_close_resolution_precedence(self):
        # 夹具配置只有 sizes.*.joint：双臂都取 joint
        self.assertEqual(self.cfg.close_for('left', 'l'), [180] * 6)
        self.assertEqual(self.cfg.close_for('right', 's'), [120] * 6)

        # 臂默认 + 尺寸级 joint 同时存在：joint 优先
        def m1(y):
            y['hand']['close'] = {'left': [1, 1, 1, 1, 1, 1],
                                  'right': [2, 2, 2, 2, 2, 2]}
        cfg = self._reload(m1)
        self.assertEqual(cfg.close_for('left', 'l'), [180] * 6)
        self.assertEqual(cfg.close_for('right', 's'), [120] * 6)

        # 去掉尺寸配置：回落到臂默认，缺该臂则报错
        def m2b(y):
            y['hand'].pop('sizes')
            y['hand']['close'] = {'left': [1, 2, 3, 4, 5, 6],
                                  'right': [6, 5, 4, 3, 2, 1]}
        cfg_b = self._reload(m2b)
        self.assertEqual(cfg_b.close_for('left', 'm'), [1, 2, 3, 4, 5, 6])
        self.assertEqual(cfg_b.close_for('right', 'm'), [6, 5, 4, 3, 2, 1])

        def m2(y):
            y['hand'].pop('sizes')
            y['hand']['close'] = {'left': [1, 2, 3, 4, 5, 6]}
        cfg2 = self._reload(m2)
        self.assertEqual(cfg2.close_for('left', 'l'), [1, 2, 3, 4, 5, 6])
        with self.assertRaises(TaskError):
            cfg2.close_for('right', 'l')

        def m3(y):
            y['hand']['sizes']['l'] = {'left': [9, 9, 0, 0, 0, 200],
                                       'right': [9, 9, 0, 0, 0, 210]}
        cfg3 = self._reload(m3)
        self.assertEqual(cfg3.close_for('left', 'l'), [9, 9, 0, 0, 0, 200])
        self.assertEqual(cfg3.close_for('right', 'l'), [9, 9, 0, 0, 0, 210])

    def test_missing_sequence_file_rejected(self):
        def m(y):
            y['left']['trace'] = '/no/such/events.jsonl'
        with self.assertRaises(TaskError):
            load_all_legs(self._reload(m))

    def test_grasp_by_size_overrides_and_fallback(self):
        def m(y):
            y['motion']['grasp_by_size'] = {
                'm': {'offset_xyz': [-0.05, 0.01, 0.0], 'z_offset': 0.02},
                's': {'hover_height': 0.08}}
        cfg = self._reload(m)
        # l 未配 -> 全部回退全局（CONFIG 里未写 offset，用缺省 -0.15）
        np.testing.assert_allclose(cfg.grasp_offset_for('l'), cfg.grasp_offset_xyz)
        self.assertEqual(cfg.grasp_z_offset_for('l'), cfg.grasp_z_offset)
        self.assertEqual(cfg.hover_height_for('l'), cfg.hover_height)
        # m 覆盖 offset/z，hover 未覆盖回退全局
        np.testing.assert_allclose(cfg.grasp_offset_for('m'), [-0.05, 0.01, 0.0])
        self.assertEqual(cfg.grasp_z_offset_for('m'), 0.02)
        self.assertEqual(cfg.hover_height_for('m'), 0.10)
        # s 只覆盖 hover
        np.testing.assert_allclose(cfg.grasp_offset_for('s'), cfg.grasp_offset_xyz)
        self.assertEqual(cfg.grasp_z_offset_for('s'), 0.0)
        self.assertEqual(cfg.hover_height_for('s'), 0.08)
        # lift_height 缺省 0.05，按尺寸覆盖优先
        self.assertEqual(cfg.lift_height, 0.05)
        self.assertEqual(cfg.lift_height_for('l'), 0.05)
        cfg_l = self._reload(lambda y: y['motion'].update(
            lift_height=0.08,
            grasp_by_size={'m': {'lift_height': 0.03}}))
        self.assertEqual(cfg_l.lift_height_for('m'), 0.03)
        self.assertEqual(cfg_l.lift_height_for('s'), 0.08)

    def test_hover_must_be_above_down(self):
        # 全局反转：hover_height <= grasp_z_offset -> 下杵桌面，加载即拒绝
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(hover_height=0.10,
                                                      grasp_z_offset=0.10))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(hover_height=0.05,
                                                      grasp_z_offset=0.10))
        # 按尺寸反转同样拒绝（其余尺寸缺省合法）
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['motion'].update(grasp_by_size={
                'l': {'z_offset': 0.12, 'hover_height': 0.10}}))
        # hover 严格高于 z_offset 即合法（含按尺寸）
        ok = self._reload(lambda y: y['motion'].update(grasp_by_size={
            'l': {'z_offset': 0.12, 'hover_height': 0.13}}))
        self.assertEqual(ok.hover_height_for('l'), 0.13)

    def test_lift_height_bad_values_rejected(self):
        for bad in (-0.01, 0.31, 'x', float('nan')):
            with self.subTest(bad=bad):
                with self.assertRaises(TaskError):
                    self._reload(lambda y, v=bad: y['motion'].update(lift_height=v))
        with self.assertRaises(TaskError):  # 按尺寸超范围
            self._reload(lambda y: y['motion'].update(
                grasp_by_size={'s': {'lift_height': 0.4}}))
        ok = self._reload(lambda y: y['motion'].update(lift_height=0.0))
        self.assertEqual(ok.lift_height_for('l'), 0.0)

    def test_grasp_by_size_bad_values_rejected(self):
        with self.assertRaises(TaskError):  # 非法尺寸键
            self._reload(lambda y: y['motion'].update(grasp_by_size={'x': {}}))
        with self.assertRaises(TaskError):  # 4 个数
            self._reload(lambda y: y['motion'].update(
                grasp_by_size={'l': {'offset_xyz': [1, 2, 3, 4]}}))
        with self.assertRaises(TaskError):  # 模长 >0.3m（毫米单位笔误）
            self._reload(lambda y: y['motion'].update(
                grasp_by_size={'l': {'offset_xyz': [0.5, 0, 0]}}))
        with self.assertRaises(TaskError):  # 非映射
            self._reload(lambda y: y['motion'].update(grasp_by_size={'l': [0, 0, 0]}))
        with self.assertRaises(TaskError):  # z 非数
            self._reload(lambda y: y['motion'].update(
                grasp_by_size={'l': {'z_offset': 'low'}}))

    def test_grasp_orientation_by_size_parses(self):
        def m(y):
            y['left']['grasp_orientation_by_size'] = {
                'm': 'grasp_m_pose',
                'l': {'sequence': 'left_ready1_001', 'point': 0}}
        cfg = self._reload(m)
        self.assertEqual(cfg.grasp_orientation_for('m'), ('grasp_m_pose', None))
        self.assertEqual(cfg.grasp_orientation_for('l')[1]['sequence'], 'left_ready1_001')
        # s 未覆盖 -> 全局 left_grasp_init
        self.assertEqual(cfg.grasp_orientation_for('s'), ('left_grasp_init', None))
        with self.assertRaises(TaskError):  # 非法尺寸键
            self._reload(lambda y: y['left'].update(
                grasp_orientation_by_size={'x': 'p'}))
        with self.assertRaises(TaskError):  # 值不是名字/映射
            self._reload(lambda y: y['left'].update(
                grasp_orientation_by_size={'l': 123}))

    def test_detect_attempts_config_validation(self):
        for bad in (0, 11, 'x', [3], 2.5):
            with self.subTest(bad=bad):
                with self.assertRaises(TaskError):
                    self._reload(lambda y, v=bad: y['detector'].update(detect_attempts=v))
        with self.assertRaises(TaskError):
            self._reload(lambda y: y['detector'].update(missing_retry_seconds=-0.1))
        ok = self._reload(lambda y: y['detector'].update(detect_attempts=5,
                                                         missing_retry_seconds=0.0))
        self.assertEqual(ok.detect_attempts, 5)
        self.assertEqual(ok.missing_retry_seconds, 0.0)

    def test_show_window_config_validation(self):
        self.assertFalse(self.cfg.show_window)
        self.assertEqual(self.cfg.show_seconds, 2.0)
        for bad in ('yes', 1, 0):
            with self.subTest(bad=bad):
                with self.assertRaises(TaskError):
                    self._reload(lambda y, v=bad: y['detector'].update(show_window=v))
        for bad in (-0.1, 31, 'x'):
            with self.subTest(bad=bad):
                with self.assertRaises(TaskError):
                    self._reload(lambda y, v=bad: y['detector'].update(show_seconds=v))
        ok = self._reload(lambda y: y['detector'].update(show_window=True,
                                                         show_seconds=0.0))
        self.assertTrue(ok.show_window)
        self.assertEqual(ok.show_seconds, 0.0)
        self.assertTrue(ok.detector_raw['show_window'])

    def test_redetect_each_nut_config_validation(self):
        self.assertTrue(self.cfg.redetect_each_nut)  # 缺省即开启逐颗重识别
        for bad in ('yes', 1, 0):
            with self.subTest(bad=bad):
                with self.assertRaises(TaskError):
                    self._reload(
                        lambda y, v=bad: y['detector'].update(redetect_each_nut=v))
        off = self._reload(lambda y: y['detector'].update(redetect_each_nut=False))
        self.assertFalse(off.redetect_each_nut)
        on = self._reload(lambda y: y['detector'].update(redetect_each_nut=True))
        self.assertTrue(on.redetect_each_nut)

    def test_missing_nut_retries_then_succeeds(self):
        d_l = Detection('l', np.zeros(3)); d_m = Detection('m', np.zeros(3))
        d_s = Detection('s', np.zeros(3))
        rounds = [[d_l, d_s], [d_l, d_m, d_s]]  # 首轮缺中螺母，第二轮齐全
        calls = []
        with mock.patch('time.sleep') as slept:
            dets, chosen = detect_until_complete(
                self.cfg, lambda: (calls.append(1) or rounds[len(calls) - 1]))
        self.assertEqual(len(calls), 2)
        self.assertIs(chosen['m'], d_m)
        self.assertEqual(set(chosen), {'l', 'm', 's'})
        slept.assert_called_once_with(1.0)

    def test_missing_nut_gives_up_after_attempt_cap(self):
        calls = []
        with mock.patch('time.sleep'):
            with self.assertRaises(TaskError):
                detect_until_complete(
                    self.cfg, lambda: (calls.append(1) or [Detection('l', np.zeros(3))]))
        self.assertEqual(len(calls), 3)  # 默认 3 轮
        cfg1 = self._reload(lambda y: y['detector'].update(detect_attempts=1))
        calls = []
        with mock.patch('time.sleep'):
            with self.assertRaises(TaskError):
                detect_until_complete(
                    cfg1, lambda: (calls.append(1) or [Detection('l', np.zeros(3))]))
        self.assertEqual(len(calls), 1)

    def test_require_all_false_skips_retry_when_missing(self):
        cfg = self._reload(lambda y: y.update(require_all=False))
        calls = []
        only_l = [Detection('l', np.zeros(3))]
        with mock.patch('time.sleep') as slept:
            dets, chosen = detect_until_complete(
                cfg, lambda: (calls.append(1) or only_l))
        self.assertEqual(len(calls), 1)       # 缺料允许跳过 -> 不重试
        slept.assert_not_called()
        self.assertEqual(set(chosen), {'l'})

    def test_resolve_grasp_euler_per_size(self):
        # 同一 PoseStore 里放全局姿态与 m 覆盖姿态
        store = PoseStore(self.cfg.poses_path)
        store.put('left', 'left_grasp_init', [0.3, 0.3, -0.3], [40.0, -6.67, -95.0], None)
        store.put('left', 'grasp_m_pose', [0.3, 0.3, -0.3], [10.0, 20.0, 30.0], None)
        store.save()
        cfg = self._reload(lambda y: y['left'].update(
            grasp_orientation_by_size={'m': 'grasp_m_pose'}))
        eul_m, src_m = resolve_grasp_euler(cfg, store, 'm')
        eul_s, src_s = resolve_grasp_euler(cfg, store, 's')
        np.testing.assert_allclose(np.degrees(eul_m), [10, 20, 30])
        self.assertIn('grasp_m_pose', src_m)
        np.testing.assert_allclose(np.degrees(eul_s), [40.0, -6.67, -95.0])
        self.assertIn('left_grasp_init', src_s)
        # 缺位姿名 -> TaskError（由 resolve_grasp_eulers 转成 (None, 原因)）
        cfg_bad = self._reload(lambda y: y['left'].update(
            grasp_orientation_by_size={'l': 'no_such_pose'}))
        with self.assertRaises(TaskError):
            resolve_grasp_euler(cfg_bad, store, 'l')
        from nut_pick_place import resolve_grasp_eulers
        poses = resolve_grasp_eulers(cfg_bad, store)
        self.assertIsNone(poses['l'][0])
        self.assertIn('no_such_pose', poses['l'][1])
        self.assertIsNotNone(poses['m'][0])


LEFT_GRASP_DIR = WORKSPACE / 'recordings/left_grasp_middle'
LEFT_BACK_DIR = WORKSPACE / 'recordings/left_middle_back'
RIGHT_GRASP_DIR = WORKSPACE / 'recordings/right_grasp_middle'
RIGHT_BACK_DIR = WORKSPACE / 'recordings/right_middle_back'

SHARED_CONFIG_YAML = f"""
namespace: /robot1
arm: right
order: [l, m, s]
require_all: true
left:
  grasp_orientation: left_grasp_init
  trace: {LEFT_GRASP_DIR}/events.jsonl
  ready: {{file: {LEFT_TRACE}, sequence: left_ready1_001}}
  legs:
    - {{sequence: left_grasp_place_middle_001, hand_after: open}}
    - {{file: {LEFT_BACK_DIR}/events.jsonl, sequence: left_middle_back_001}}
right:
  trace: {RIGHT_GRASP_DIR}/events.jsonl
  ready: {{file: {RIGHT_TRACE}, sequence: right_ready1_001}}
  approach: {{sequence: right_grasp_middle_001, hand_after: close}}
  place:
    file: {RIGHT_BACK_DIR}/events.jsonl
    sequence: right_middle_back_001
    hand_after: open
motion: {{join_speed: 0.15, join_acce: 0.15, sequence_speed: 0.3, sequence_acce: 0.5}}
hand:
  speed: [80, 80, 80, 80, 80, 80]
  force: [100, 100, 100, 100, 100, 100]
  open: [0, 0, 0, 0, 0, 0]
  sizes:
    l: {{joint: [180, 180, 180, 180, 180, 180]}}
    m: {{joint: [150, 150, 150, 150, 150, 150]}}
    s: {{joint: [120, 120, 120, 120, 120, 120]}}
vision: {{}}
detector: {{type: manual}}
"""

@unittest.skipUnless(all(p.exists() for p in
                         (LEFT_GRASP_DIR, LEFT_BACK_DIR, RIGHT_GRASP_DIR, RIGHT_BACK_DIR,
                          LEFT_TRACE.parent, RIGHT_TRACE.parent)),
                     '2026-09-10 正式录制文件夹或 left/right_trace 不在')
class SharedPlaceRealRecordingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.yaml_path = Path(self.tmp.name) / 'nut_task_shared.yaml'
        self.yaml_path.write_text(SHARED_CONFIG_YAML, encoding='utf-8')
        self.cfg = TaskConfig(self.yaml_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_place_config(self):
        self.assertTrue(self.cfg.place_shared)
        specs = {self.cfg.right_place[k] for k in SIZE_LABELS}
        self.assertEqual(len(specs), 1)
        self.assertEqual(self.cfg.right_place['l'][1], 'right_middle_back_001')
        self.assertEqual(self.cfg.left_legs[0][1], 'left_grasp_place_middle_001')
        self.assertEqual(self.cfg.left_legs[0][2], 'open')
        self.assertEqual(self.cfg.left_legs[1][1], 'left_middle_back_001')
        self.assertIsNone(self.cfg.left_legs[1][2])
        self.assertTrue(self.cfg.approach_shared)
        self.assertEqual(self.cfg.right_approach[1:3],
                         ('right_grasp_middle_001', 'close'))
        self.assertIs(self.cfg.right_approaches['m'], self.cfg.right_approach)

    def test_load_shared_legs(self):
        left_legs, apprs, places, release, ready_left, ready_right = load_all_legs(self.cfg)
        self.assertEqual(len(left_legs), 2)
        self.assertIs(release, left_legs[0])
        self.assertEqual([l.target for l in left_legs],
                         ['left_grasp_place_middle_001', 'left_middle_back_001'])
        self.assertIs(apprs['l'], apprs['m'])
        self.assertIs(apprs['m'], apprs['s'])
        self.assertEqual(apprs['l'].target, 'right_grasp_middle_001')
        self.assertIs(places['l'], places['m'])
        self.assertIs(places['m'], places['s'])
        self.assertEqual(places['l'].target, 'right_middle_back_001')
        self.assertEqual(ready_left.target, 'left_ready1_001')
        self.assertEqual(ready_right.target, 'right_ready1_001')

    def test_load_per_size_approaches(self):
        def m(y):
            spec = {'sequence': 'right_grasp_middle_001', 'hand_after': 'close'}
            y['right']['approach'] = {'l': dict(spec), 'm': dict(spec), 's': dict(spec)}
        cfg = self._reload_shared(m)
        self.assertFalse(cfg.approach_shared)
        left_legs, apprs, places, release, _, _ = load_all_legs(cfg)
        for k in SIZE_LABELS:
            self.assertEqual(apprs[k].target, 'right_grasp_middle_001')
            self.assertEqual(apprs[k].hand_after, 'close')
        rows = handoff_check(release, apprs, False)
        self.assertEqual(len(rows), 3)  # 每尺寸各一行
        self.assertTrue(all(f'approach_{k} 末点' in r for k, r in zip(SIZE_LABELS, rows)))
        gap = join_gap_rows(left_legs, apprs, False, places, True)  # 无 ready
        # 三段是同一序列复制时，首点一致，不应有首点不一致警告
        self.assertFalse(any('首点不一致' in r for r in gap))

    def test_real_handoff_gap_reported(self):
        left_legs, apprs, places, release, ready_left, ready_right = load_all_legs(self.cfg)
        row = handoff_check(release, apprs, True)[0]
        # right_grasp_middle 末点 2026-09-10 抬高 3cm（-337.7→-307.7mm）后，交接差 69→80mm
        self.assertIn('80mm', row)
        self.assertIn('⚠', row)
        rows = join_gap_rows(left_legs, apprs, True, places, True,
                             ready_left, ready_right)
        # 开机 ready 两行 + 左段间接入/左臂回环/右臂抓后抬离/右臂回环 = 6 行
        self.assertEqual(len(rows), 6)
        self.assertTrue(any('place段首' in r for r in rows))
        ready_rows = [r for r in rows if 'ready' in r]
        self.assertEqual(len(ready_rows), 2)
        self.assertTrue(any('left_ready1_001' in r for r in ready_rows))
        self.assertTrue(any('right_ready1_001' in r for r in ready_rows))

    def test_join_rows_without_ready_stays_four(self):
        # 不配 ready 段时不增加开机行（旧行为）
        left_legs, apprs, places, _, _, _ = load_all_legs(self.cfg)
        rows = join_gap_rows(left_legs, apprs, True, places, True)
        self.assertEqual(len(rows), 4)

    def _reload_shared(self, mutate):
        y = yaml.safe_load(SHARED_CONFIG_YAML)
        mutate(y)
        p = Path(self.tmp.name) / 'bad_shared.yaml'
        p.write_text(yaml.safe_dump(y, allow_unicode=True), encoding='utf-8')
        return TaskConfig(p)

    def test_grasp_euler_from_recording_point(self):
        # 姿态源指向正式 left_grasp_middle 段的点：欧拉角应与该记录点一致
        cfg = TaskConfig(DEFAULT_CONFIG)
        rec = cfg.grasp_orientation_rec
        self.assertIsNone(rec)  # 默认配置仍走位姿库字符串
        left_legs, _, _, _, _, _ = load_all_legs(cfg)
        eul, src = resolve_grasp_euler(cfg, PoseStore(cfg.poses_path))
        self.assertIn('left_grasp_init', src)
        np.testing.assert_allclose(np.degrees(eul), [40.0, -6.67, -95.0], atol=0.02)

        def m(y):
            y['left']['grasp_orientation'] = {
                'file': str(LEFT_GRASP_DIR / 'events.jsonl'),
                'sequence': 'left_grasp_place_middle_001', 'point': 0}
        cfg2 = self._reload_shared(m)
        eul2, src2 = resolve_grasp_euler(cfg2, PoseStore(cfg2.poses_path))
        self.assertIn('left_grasp_place_middle_001 pt0', src2)
        # 注意正式 yaml 的首段已换成 left_grasp_middle1/left_middle_grasp_001，
        # 这里姿态源显式指向旧录段，需单独加载旧录段取期望值（不能用 left_legs[0]）
        old_leg = load_leg('left', (LEFT_GRASP_DIR / 'events.jsonl',
                                    'left_grasp_place_middle_001', None, False))
        np.testing.assert_allclose(eul2,
                                   np.radians(old_leg.poses[0]['eul_deg']), atol=1e-9)

        def m_bad(y):
            y['left']['grasp_orientation'] = {
                'file': str(LEFT_GRASP_DIR / 'events.jsonl'),
                'sequence': 'left_grasp_place_middle_001', 'point': 9}
        with self.assertRaises(TaskError):
            resolve_grasp_euler(self._reload_shared(m_bad), PoseStore(cfg.poses_path))

    def test_ik_check_multiple_seeds(self):
        # 驱动数值逆解依赖种子：当前关节角失败、记录段臂型成功时应判可达
        from nut_robot import RobotClient

        class _FakeReq:
            def __init__(self):
                self.position = type('P', (), {'x': 0, 'y': 0, 'z': 0})()
                self.euler = type('E', (), {'x': 0, 'y': 0, 'z': 0})()
                self.joints = None

        class _FakeCli:
            srv_type = type('T', (), {'Request': _FakeReq})

        rc = object.__new__(RobotClient)
        rc.ik_cli = _FakeCli()
        calls = []

        def fake_spin(cli, req, timeout):
            calls.append(list(req.joints))
            ok = len(calls) > 1  # 第一种子(当前角)失败，其余通过
            return type('R', (), {'success': ok,
                                  'joints': [0.1] * 7 if ok else []})()
        rc._spin_call = fake_spin
        rc.joints = [0.0] * 7
        ok, used = rc.ik_check([0.3, 0.3, -0.3], [0.9, 0, -1.5],
                               extra_seeds=[[1.0] * 7])
        self.assertTrue(ok)
        self.assertEqual(used, 1)
        self.assertEqual(len(calls), 2)
        # 回归保护：ik 系列绝不能发空 joints —— 驱动在 request->joints.empty()
        # 时把 nullptr 交给 SDK，lbot_create_ik_request+0x50 直接段错误，
        # 整个驱动进程死掉（2026-09-16 定位）。
        self.assertTrue(all(len(c) == 7 for c in calls), f'出现空种子调用：{calls}')

        def all_fail(cli, req, timeout):
            calls.append(list(req.joints))
            return type('R', (), {'success': False, 'joints': []})()
        rc._spin_call = all_fail
        ok, used = rc.ik_check([0.3, 0.3, -0.3], [0.9, 0, -1.5],
                               extra_seeds=[[1.0] * 7, [2.0] * 7])
        self.assertFalse(ok)
        self.assertIsNone(used)


class PureLogicTest(unittest.TestCase):
    def test_parse_order(self):
        self.assertEqual(parse_order('sml'), ['s', 'm', 'l'])
        self.assertEqual(parse_order('s,m,l'), ['s', 'm', 'l'])
        self.assertEqual(parse_order('m l s'), ['m', 'l', 's'])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_order('abc')

    def test_cam_transform(self):
        R = np.diag([1., -1., 1.])
        t = np.array([1., 2., 3.])
        np.testing.assert_allclose(cam_to_base([0, 0, 0], R, t), t)
        np.testing.assert_allclose(cam_to_base([0, 1, 0], R, t), [1, 1, 3])

    def test_base_frame_detection_skips_extrinsics(self):
        R = np.diag([1., -1., 1.])
        t = np.array([1., 2., 3.])
        det = Detection('l', np.array([0.378, 0.326, -0.348]),
                        extra={'frame': 'base_link'})
        np.testing.assert_allclose(det_base_xyz(det, R, t), [0.378, 0.326, -0.348])
        det_cam = Detection('l', np.zeros(3))
        np.testing.assert_allclose(det_base_xyz(det_cam, R, t), t)

    def test_grasp_points_apply_base_offset_after_transform(self):
        class _GraspCfg:
            """只实现 grasp_points 需要的按尺寸访问器（生产中是 TaskConfig）。"""
            def __init__(self, off, hover, z, by_size=None, lift=0.05):
                self.off, self.hover, self.z, self.lift = \
                    np.array(off, float), hover, z, lift
                self.by_size = by_size or {}

            def grasp_offset_for(self, label):
                return self.by_size.get(label, {}).get('offset_xyz', self.off)

            def hover_height_for(self, label):
                return self.by_size.get(label, {}).get('hover_height', self.hover)

            def grasp_z_offset_for(self, label):
                return self.by_size.get(label, {}).get('z_offset', self.z)

            def lift_height_for(self, label):
                return self.by_size.get(label, {}).get('lift_height', self.lift)

        cfg = _GraspCfg([-0.15, 0.0, 0.0], 0.10, 0.0)
        R = np.diag([1., -1., 1.])
        t = np.array([1., 2., 3.])
        # 相机系结果：先外参变换到 base，再在 base 系向机体方向退 15cm
        det_cam = Detection('l', np.array([0., 1., 0.]))
        pb_raw, pb, hover, down, lift = grasp_points(cfg, det_cam, R, t)
        np.testing.assert_allclose(pb_raw, [1, 1, 3])
        np.testing.assert_allclose(pb, [0.85, 1, 3])
        np.testing.assert_allclose(hover, [0.85, 1, 3.1])
        np.testing.assert_allclose(down, [0.85, 1, 3])
        np.testing.assert_allclose(lift, [0.85, 1, 3.05])  # 抓后从 down 直上 5cm
        # base 直给（input/json frame=base_link）同样施加偏移，不走外参
        det_base = Detection('m', np.array([0.378, 0.326, -0.348]),
                             extra={'frame': 'base_link'})
        pb_raw, pb, hover, down, lift = grasp_points(cfg, det_base, R, t)
        np.testing.assert_allclose(pb_raw, [0.378, 0.326, -0.348])
        np.testing.assert_allclose(pb, [0.228, 0.326, -0.348])
        # 零偏移 + z 微调时退化为原来的 hover/down 公式；lift 永远从 down 起算
        cfg0 = _GraspCfg(np.zeros(3), 0.10, -0.01)
        _, pb, hover, down, lift = grasp_points(cfg0, det_base, R, t)
        np.testing.assert_allclose(pb, pb_raw)
        np.testing.assert_allclose(hover, pb_raw + [0, 0, 0.10])
        np.testing.assert_allclose(down, pb_raw + [0, 0, -0.01])
        np.testing.assert_allclose(lift, down + [0, 0, 0.05])
        # 同一检测点，按尺寸覆盖后 l 走全局、m 走自己的偏移与高度
        cfg_size = _GraspCfg([-0.15, 0, 0], 0.10, 0.0, {
            'm': {'offset_xyz': np.array([-0.05, 0.01, 0.0]),
                  'z_offset': -0.02, 'hover_height': 0.08,
                  'lift_height': 0.07}})
        det_l = Detection('l', np.array([0.378, 0.326, -0.348]),
                          extra={'frame': 'base_link'})
        det_m = Detection('m', np.array([0.378, 0.326, -0.348]),
                          extra={'frame': 'base_link'})
        _, pb_l, hov_l, down_l, lift_l = grasp_points(cfg_size, det_l, R, t)
        _, pb_m, hov_m, down_m, lift_m = grasp_points(cfg_size, det_m, R, t)
        np.testing.assert_allclose(pb_l, [0.228, 0.326, -0.348])
        np.testing.assert_allclose(hov_l, [0.228, 0.326, -0.248])
        np.testing.assert_allclose(down_l, [0.228, 0.326, -0.348])
        np.testing.assert_allclose(lift_l, [0.228, 0.326, -0.298])
        np.testing.assert_allclose(pb_m, [0.328, 0.336, -0.348])
        np.testing.assert_allclose(hov_m, [0.328, 0.336, -0.268])
        np.testing.assert_allclose(down_m, [0.328, 0.336, -0.368])
        np.testing.assert_allclose(lift_m, [0.328, 0.336, -0.298])

    def _dets(self, labels=SIZE_LABELS):
        pts = {'l': [-0.1, 0.0, 0.7], 'm': [0.0, 0.0, 0.7], 's': [0.1, 0.0, 0.7]}
        return [Detection(k, np.array(pts[k])) for k in labels]

    def test_validate_ok_and_order(self):
        cfg = _CfgStub(order=('s', 'm', 'l'), require_all=True)
        by = validate_detections(cfg, self._dets())
        self.assertEqual(list(by), ['s', 'm', 'l'])

    def test_duplicate_detection_aborts(self):
        cfg = _CfgStub(order=('l', 'm', 's'), require_all=True)
        with self.assertRaises(TaskError):
            validate_detections(cfg, [Detection('l', np.zeros(3)),
                                      Detection('l', np.ones(3))])

    def test_duplicate_random_picks_one_and_continues(self):
        from unittest import mock
        cfg = _CfgStub(order=('l', 'm', 's'), require_all=False, policy='random')
        d1, d2 = Detection('m', np.array([0., 0., 0.7])), \
                 Detection('m', np.array([0.1, 0.1, 0.7]))
        # 两颗同类：不中止，返回且只返回其中一颗；random.choice 选谁就用谁
        with mock.patch('random.choice', side_effect=[d1, d2]):
            self.assertIs(validate_detections(cfg, [d1, d2])['m'], d1)
            self.assertIs(validate_detections(cfg, [d1, d2])['m'], d2)
        # 其余尺寸正常入选
        with mock.patch('random.choice', return_value=d1):
            by = validate_detections(cfg, [d1, d2, Detection('l', np.zeros(3))])
        self.assertEqual(set(by), {'l', 'm'})

    def test_duplicate_first_picks_list_head(self):
        cfg = _CfgStub(order=('l', 'm', 's'), require_all=False, policy='first')
        d1, d2 = Detection('m', np.array([0., 0., 0.7])), \
                 Detection('m', np.array([0.1, 0.1, 0.7]))
        self.assertIs(validate_detections(cfg, [d1, d2])['m'], d1)

    def test_missing_required_aborts(self):
        cfg = _CfgStub(order=('l', 'm', 's'), require_all=True)
        with self.assertRaises(TaskError):
            validate_detections(cfg, self._dets(['l', 's']))

    def test_missing_optional_skips(self):
        cfg = _CfgStub(order=('l', 'm', 's'), require_all=False)
        by = validate_detections(cfg, self._dets(['l', 's']))
        self.assertEqual(set(by), {'l', 's'})

    def test_validate_labels_subset_ignores_finished_sizes(self):
        from unittest import mock
        cfg = _CfgStub(order=('l', 'm', 's'), require_all=True)
        # 逐颗重识别：只在剩余子集 m/s 里挑选，画面里残留的 l 被忽略
        by = validate_detections(cfg, self._dets(), labels=('m', 's'))
        self.assertEqual(set(by), {'m', 's'})
        by = validate_detections(cfg, self._dets(), labels=('s',))
        self.assertEqual(set(by), {'s'})
        # 子集里缺 m -> require_all=true 仍中止（即使画面里有已抓走的 l）
        with self.assertRaises(TaskError):
            validate_detections(cfg, self._dets(['l', 's']), labels=('m', 's'))
        # 子集外的重复目标不触发 duplicate_policy=abort
        cfg_abort = _CfgStub(order=('l', 'm', 's'), require_all=True, policy='abort')
        dl1 = Detection('l', np.zeros(3)); dl2 = Detection('l', np.ones(3))
        ds = Detection('s', np.array([0.1, 0., 0.7]))
        by = validate_detections(cfg_abort, [dl1, dl2, ds], labels=('s',))
        self.assertIs(by['s'], ds)
        # 子集内的重复目标仍按 random 策略选
        cfg_rnd = _CfgStub(order=('l', 'm', 's'), require_all=True, policy='random')
        dm1 = Detection('m', np.zeros(3)); dm2 = Detection('m', np.ones(3))
        with mock.patch('random.choice', return_value=dm2):
            self.assertIs(validate_detections(cfg_rnd, [dm1, dm2, ds],
                                              labels=('m', 's'))['m'], dm2)

    def test_detect_until_complete_retry_uses_labels_subset(self):
        from unittest import mock
        cfg = _CfgStub(order=('l', 'm', 's'), require_all=True)
        d_m = Detection('m', np.zeros(3)); d_s = Detection('s', np.zeros(3))
        # 首轮只看到 m（子集中缺 s）-> 重试，第二轮齐全
        rounds = [[d_m], [d_m, d_s]]
        calls = []
        with mock.patch('time.sleep') as slept:
            dets, chosen = detect_until_complete(
                cfg, lambda: (calls.append(1) or rounds[len(calls) - 1]),
                what='重识别', labels=('m', 's'))
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(chosen), {'m', 's'})
        slept.assert_called_once_with(1.0)
        # 画面一直只有 l：对 ('m','s') 子集而言始终缺料，attempts 用尽后中止
        calls = []
        with mock.patch('time.sleep'):
            with self.assertRaises(TaskError):
                detect_until_complete(
                    cfg, lambda: (calls.append(1)
                                  or [Detection('l', np.zeros(3))]),
                    labels=('m', 's'))
        self.assertEqual(len(calls), 3)
        # require_all=false：子集缺料也不重试，只返回看到的
        cfg_loose = _CfgStub(order=('l', 'm', 's'), require_all=False)
        calls = []
        with mock.patch('time.sleep') as slept:
            _, chosen = detect_until_complete(
                cfg_loose, lambda: (calls.append(1) or [d_m]), labels=('m', 's'))
        self.assertEqual(len(calls), 1)
        slept.assert_not_called()
        self.assertEqual(set(chosen), {'m'})

    def test_handoff_check_flags_gap(self):
        release = _fake_leg((0.40, 0.05, -0.36))
        appr = _fake_leg((0.42, -0.20, -0.18))
        apprs = {k: appr for k in SIZE_LABELS}
        self.assertIn('⚠', handoff_check(release, apprs, True)[0])
        # 分段模式：每个尺寸各报一行
        rows = handoff_check(release, apprs, False)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all('⚠' in r for r in rows))

    def test_handoff_check_ok_aligned(self):
        p = (0.40, 0.05, -0.36)
        leg = _fake_leg(p)
        self.assertIn('ok', handoff_check(_fake_leg(p),
                                          {k: leg for k in SIZE_LABELS}, True)[0])

    def test_persize_place_start_alignment_rows(self):
        # 分立 place 模式：各 place 段起点应对齐右臂重抓点
        center = (0.42, -0.15, -0.24)
        appr = _fake_leg(center)
        apprs = {k: appr for k in SIZE_LABELS}
        places = {k: _fake_leg((0.5, -0.35, -0.20), xyz_start=center)
                  for k in SIZE_LABELS}
        rows = join_gap_rows([_fake_leg((0.5, 0.4, -0.2))], apprs, True, places, False)
        per = [r for r in rows if 'place_' in r and '段首' in r]
        self.assertEqual(len(per), 3)
        self.assertTrue(all('ok' in r for r in per))

        places_bad = {k: _fake_leg((0.5, -0.35, -0.20),
                                   xyz_start=(0.5, -0.0, -0.2))
                      for k in SIZE_LABELS}
        rows = join_gap_rows([_fake_leg((0.5, 0.4, -0.2))], apprs, True, places_bad, False)
        per = [r for r in rows if 'place_' in r and '段首' in r]
        self.assertTrue(all('⚠' in r for r in per))

    def test_persize_approach_start_mismatch_warns(self):
        # approach 三段首点不一致时（没走"复制只改末点"流程）要显式警告
        a0 = _fake_leg((0.42, -0.15, -0.24))
        apprs = {'l': a0,
                 'm': _fake_leg((0.42, -0.15, -0.24),
                                xyz_start=(0.30, -0.10, -0.10)),
                 's': a0}
        places = {k: _fake_leg((0.5, -0.35, -0.20)) for k in SIZE_LABELS}
        rows = join_gap_rows([_fake_leg((0.5, 0.4, -0.2))], apprs, False, places, True)
        warn = [r for r in rows if '首点不一致' in r]
        self.assertEqual(len(warn), 1)
        self.assertIn('approach_m', warn[0])

    def test_dwell_keeps_spinning(self):
        # 回归：停顿期间必须持续 spin 节点，否则反馈时间戳老化 >state_timeout 被误判过期
        import sys
        import types
        calls = []

        def fake_spin_once(node, timeout_sec=0.0):
            calls.append(timeout_sec)

        import contextlib
        import io
        fake_rclpy = types.ModuleType('rclpy')
        fake_rclpy.spin_once = fake_spin_once
        prev = sys.modules.get('rclpy')
        sys.modules['rclpy'] = fake_rclpy
        try:
            runner = SequenceRunner.__new__(SequenceRunner)
            runner.clients = {'left': types.SimpleNamespace(node=object()),
                              'right': types.SimpleNamespace(node=object())}
            with contextlib.redirect_stdout(io.StringIO()):
                runner._dwell(0.16)  # 至少应切成 3 次 50ms 的 spin
            self.assertGreaterEqual(len(calls), 3)
            n0 = len(calls)
            runner._dwell(0)    # 不停顿就不 spin
            self.assertEqual(len(calls), n0)
        finally:
            if prev is not None:
                sys.modules['rclpy'] = prev
            else:
                sys.modules.pop('rclpy', None)

    def _goto_harness(self, outcomes):
        """合成 runner/robot：move_joint 第 k 次调用后关节停在 outcomes[k]；
        假时钟随 spin 推进，避免真实等待。"""
        import io
        import types
        import nut_sequences

        class Robot:
            arm = 'right'
            node = object()
            joint_names = [f'j{i}' for i in range(7)]

            def __init__(self):
                self.joints = np.zeros(7)
                self.calls = 0

            def feedback_fresh(self, timeout):
                return True

            def joint_diffs(self, target):
                return [abs(a - b) for a, b in zip(self.joints, target)]

            def move_joint(self, q, speed, acce, timeout):
                self.joints = np.array(outcomes[self.calls], float)
                self.calls += 1

        cfg = types.SimpleNamespace(reached_wait_seconds=0.5, state_timeout=0.5,
                                    reached_tolerance=0.03, reached_tolerance_left=None,
                                    reached_tolerance_right=None, sequence_speed=0.3,
                                    sequence_acce=0.5, move_timeout=60,
                                    reached_reissue_count=2)
        runner = SequenceRunner.__new__(SequenceRunner)
        runner.cfg = cfg
        clock = {'t': 0.0}
        fake_rclpy = types.ModuleType('rclpy')
        fake_rclpy.spin_once = lambda node, timeout_sec=0.0: clock.__setitem__(
            't', clock['t'] + 0.02)
        prev = sys.modules.get('rclpy')
        sys.modules['rclpy'] = fake_rclpy
        mono_patch = mock.patch.object(nut_sequences.time, 'monotonic',
                                                lambda: clock['t'])
        return runner, Robot(), mono_patch, prev, io.StringIO()

    def test_goto_point_reissues_once_then_arrives(self):
        import contextlib
        runner, robot, mp, prev, buf = self._goto_harness(
            [np.full(7, 0.05), np.zeros(7)])  # 第一次停在 0.05rad，补发后到位
        try:
            mp.start()
            with contextlib.redirect_stdout(buf):
                runner._goto_point(robot, np.zeros(7), '回放 t 第2点')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.calls, 2)       # 首下 + 1 次补发
        self.assertIn('第1次补发后到位', buf.getvalue())

    def _goto_pose_harness(self, xyz_outcomes, target=None, eul=None):
        """合成笛卡尔运动：move_pose/move_linear 第 k 次后末端停在 xyz_outcomes[k]。"""
        import types
        from scipy.spatial.transform import Rotation as Rot
        import nut_sequences

        if target is None:
            target = np.array([0.4, 0.1, -0.35])
        if eul is None:
            eul = np.array([1.1, -0.1, -1.55])
        q_t = Rot.from_euler('xyz', eul).as_quat()

        class Robot:
            arm = 'left'
            node = object()

            def __init__(self):
                self.pose = None
                self.pose_calls = 0
                self.linear_calls = 0

            def pose_errors(self, position, euler_rad):
                p = self.pose[0]
                return float(np.linalg.norm(p - np.asarray(position))), 0.0, p

            def move_pose(self, position, e, speed, acce, timeout):
                self.pose = (np.array(xyz_outcomes[self.pose_calls], float), q_t)
                self.pose_calls += 1

            def move_linear(self, position, e, speed, acce, timeout):
                self.pose = (np.array(xyz_outcomes[self.pose_calls], float), q_t)
                self.pose_calls += 1
                self.linear_calls += 1

        cfg = types.SimpleNamespace(reached_wait_seconds=0.5, state_timeout=0.5,
                                    reached_reissue_count=2, pose_pos_tolerance=0.01,
                                    pose_ori_tolerance=0.05, speed=0.3, acce=0.3,
                                    linear_speed=0.2, linear_acce=0.2, move_timeout=60)
        runner = SequenceRunner.__new__(SequenceRunner)
        runner.cfg = cfg
        clock = {'t': 0.0}
        fake_rclpy = types.ModuleType('rclpy')
        fake_rclpy.spin_once = lambda node, timeout_sec=0.0: clock.__setitem__(
            't', clock['t'] + 0.02)
        prev = sys.modules.get('rclpy')
        sys.modules['rclpy'] = fake_rclpy
        mp = mock.patch.object(nut_sequences.time, 'monotonic', lambda: clock['t'])
        return runner, Robot(), mp, prev, target, eul

    def test_goto_pose_first_try_arrives(self):
        import contextlib, io
        runner, robot, mp, prev, tgt, eul = self._goto_pose_harness([np.array([0.4, 0.1, -0.35])])
        try:
            mp.start()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                runner._goto_pose(robot, tgt, eul, 'hover')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.pose_calls, 1)
        self.assertEqual(robot.linear_calls, 0)
        self.assertEqual(buf.getvalue(), '')

    def test_goto_pose_reissues_then_arrives_and_uses_linear(self):
        import contextlib, io
        runner, robot, mp, prev, tgt, eul = self._goto_pose_harness(
            [np.array([0.4, 0.1, -0.35]) + [0, 0, 0.02], np.array([0.4, 0.1, -0.35])])
        try:
            mp.start()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                runner._goto_pose(robot, tgt, eul, 'down', linear=True)
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.pose_calls, 2)
        self.assertEqual(robot.linear_calls, 2)   # 补发的也是 MoveL
        self.assertIn('实际', buf.getvalue())

    def test_goto_pose_failure_lists_actual_vs_command(self):
        import contextlib, io
        runner, robot, mp, prev, tgt, eul = self._goto_pose_harness(
            [(np.array([0.4, 0.1, -0.35])) + [0, 0, 0.03]] * 3)
        try:
            mp.start()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(TaskError) as cm:
                    runner._goto_pose(robot, tgt, eul, 'down', linear=True)
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.pose_calls, 3)
        msg = str(cm.exception)
        self.assertIn('30.0mm', msg)          # 位置差
        self.assertIn('-320.', msg)           # 实际 z mm
        self.assertIn('-350.', msg)           # 指令 z mm

    def test_goto_pose_still_moving_is_not_reenqueued(self):
        # block 提前返回但末端残差持续减小：只允许等，不许重发打断轨迹；
        # 第二个等待窗内自然收敛到位，全程只下发 1 次
        import contextlib, io, types
        import nut_sequences
        target = np.array([0.4, 0.1, -0.35])
        state = {'n': 0}

        class Robot:
            arm = 'left'
            node = object()

            def __init__(self):
                self.sends = 0

            def pose_errors(self, position, euler_rad):
                state['n'] += 1
                err = max(0.0, 0.030 - 0.0008 * state['n'])
                return err, 0.0, np.array(position) + np.array([0, 0, err])

            def move_pose(self, *a):
                self.sends += 1

        cfg = types.SimpleNamespace(reached_wait_seconds=0.5, state_timeout=0.5,
                                    reached_reissue_count=2, pose_pos_tolerance=0.01,
                                    pose_ori_tolerance=0.05, speed=0.3, acce=0.3,
                                    linear_speed=0.2, linear_acce=0.2, move_timeout=60)
        runner = SequenceRunner.__new__(SequenceRunner)
        runner.cfg = cfg
        clock = {'t': 0.0}
        fake = types.ModuleType('rclpy')
        fake.spin_once = lambda node, timeout_sec=0.0: clock.__setitem__('t', clock['t'] + 0.02)
        prev = sys.modules.get('rclpy')
        sys.modules['rclpy'] = fake
        mp = mock.patch.object(nut_sequences.time, 'monotonic', lambda: clock['t'])
        robot = Robot()
        try:
            mp.start()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                runner._goto_pose(robot, target, np.zeros(3), 'hover')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.sends, 1)
        self.assertIn('继续等待，不打断运动', buf.getvalue())
        self.assertNotIn('补发', buf.getvalue())

    def test_goto_pose_no_feedback_aborts(self):
        import contextlib, io
        runner, robot, mp, prev, tgt, eul = self._goto_pose_harness([np.array([0.4, 0.1, -0.35])])
        robot.pose = None
        robot.pose_errors = lambda p, e: None
        try:
            mp.start()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(TaskError):
                    runner._goto_pose(robot, tgt, eul, 'hover')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.pose_calls, 1)  # 无反馈不补发盲动

    def test_goto_point_right_arm_steady_residual_passes_with_override(self):
        # 现场场景：右臂肩滚转稳态停在 0.033rad（重发也不变），右臂容差放宽到 0.05 即直接通过
        import contextlib
        runner, robot, mp, prev, buf = self._goto_harness(
            [np.full(7, 0.033)])
        runner.cfg.reached_tolerance_right = 0.05
        try:
            mp.start()
            with contextlib.redirect_stdout(buf):
                runner._goto_point(robot, np.zeros(7), '接入 t 起点')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.calls, 1)       # 稳态残差在右臂容差内，不补发
        self.assertEqual(buf.getvalue(), '')

    def test_goto_point_left_keeps_global_tolerance(self):
        # 左臂无覆盖仍用全局 0.03：0.033 要补发；两臂容差相互独立
        import contextlib
        runner, robot, mp, prev, buf = self._goto_harness(
            [np.full(7, 0.033)] * 3)
        runner.cfg.reached_tolerance_right = 0.05
        robot.arm = 'left'
        try:
            mp.start()
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(TaskError):
                    runner._goto_point(robot, np.zeros(7), '回放 t 第2点')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.calls, 3)       # 左臂按 0.03 判，补发 2 次后中止
        self.assertIn('> 0.03', str(buf.getvalue()))

    def test_goto_point_arrives_on_second_reissue(self):
        # 现场场景：第一次补发还差 1.9°（0.033rad），第二次补发收敛
        import contextlib
        runner, robot, mp, prev, buf = self._goto_harness(
            [np.full(7, 0.05), np.full(7, 0.033), np.zeros(7)])
        try:
            mp.start()
            with contextlib.redirect_stdout(buf):
                runner._goto_point(robot, np.zeros(7), '接入 t 起点')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.calls, 3)       # 首下 + 2 次补发
        self.assertIn('第2次补发后到位', buf.getvalue())

    def test_goto_point_failure_reports_per_joint(self):
        import contextlib
        runner, robot, mp, prev, buf = self._goto_harness(
            [np.full(7, 0.05)] * 3)  # 补发 2 次后仍不到位
        try:
            mp.start()
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(TaskError) as cm:
                    runner._goto_point(robot, np.zeros(7), '回放 t 第2点')
        finally:
            mp.stop()
            sys.modules.pop('rclpy', None)
            if prev is not None:
                sys.modules['rclpy'] = prev
        self.assertEqual(robot.calls, 3)       # 首下+2 次补发，不再多发
        msg = str(cm.exception)
        self.assertIn('j6', msg)               # 列出超差关节名
        self.assertIn('2.9°', msg)             # 0.05rad ≈ 2.9°
        self.assertIn('补发 2 次后', msg)

    def test_ik_diagnose_probes(self):
        # 假 IK：只有在录段末点 (0.4,0.07,-0.35) 附近、不管姿态 e 都可解
        p_rec = np.array([0.4, 0.07, -0.35])

        class _R:
            def ik_try(self, p, e, seed, timeout=8.0):
                return 'ok' if np.linalg.norm(np.asarray(p) - p_rec) < 0.01 else 'fail'

        leg = _fake_leg(tuple(p_rec))
        leg.joints = np.zeros((2, 7))
        rows = ik_diagnose(_R(), leg, np.array([0.9, 0.9, -0.9]), np.zeros(3))
        status = [r.strip().split()[0] for r in rows
                  if r.strip().startswith(('ok', 'fail', 'TIMEOUT'))]
        # 录段末点自身 -> ok；手动验证位姿/失败点位置 -> fail；录段位置+目标姿态 -> ok
        self.assertEqual(status, ['ok', 'fail', 'fail', 'ok'])

    def test_normalize_deproject(self):
        K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
        det = normalize('l', u=320, v=240, z=0.5, K=K)
        np.testing.assert_allclose(det.p_cam, [0, 0, 0.5])
        det2 = normalize('m', p_cam=[0.1, 0.2, 0.6])
        np.testing.assert_allclose(det2.p_cam, [0.1, 0.2, 0.6])
        with self.assertRaises(TaskError):
            normalize('x', [0, 0, 0])


class DetectorAndStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_detector(self):
        dj = self.d / 'detections.json'
        dj.write_text(json.dumps({'detections': [
            {'label': 'l', 'p_cam': [1, 2, 3]},
            {'label': 's', 'p_cam': [4, 5, 6]},
        ]}), encoding='utf-8')
        out = JsonDetector(None, {'json_path': str(dj)}, None).detect(('l', 'm', 's'))
        self.assertEqual([d.label for d in out], ['l', 's'])
        np.testing.assert_allclose(out[0].p_cam, [1, 2, 3])

    def test_json_detector_reread_each_call(self):
        dj = self.d / 'detections.json'
        det = JsonDetector(None, {'json_path': str(dj)}, None)
        dj.write_text(json.dumps({'detections': [{'label': 'm', 'p_cam': [0, 0, 0.5]}]}),
                      encoding='utf-8')
        self.assertEqual([d.label for d in det.detect(('l', 'm', 's'))], ['m'])
        dj.write_text(json.dumps({'detections': [{'label': 's', 'p_cam': [0, 0, 0.6]}]}),
                      encoding='utf-8')
        self.assertEqual([d.label for d in det.detect(('l', 'm', 's'))], ['s'])

    def test_json_base_frame_passthrough(self):
        dj = self.d / 'detections.json'
        dj.write_text(json.dumps({'detections': [{
            'label': 'l', 'frame': 'base_link',
            'p_base': [0.378, 0.326, -0.348],
            'euler_rad': [0.924, 0.0, -1.506]}]}), encoding='utf-8')
        out = JsonDetector(None, {'json_path': str(dj)}, None).detect(('l', 'm', 's'))
        self.assertEqual(len(out), 1)
        np.testing.assert_allclose(out[0].p_cam, [0.378, 0.326, -0.348])
        self.assertEqual(out[0].extra['frame'], 'base_link')
        self.assertEqual(out[0].extra['euler_rad'], [0.924, 0.0, -1.506])

    def test_input_detector(self):
        lines = ['0.378 0.326 -0.348', '0.30,0.30,-0.35', '0.25 0.35 -0.34']
        prompts = []

        def reader(prompt):
            prompts.append(prompt)
            return lines.pop(0)

        det = InputDetector(reader=reader, printer=lambda s: None)
        out = det.detect(('l', 'm', 's'))
        self.assertEqual([d.label for d in out], ['l', 'm', 's'])
        np.testing.assert_allclose(out[0].p_cam, [0.378, 0.326, -0.348])
        np.testing.assert_allclose(out[1].p_cam, [0.30, 0.30, -0.35])
        self.assertEqual(out[0].extra['frame'], 'base_link')
        self.assertEqual(out[0].extra['source'], 'terminal_input')
        self.assertEqual(len(prompts), 3)

        # 空行=跳过该颗（require_all 由主流程校验）
        skip_lines = iter(['', '0.3 0.3 -0.35', ''])
        det_skip = InputDetector(reader=lambda p: next(skip_lines),
                                 printer=lambda s: None)
        out_skip = det_skip.detect(('l', 'm', 's'))
        self.assertEqual([d.label for d in out_skip], ['m'])

        # 全空 -> 中止
        det2 = InputDetector(reader=lambda p: '', printer=lambda s: None)
        with self.assertRaises(TaskError):
            det2.detect(('l', 'm', 's'))
        # 格式错 / 非数字
        det3 = InputDetector(reader=lambda p: '0.3 0.3', printer=lambda s: None)
        with self.assertRaises(TaskError):
            det3.detect(('l',))
        det4 = InputDetector(reader=lambda p: 'a b c', printer=lambda s: None)
        with self.assertRaises(TaskError):
            det4.detect(('l',))

    def test_pixel_wrapper_preserves_base_frame(self):
        det = Detection('l', np.array([0.378, 0.326, -0.348]),
                        extra={'frame': 'base_link',
                               'quat_xyzw': [0.325, -0.305, -0.612, 0.653]})
        w = self._pixel_wrapper([det], depth=None, K=None)  # 无深度/K 也应透传
        out = w.detect(('l', 'm', 's'))
        np.testing.assert_allclose(out[0].p_cam, [0.378, 0.326, -0.348])
        self.assertEqual(out[0].extra['frame'], 'base_link')

    def test_pixel_wrapper_dict_base_frame(self):
        w = self._pixel_wrapper([{'label': 'm', 'frame': 'base_link',
                                  'p_base': [0.4, 0.1, -0.3]}], None, None)
        out = w.detect(('l', 'm', 's'))
        np.testing.assert_allclose(out[0].p_cam, [0.4, 0.1, -0.3])
        self.assertEqual(out[0].extra['frame'], 'base_link')

    def test_pose_store_missing_hint(self):
        store = PoseStore(self.d / 'task_poses.yaml')
        with self.assertRaises(TaskError) as cm:
            store.get('left', 'left_grasp_init')
        self.assertIn('left_grasp_init', str(cm.exception))
        self.assertIn('capture_task_pose.py', str(cm.exception))

    def _pixel_wrapper(self, inner_out, depth=None, K=None):
        """绕过 ROS 订阅构造 DepthPixelDetector（深度/K 直接注入）。"""
        w = object.__new__(DepthPixelDetector)
        w.inner = type('I', (), {'detect': lambda self, exp: inner_out})()
        w.node = None
        w._depth = depth
        w._K = K
        w.cfg = {'_vision': {'depth_topic': '/x', 'color_info_topic': '/y'}}
        return w

    def test_pixel_only_autodepth_deproject(self):
        K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
        depth = np.full((480, 640), 500, np.uint16)  # 0.5m
        w = self._pixel_wrapper([Detection('l', None, u=320, v=240)], depth, K)
        out = w.detect(('l', 'm', 's'))
        self.assertEqual(len(out), 1)
        np.testing.assert_allclose(out[0].p_cam, [0, 0, 0.5])

    def test_dict_results_normalized(self):
        K = np.array([[600., 0, 320.], [0, 600., 140.], [0, 0, 1.]])
        depth = np.full((240, 640), 0.7)
        w = self._pixel_wrapper([{'label': 'm', 'u': 320, 'v': 140}], depth, K)
        out = w.detect(('l', 'm', 's'))
        np.testing.assert_allclose(out[0].p_cam, [0, 0, 0.7])

    def test_p_cam_passthrough_no_depth_needed(self):
        w = self._pixel_wrapper([Detection('s', np.array([0.1, 0.2, 0.6]))],
                                depth=None, K=None)
        out = w.detect(('l', 'm', 's'))
        np.testing.assert_allclose(out[0].p_cam, [0.1, 0.2, 0.6])

    def test_pixel_depth_hole_aborts(self):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 600.0
        w = self._pixel_wrapper([Detection('l', None, u=100, v=100)],
                                np.zeros((200, 200), np.uint16), K)
        with self.assertRaises(TaskError):
            w.detect(('l', 'm', 's'))

    def test_pixel_out_of_frame_aborts(self):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 600.0
        w = self._pixel_wrapper([Detection('l', None, u=999, v=10)],
                                np.full((200, 300), 0.5), K)
        with self.assertRaises(TaskError):
            w.detect(('l', 'm', 's'))

    def test_pose_store_roundtrip(self):
        path = self.d / 'task_poses.yaml'
        store = PoseStore(path)
        store.put('left', 'left_grasp_init', [0.3, 0.4, 0.2], [90, 0, 45],
                  [0.0] * 7)
        store.save()
        store2 = PoseStore(path)
        pose = store2.get('left', 'left_grasp_init')
        self.assertEqual(pose['position_m'], [0.3, 0.4, 0.2])
        self.assertEqual(pose['euler_deg'], [90, 0, 45])


class ResumeReadyTest(unittest.TestCase):
    """逐颗重识别前回 ready 末点：只 MoveJ 到末关节角，不接入 pt0/不重放开场段。"""

    def _ready_leg(self, name):
        return types.SimpleNamespace(
            target=name,
            joints=[np.full(6, 0.1), np.full(6, 0.9)])  # pt0 vs 末点明显不同

    def _env(self, diff_left=0.3, diff_right=0.3,
             ready_l=True, ready_r=True):
        cfg = types.SimpleNamespace(start_tolerance=0.05, join_speed=0.15,
                                 join_acce=0.15, point_dwell_seconds=0.0,
                                 ready_hold_seconds=0.0)

        class _R:
            def __init__(self, arm, diff):
                self.arm, self._diff = arm, diff

            def joint_diff(self, q):
                return self._diff

            def wait_state(self, t):
                return True

        class _Runner:
            def __init__(self):
                self.goto, self.dwells, self.joins = [], [], []

            def _goto_point(self, robot, q, where, speed=None, acce=None):
                self.goto.append((robot.arm, np.array(q, float).copy(), where,
                                  speed, acce))

            def _dwell(self, sec, what):
                self.dwells.append(what)

            def join_to_start(self, leg, tag=''):
                self.joins.append((leg.target, tag))

        left = _R('left', diff_left)
        right = _R('right', diff_right)
        runner = _Runner()
        left_legs = [types.SimpleNamespace(target='left_task', joints=[np.zeros(6)])]
        apprs = {"l": types.SimpleNamespace(target='right_approach',
                                         joints=[np.zeros(6)])}
        rl = self._ready_leg('left_ready_001') if ready_l else None
        rr = self._ready_leg('right_ready_001') if ready_r else None
        return cfg, runner, left_legs, apprs, left, right, rl, rr

    def test_direct_movej_to_ready_endpoint_not_pt0(self):
        cfg, runner, left_legs, apprs, left, right, rl, rr = self._env()
        with mock.patch('time.sleep'):
            resume_ready_for_detect(cfg, runner, left, right, left_legs, apprs,
                                    rl, rr)
        # 两臂各一条 MoveJ，目标必须是 ready 末关节角（0.9），绝不是 pt0（0.1）
        self.assertEqual([g[0] for g in runner.goto], ['left', 'right'])  # 左先右后
        for arm, q, where, speed, acce in runner.goto:
            np.testing.assert_allclose(q, np.full(6, 0.9))
            self.assertEqual(speed, cfg.join_speed)
            self.assertEqual(acce, cfg.join_acce)
        self.assertEqual(runner.joins, [])  # 没有接入段起点

    def test_already_at_endpoint_skips_move(self):
        cfg, runner, left_legs, apprs, left, right, rl, rr = \
            self._env(diff_left=0.01, diff_right=0.3)
        with mock.patch('time.sleep'):
            resume_ready_for_detect(cfg, runner, left, right, left_legs, apprs,
                                    rl, rr)
        self.assertEqual([g[0] for g in runner.goto], ['right'])  # 左臂已在末点不补动

    def test_no_ready_leg_falls_back_to_join_task_start(self):
        cfg, runner, left_legs, apprs, left, right, rl, rr = \
            self._env(ready_l=False, ready_r=False)
        with mock.patch('time.sleep'):
            resume_ready_for_detect(cfg, runner, left, right, left_legs, apprs,
                                    rl, rr)
        self.assertEqual(runner.goto, [])
        self.assertEqual([j[0] for j in runner.joins],
                         ['left_task', 'right_approach'])

    def test_feedback_missing_aborts(self):
        cfg, runner, left_legs, apprs, left, right, rl, rr = self._env()
        left._diff = None
        with mock.patch('time.sleep'):
            with self.assertRaises(TaskError):
                resume_ready_for_detect(cfg, runner, left, right, left_legs,
                                        apprs, rl, rr)


class _CfgStub:
    def __init__(self, order, require_all, policy='abort'):
        self.order = tuple(order)
        self.require_all = require_all
        self.duplicate_policy = policy


class _NeedleCfgStub:
    def __init__(self, **needle):
        self.needle_raw = needle


class _PoseRobotStub:
    """只带末端位姿反馈的最小机器人桩（pose = (xyz, quat_xyzw)）。"""
    def __init__(self, quat=None):
        self.pose = None if quat is None else (np.zeros(3), np.asarray(quat, float))


class RightSeedBankAndNeedleEulerTest(unittest.TestCase):
    """任务二：右臂种子库 + 叠针释放姿态。"""

    def setUp(self):
        self.cfg = TaskConfig(DEFAULT_CONFIG)
        self.left_legs, self.apprs, self.places, *_ = load_all_legs(self.cfg)

    def test_right_seed_bank_names_align_with_seeds(self):
        seeds, names = right_ik_seed_bank(self.apprs, self.places)
        # move_via_ik 的约定：names[0] 是「当前关节角」，比 seeds 多一个
        self.assertEqual(len(names), len(seeds) + 1)
        self.assertEqual(names[0], '当前关节角')
        self.assertTrue(seeds)
        self.assertTrue(all(len(np.asarray(q)) == 7 for q in seeds))

    def test_right_seed_bank_dedupes_shared_size_legs(self):
        # white 复用 l 档 -> 同一 Leg 对象被两个尺寸指向，只能取一次
        apprs = dict(self.apprs, white=self.apprs['l'])
        places = dict(self.places, white=self.places['l'])
        base_seeds, _ = right_ik_seed_bank(self.apprs, self.places)
        seeds, names = right_ik_seed_bank(apprs, places)
        self.assertEqual(len(seeds), len(base_seeds))
        self.assertEqual(len(set(names)), len(names))
        # 种子必须来自右臂自己的录段端点，而不是左臂的臂型
        known = set()
        for table in (self.apprs, self.places):
            for k in SIZE_LABELS:
                leg = table[k]
                known.add(tuple(np.asarray(leg.joints[0])))
                known.add(tuple(np.asarray(leg.joints[-1])))
        self.assertTrue(seeds)
        for q in seeds:
            self.assertIn(tuple(np.asarray(q)), known)

    def test_grip_euler_rad_matches_recorded_convention(self):
        from scipy.spatial.transform import Rotation as Rot
        want = [10.0, -5.0, 30.0]
        q = Rot.from_euler('xyz', want, degrees=True).as_quat()
        got = np.degrees(grip_euler_rad(_PoseRobotStub(q)))
        np.testing.assert_allclose(got, want, atol=1e-6)
        self.assertIsNone(grip_euler_rad(_PoseRobotStub(None)))

    def test_release_euler_precedence(self):
        from scipy.spatial.transform import Rotation as Rot
        q = Rot.from_euler('xyz', [1.0, 2.0, 3.0], degrees=True).as_quat()
        robot = _PoseRobotStub(q)
        # 1) 显式配置优先
        cfg = _NeedleCfgStub(release_euler_deg=[40.0, 0.0, 90.0])
        np.testing.assert_allclose(np.degrees(release_euler(cfg, robot, [7., 7., 7.])),
                                   [40.0, 0.0, 90.0], atol=1e-6)
        # 2) 未配置 -> 当前抓握姿态
        np.testing.assert_allclose(
            np.degrees(release_euler(_NeedleCfgStub(), robot, [7., 7., 7.])),
            [1.0, 2.0, 3.0], atol=1e-6)
        # 3) 无姿态反馈 -> 回退入盒段末点姿态
        np.testing.assert_allclose(
            np.degrees(release_euler(_NeedleCfgStub(), _PoseRobotStub(None), [7., 8., 9.])),
            [7.0, 8.0, 9.0], atol=1e-6)

    def test_needle_options_validates_new_keys(self):
        opt = needle_options(self.cfg)
        self.assertIsNone(opt['release_euler_deg'])
        self.assertAlmostEqual(opt['max_joint_diff_deg'], 60.0)
        np.testing.assert_allclose(
            needle_options(_NeedleCfgStub(release_euler_deg=[1, 2, 3]))['release_euler_deg'],
            [1, 2, 3])
        with self.assertRaises(TaskError):
            needle_options(_NeedleCfgStub(release_euler_deg=[1, 2]))
        with self.assertRaises(TaskError):
            needle_options(_NeedleCfgStub(max_joint_diff_deg=0))
        with self.assertRaises(TaskError):
            needle_options(_NeedleCfgStub(max_joint_diff_deg='x'))


if __name__ == '__main__':
    unittest.main()
