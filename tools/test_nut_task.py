#!/usr/bin/env python3
"""双臂交接版 nut_pick_place 离线测试（不连机器人/相机；用真实预录 trace 加载段）：

  cd tools
  /usr/bin/python3 -m unittest -v test_nut_task.py
"""
import argparse
import json
import sys
import tempfile
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
from nut_pick_place import (cam_to_base, det_base_xyz, grasp_points, handoff_check,
                            ik_diagnose, join_gap_rows, load_all_legs, parse_order,
                            resolve_grasp_euler, validate_detections)

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
        self.assertEqual(cfg.order, ('l', 'm', 's'))
        self.assertEqual(cfg.left_grasp_pose_name, 'left_grasp_init')
        self.assertTrue(cfg.require_all)

    def test_leg_specs_parsed(self):
        self.assertEqual(len(self.cfg.left_legs), 2)
        f0, t0, h0, r0 = self.cfg.left_legs[0]
        self.assertEqual((t0, h0, r0), ('left_ready1_001', None, False))
        f1, t1, h1, r1 = self.cfg.left_legs[1]
        self.assertEqual((t1, h1, r1), ('left_place1_002', 'open', True))
        self.assertEqual(f1, LEFT_TRACE)
        self.assertEqual(self.cfg.right_approach[1:], ('right_ready1_001', 'close', False))
        for k in SIZE_LABELS:
            self.assertEqual(self.cfg.right_place[k][2], 'open')
            self.assertTrue(self.cfg.right_place[k][3])

    def test_ready_optional_when_unconfigured(self):
        self.assertIsNone(self.cfg.left_ready)
        self.assertIsNone(self.cfg.right_ready)

    def test_load_all_legs_real_traces(self):
        left_legs, appr, places, release, ready_left, ready_right = load_all_legs(self.cfg)
        self.assertIsNone(ready_left)
        self.assertIsNone(ready_right)
        self.assertEqual([l.target for l in left_legs],
                         ['left_ready1_001', 'left_place1_002'])
        self.assertIs(release, left_legs[1])
        self.assertEqual(appr.target, 'right_ready1_001')
        self.assertEqual(appr.hand_after, 'close')
        self.assertEqual([places[k].target for k in SIZE_LABELS],
                         ['right_place1_002', 'right_place2_003', 'right_place3_004'])
        for leg in left_legs + [appr] + list(places.values()):
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
            self.assertEqual(cfg.close_for('left', k), [0, 40, 0, 0, 0, 0])
            self.assertEqual(cfg.close_for('right', k), [0, 0, 0, 0, 0, 0])

    def test_real_config_open_and_pacing(self):
        cfg = TaskConfig(DEFAULT_CONFIG)
        self.assertEqual(cfg.hand_open_vals, [200, 80, 255, 255, 255, 255])
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
        # 夹具/缺省无分臂覆盖；真实 yaml 右臂 0.05、左臂 None（用全局 0.03）
        bare = TaskConfig(self.yaml_path)
        self.assertIsNone(bare.reached_tolerance_left)
        self.assertIsNone(bare.reached_tolerance_right)
        real = TaskConfig(DEFAULT_CONFIG)
        self.assertIsNone(real.reached_tolerance_left)
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
        self.assertEqual(cfg.left_ready[0].parent.name, 'left_trace2')
        self.assertEqual(cfg.left_ready[1], 'left_ready2_001')
        self.assertIsNone(cfg.left_ready[2])
        self.assertEqual(cfg.right_ready[0].parent.name, 'right_trace')
        self.assertEqual(cfg.right_ready[1], 'right_ready1_001')
        _, _, _, _, rl, rr = load_all_legs(cfg)
        self.assertEqual(rl.target, 'left_ready2_001')
        self.assertEqual(rr.target, 'right_ready1_001')

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
        self.assertEqual(self.cfg.right_approach[1:3],
                         ('right_grasp_middle_001', 'close'))

    def test_load_shared_legs(self):
        left_legs, appr, places, release, ready_left, ready_right = load_all_legs(self.cfg)
        self.assertEqual(len(left_legs), 2)
        self.assertIs(release, left_legs[0])
        self.assertEqual([l.target for l in left_legs],
                         ['left_grasp_place_middle_001', 'left_middle_back_001'])
        self.assertIs(places['l'], places['m'])
        self.assertIs(places['m'], places['s'])
        self.assertEqual(places['l'].target, 'right_middle_back_001')
        self.assertEqual(ready_left.target, 'left_ready1_001')
        self.assertEqual(ready_right.target, 'right_ready1_001')

    def test_real_handoff_gap_reported(self):
        left_legs, appr, places, release, ready_left, ready_right = load_all_legs(self.cfg)
        row = handoff_check(release, appr)[0]
        # right_grasp_middle 末点 2026-09-10 抬高 3cm（-337.7→-307.7mm）后，交接差 69→80mm
        self.assertIn('80mm', row)
        self.assertIn('⚠', row)
        rows = join_gap_rows(left_legs, appr, places, True, ready_left, ready_right)
        # 开机 ready 两行 + 左段间接入/左臂回环/右臂抓后抬离/右臂回环 = 6 行
        self.assertEqual(len(rows), 6)
        self.assertTrue(any('place段首' in r for r in rows))
        ready_rows = [r for r in rows if 'ready' in r]
        self.assertEqual(len(ready_rows), 2)
        self.assertTrue(any('left_ready1_001' in r for r in ready_rows))
        self.assertTrue(any('right_ready1_001' in r for r in ready_rows))

    def test_join_rows_without_ready_stays_four(self):
        # 不配 ready 段时不增加开机行（旧行为）
        left_legs, appr, places, _, _, _ = load_all_legs(self.cfg)
        rows = join_gap_rows(left_legs, appr, places, True)
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
            return type('R', (), {'success': ok})()
        rc._spin_call = fake_spin
        rc.joints = [0.0] * 7
        ok, used = rc.ik_check([0.3, 0.3, -0.3], [0.9, 0, -1.5],
                               extra_seeds=[[1.0] * 7])
        self.assertTrue(ok)
        self.assertEqual(used, 1)
        self.assertEqual(len(calls), 2)

        def all_fail(cli, req, timeout):
            return type('R', (), {'success': False})()
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
        from types import SimpleNamespace
        cfg = SimpleNamespace(grasp_offset_xyz=np.array([-0.15, 0.0, 0.0]),
                              hover_height=0.10, grasp_z_offset=0.0)
        R = np.diag([1., -1., 1.])
        t = np.array([1., 2., 3.])
        # 相机系结果：先外参变换到 base，再在 base 系向机体方向退 15cm
        det_cam = Detection('l', np.array([0., 1., 0.]))
        pb_raw, pb, hover, down = grasp_points(cfg, det_cam, R, t)
        np.testing.assert_allclose(pb_raw, [1, 1, 3])
        np.testing.assert_allclose(pb, [0.85, 1, 3])
        np.testing.assert_allclose(hover, [0.85, 1, 3.1])
        np.testing.assert_allclose(down, [0.85, 1, 3])
        # base 直给（input/json frame=base_link）同样施加偏移，不走外参
        det_base = Detection('m', np.array([0.378, 0.326, -0.348]),
                             extra={'frame': 'base_link'})
        pb_raw, pb, hover, down = grasp_points(cfg, det_base, R, t)
        np.testing.assert_allclose(pb_raw, [0.378, 0.326, -0.348])
        np.testing.assert_allclose(pb, [0.228, 0.326, -0.348])
        # 零偏移 + z 微调时退化为原来的 hover/down 公式
        cfg0 = SimpleNamespace(grasp_offset_xyz=np.zeros(3),
                               hover_height=0.10, grasp_z_offset=-0.01)
        _, pb, hover, down = grasp_points(cfg0, det_base, R, t)
        np.testing.assert_allclose(pb, pb_raw)
        np.testing.assert_allclose(hover, pb_raw + [0, 0, 0.10])
        np.testing.assert_allclose(down, pb_raw + [0, 0, -0.01])

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

    def test_handoff_check_flags_gap(self):
        release = _fake_leg((0.40, 0.05, -0.36))
        appr = _fake_leg((0.42, -0.20, -0.18))
        self.assertIn('⚠', handoff_check(release, appr)[0])

    def test_handoff_check_ok_aligned(self):
        p = (0.40, 0.05, -0.36)
        self.assertIn('ok', handoff_check(_fake_leg(p), _fake_leg(p))[0])

    def test_persize_place_start_alignment_rows(self):
        # 分立 place 模式：各 place 段起点应对齐右臂重抓点
        center = (0.42, -0.15, -0.24)
        appr = _fake_leg(center)
        places = {k: _fake_leg((0.5, -0.35, -0.20), xyz_start=center)
                  for k in SIZE_LABELS}
        rows = join_gap_rows([_fake_leg((0.5, 0.4, -0.2))], appr, places, False)
        per = [r for r in rows if 'place_' in r and '段首' in r]
        self.assertEqual(len(per), 3)
        self.assertTrue(all('ok' in r for r in per))

        places_bad = {k: _fake_leg((0.5, -0.35, -0.20),
                                   xyz_start=(0.5, -0.0, -0.2))
                      for k in SIZE_LABELS}
        rows = join_gap_rows([_fake_leg((0.5, 0.4, -0.2))], appr, places_bad, False)
        per = [r for r in rows if 'place_' in r and '段首' in r]
        self.assertTrue(all('⚠' in r for r in per))

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


class _CfgStub:
    def __init__(self, order, require_all, policy='abort'):
        self.order = tuple(order)
        self.require_all = require_all
        self.duplicate_policy = policy


if __name__ == '__main__':
    unittest.main()
