#!/usr/bin/env python3
"""双臂交接版 nut_pick_place 离线测试（不连机器人/相机；用真实预录 trace 加载段）：

  cd tools
  /usr/bin/python3 -m unittest -v test_nut_task.py
"""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from nut_robot import (DEFAULT_CONFIG, PoseStore, SIZE_LABELS, TaskConfig,
                       TaskError)
from nut_detectors import (DepthPixelDetector, Detection, JsonDetector,
                           normalize)
from nut_sequences import Leg, load_leg
from nut_pick_place import (cam_to_base, det_base_xyz, handoff_check,
                            join_gap_rows, load_all_legs, parse_order,
                            validate_detections)

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

    def test_load_all_legs_real_traces(self):
        left_legs, appr, places, release = load_all_legs(self.cfg)
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
            self.assertEqual(cfg.close_for('left', k), [0, 40, 0, 0, 0, 255])
            self.assertEqual(cfg.close_for('right', k), [0, 80, 0, 0, 0, 255])

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
  legs:
    - {{sequence: left_grasp_place_middle_001, hand_after: open}}
    - {{file: {LEFT_BACK_DIR}/events.jsonl, sequence: left_middle_back_001}}
right:
  trace: {RIGHT_GRASP_DIR}/events.jsonl
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
                         (LEFT_GRASP_DIR, LEFT_BACK_DIR, RIGHT_GRASP_DIR, RIGHT_BACK_DIR)),
                     '2026-09-10 正式录制文件夹不在')
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
        left_legs, appr, places, release = load_all_legs(self.cfg)
        self.assertEqual(len(left_legs), 2)
        self.assertIs(release, left_legs[0])
        self.assertEqual([l.target for l in left_legs],
                         ['left_grasp_place_middle_001', 'left_middle_back_001'])
        self.assertIs(places['l'], places['m'])
        self.assertIs(places['m'], places['s'])
        self.assertEqual(places['l'].target, 'right_middle_back_001')

    def test_real_handoff_gap_reported(self):
        left_legs, appr, places, release = load_all_legs(self.cfg)
        row = handoff_check(release, appr)[0]
        self.assertIn('69mm', row)
        self.assertIn('⚠', row)
        rows = join_gap_rows(left_legs, appr, places, True)
        # 左段间接入、左臂回环、右臂抓后抬离、右臂回环 = 4 行；共享模式无 per-size 行
        self.assertEqual(len(rows), 4)
        self.assertTrue(any('place段首' in r for r in rows))


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
    def __init__(self, order, require_all):
        self.order = tuple(order)
        self.require_all = require_all


if __name__ == '__main__':
    unittest.main()
