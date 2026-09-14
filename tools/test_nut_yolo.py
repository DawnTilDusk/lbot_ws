import os
import time
import unittest
from unittest import mock
import numpy as np
import nut_yolo
from nut_yolo import locate, resolve_records, annotate, DetectionWindow
from nut_robot import TaskError
from nut_pick_place import det_base_xyz, validate_detections
from nut_detectors import Detection
from types import SimpleNamespace


def _pair(age_s=0.0, tag=0):
    """合成 (color, depth) 配对帧；帧结构 (消息戳, 接收 monotonic, frame)。"""
    mono = time.monotonic() - age_s
    return ((float(tag), mono, np.full((4, 4, 3), tag, np.uint8)),
            (float(tag), mono, np.full((4, 4), 0.5, np.float32)))


class ResolveRecordsRetryTests(unittest.TestCase):
    def test_fresh_first_attempt_runs_infer_once(self):
        pair0 = _pair(0.0)
        calls = []
        records, pair = resolve_records(
            {'max_result_age': 10},
            lambda timeout: pair0,
            lambda img: calls.append(img) or [{'label': 'm'}],
            log=lambda *_: None)
        self.assertEqual(records, [{'label': 'm'}])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], pair0[0][2])
        self.assertIs(pair, pair0)

    def test_stale_snapshot_retries_with_fresh_pair(self):
        pairs = [_pair(age_s=30, tag=1), _pair(age_s=0, tag=2)]
        used_frames, waits = [], []

        def wait_pair(timeout):
            waits.append(timeout)
            return pairs.pop(0)

        def run_infer(img):
            used_frames.append(int(img[0, 0, 0]))
            return [{'label': 'l'}]

        records, pair = resolve_records({'max_result_age': 10}, wait_pair, run_infer,
                                        log=lambda *_: None)
        self.assertEqual(used_frames, [1, 2])          # 旧帧丢弃，用新帧重新识别
        self.assertEqual(int(pair[0][2][0, 0, 0]), 2)  # 返回的是第二次的配对
        self.assertEqual(len(waits), 2)

    def test_inference_failure_retries_then_succeeds(self):
        pairs = iter([_pair(0, 1), _pair(0, 2)])
        attempts = {'n': 0}

        def run_infer(img):
            attempts['n'] += 1
            if attempts['n'] == 1:
                raise TaskError('YOLO 推理失败：超时')
            return [{'label': 's'}]

        records, _ = resolve_records({'max_result_age': 10},
                                     lambda timeout: next(pairs), run_infer,
                                     log=lambda *_: None)
        self.assertEqual(records, [{'label': 's'}])
        self.assertEqual(attempts['n'], 2)

    def test_all_stale_aborts_after_three_attempts_by_default(self):
        calls = {'wait': 0, 'infer': 0}

        def wait_pair(timeout):
            calls['wait'] += 1
            return _pair(age_s=99)

        def run_infer(img):
            calls['infer'] += 1
            return []

        with self.assertRaises(TaskError):
            resolve_records({'max_result_age': 10}, wait_pair, run_infer,
                            log=lambda *_: None)
        self.assertEqual(calls, {'wait': 3, 'infer': 3})

    def test_attempt_cap_configurable(self):
        calls = {'n': 0}
        with self.assertRaises(TaskError):
            resolve_records({'max_result_age': 10, 'inference_retries': 1},
                            lambda timeout: _pair(age_s=99),
                            lambda img: (calls.__setitem__('n', calls['n'] + 1), []),
                            log=lambda *_: None)
        self.assertEqual(calls['n'], 1)

    def test_frame_timeout_aborts_immediately_without_inference(self):
        infer_calls = [0]
        with self.assertRaises(TaskError):
            resolve_records({'max_result_age': 10},
                            lambda timeout: None,
                            lambda img: infer_calls.__setitem__(0, infer_calls[0] + 1),
                            log=lambda *_: None)
        self.assertEqual(infer_calls[0], 0)


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100.,0,20],[0,100.,20],[0,0,1]])
        self.records = [dict(label='l',u=30.,v=10.,confidence=.9,bbox=[25,5,35,15])]

    def test_mm_and_meters_agree_and_business_transform(self):
        for depth in (np.full((40,40),500,np.uint16), np.full((40,40),.5,np.float32)):
            d = locate(self.records,depth,self.K,(40,40,3),{})[0]
            np.testing.assert_allclose(d.p_cam,[.05,-.05,.5])
            # Existing business must apply camera->base transform exactly once.
            R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
            np.testing.assert_allclose(det_base_xyz(d,R,np.array([1,2,3])),[1.05,2.05,3.5])

    def test_hole_and_wrong_dimensions_rejected(self):
        with self.assertRaises(TaskError):
            locate(self.records,np.zeros((40,40),np.uint16),self.K,(40,40,3),{})
        with self.assertRaises(TaskError):
            locate(self.records,np.ones((20,20),np.float32),self.K,(40,40,3),{})

    def test_roi(self):
        self.assertEqual(locate(self.records,np.ones((40,40),np.float32),self.K,
                                (40,40,3),{'roi':[0,0,20,40]}),[])

    def test_business_rejects_duplicate_or_missing(self):
        d=locate(self.records,np.ones((40,40),np.float32),self.K,(40,40,3),{})[0]
        cfg=SimpleNamespace(order=('l','m','s'),require_all=True)
        with self.assertRaises(TaskError): validate_detections(cfg,[d])
        with self.assertRaises(TaskError): validate_detections(cfg,[d,d])

    def test_invalid_intrinsics(self):
        with self.assertRaises(TaskError):
            locate(self.records,np.ones((40,40),np.float32),np.zeros((3,3)),(40,40,3),{})


class AnnotateTests(unittest.TestCase):
    COLOR = np.zeros((60, 80, 3), np.uint8)
    RECORDS = [{'label': 'm', 'confidence': .92, 'bbox': [10.0, 10.0, 40.0, 40.0],
                'u': 25.0, 'v': 25.0},
               {'label': 's', 'confidence': .5, 'bbox': [50.0, 40.0, 70.0, 55.0],
                'u': 60.0, 'v': 47.0}]

    def test_draws_boxes_without_mutating_input(self):
        out = annotate(self.COLOR, self.RECORDS)
        self.assertEqual(out.shape, self.COLOR.shape)
        self.assertFalse(np.array_equal(out, self.COLOR))
        self.assertTrue(np.array_equal(self.COLOR, np.zeros_like(self.COLOR)))

    def test_failed_label_box_red_and_located_depth_appended(self):
        det = Detection('m', np.zeros(3), u=25.0, v=25.0, z=.4, extra={})
        out = annotate(self.COLOR, self.RECORDS,
                       located=[det], failed_label='m', status='bad depth')
        self.assertEqual(out.shape, self.COLOR.shape)
        self.assertFalse(np.array_equal(out, self.COLOR))

    def test_status_only_frame(self):
        out = annotate(self.COLOR, [], status='waiting ...')
        self.assertEqual(out.shape, self.COLOR.shape)
        self.assertFalse(np.array_equal(out, self.COLOR))


class _FakeCv2:
    """记录 imshow/waitKey 调用，键值按序列返回；-1&0xFF=255 表示无键。"""
    def __init__(self, keys=(), imshow_error=False):
        self.keys = list(keys)
        self.shown = 0
        self.destroyed = False
        self._imshow_error = imshow_error

    def namedWindow(self, *a, **k): pass
    def imshow(self, *a, **k):
        self.shown += 1
        if self._imshow_error:
            raise RuntimeError('display lost')

    def waitKey(self, ms=0):
        return self.keys.pop(0) if self.keys else 255

    def destroyWindow(self, *a, **k):
        self.destroyed = True


class DetectionWindowTests(unittest.TestCase):
    def _bare(self, seconds=2.0, keys=()):
        w = DetectionWindow.__new__(DetectionWindow)
        w.enabled = True
        w.seconds = seconds
        w.cv2 = _FakeCv2(keys)
        return w

    def test_disabled_by_default_is_noop(self):
        w = DetectionWindow({})
        self.assertFalse(w.enabled)
        frame = np.zeros((4, 4, 3), np.uint8)
        w.live(frame)
        w.result(frame, [])
        w.dwell()  # 不应阻塞/报错
        w.close()

    def test_headless_disables_even_when_requested(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            w = DetectionWindow({'show_window': True, 'show_seconds': 1.0})
        self.assertFalse(w.enabled)
        self.assertIsNone(w.cv2)

    def test_space_and_enter_skip_dwell(self):
        w = self._bare(keys=(32,))
        w.dwell()  # 空格立即放行，不等 2s
        self.assertEqual(w.cv2.keys, [])
        w = self._bare(keys=(13,))
        w.dwell()  # 回车同样放行

    def test_q_or_esc_aborts(self):
        w = self._bare(keys=(ord('q'),))
        with self.assertRaises(TaskError):
            w.dwell()
        w = self._bare(keys=(27,))
        with self.assertRaises(TaskError):
            w.dwell()

    def test_dwell_auto_continues_after_timeout(self):
        w = self._bare(seconds=2.0)
        ticks = iter([1000.0] + [1000.0 + 0.1 * i for i in range(1, 25)])
        with mock.patch.object(nut_yolo.time, 'monotonic', side_effect=lambda: next(ticks)):
            w.dwell()  # 假时钟推进 2s 后自动放行（无真实等待）

    def test_live_result_close_lifecycle(self):
        w = self._bare(keys=(32,))
        frame = np.zeros((6, 6, 3), np.uint8)
        records = [{'label': 'l', 'confidence': .8, 'bbox': [0, 0, 4, 4], 'u': 2, 'v': 2}]
        w.live(frame)
        w.result(frame, records)
        w.dwell()
        w.close()
        self.assertEqual(w.cv2.shown, 2)
        self.assertTrue(w.cv2.destroyed)

    def test_imshow_failure_self_disables(self):
        w = self._bare()
        w.cv2 = _FakeCv2(imshow_error=True)
        w.live(np.zeros((4, 4, 3), np.uint8))
        self.assertFalse(w.enabled)
        w.dwell()  # 已禁用：不阻塞
        w.close()

    def test_bad_show_seconds_disables_window_not_detection(self):
        # 构造器吞掉非法秒数：窗口禁用，识别流程不受影响（headless 也安全）
        with mock.patch.dict(os.environ, {}, clear=True):
            w = DetectionWindow({'show_window': True, 'show_seconds': 'x'})
        self.assertFalse(w.enabled)
        self.assertEqual(w.seconds, 2.0)


if __name__ == '__main__':
    unittest.main()
