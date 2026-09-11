import time
import unittest
import numpy as np
from nut_yolo import locate, resolve_records
from nut_robot import TaskError
from nut_pick_place import det_base_xyz, validate_detections
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


if __name__ == '__main__':
    unittest.main()
