import unittest
import numpy as np
from nut_yolo import locate
from nut_robot import TaskError
from nut_pick_place import det_base_xyz, validate_detections
from types import SimpleNamespace


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
