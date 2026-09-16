import unittest
from types import SimpleNamespace
import numpy as np
from PIL import ImageFont
from nut_yolo_infer import pose_keypoints, validate_model
from nut_yolo_live import coordinates, render, preview_result


class NeedleLiveTests(unittest.TestCase):
    def test_bad_keypoints_are_not_drawn_or_serialized_as_coordinates(self):
        for bad in ([float('nan'), 20, .9], [20,20,.1], [100,20,.9], [0,0,.9]):
            points=pose_keypoints([bad,[40,30,.9]],100,100,.5)
            self.assertFalse(points[0]['valid'])
            self.assertIsNone(points[0]['u'])
            self.assertTrue(points[1]['valid'])
            row=dict(task='pose',label='needle',confidence=.9,bbox=[10,10,60,60],keypoints=points)
            color=np.zeros((100,100,3),np.uint8)
            pair=((1,1,color),(1,1,np.full((100,100),500,np.uint16)))
            rows=coordinates([row],pair,dict(width=100,height=100,frame='camera'),np.eye(3),np.zeros(3),{}, {})
            self.assertNotIn('p_cam',rows[0])
            self.assertNotIn('p_base',rows[0])
            view=render(color,rows,'test',ImageFont.truetype('DejaVuSans.ttf',16))
            self.assertEqual(view.shape[1],680)
            self.assertIsNone(preview_result((pair,rows),20,False,1,10))

    def test_pose_is_opt_in_so_robot_detection_path_still_rejects_it(self):
        m=SimpleNamespace(task='pose',names={0:'needle'},model=SimpleNamespace(model=[SimpleNamespace(kpt_shape=[2,3])]))
        with self.assertRaises(ValueError):validate_model(m)
        validate_model(m,True)
        m.model.model[-1].kpt_shape=[17,3]
        with self.assertRaises(ValueError):validate_model(m,True)
        for names in [{0:'large',1:'medium',2:'small'}, {0:'large',1:'medium',2:'small',3:'white'}]:
            validate_model(SimpleNamespace(task='detect',names=names))

if __name__=='__main__':unittest.main()
