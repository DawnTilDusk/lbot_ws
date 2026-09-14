import unittest
import numpy as np
from nut_yolo_live import choose_pair, coordinates, preview_result


class LiveTests(unittest.TestCase):
    def test_delayed_detection_keeps_boxes_on_original_frame_but_hides_coordinates(self):
        pair=((1,10,object()),(1,10,object()))
        rows=[dict(bbox=[1,2,3,4],p_cam=[0,0,1],p_base=[1,0,0],z=1)]
        shown=preview_result((pair,rows),12,True,1,10)
        self.assertIs(shown[0],pair)
        self.assertEqual(shown[1][0]['bbox'],rows[0]['bbox'])
        self.assertNotIn('p_cam',shown[1][0])
        self.assertNotIn('p_base',shown[1][0])
        self.assertNotIn('z',shown[1][0])
        self.assertIn('p_cam',rows[0])
        self.assertIs(preview_result((pair,rows),10.5,True,1,10)[1],rows)
        self.assertIsNone(preview_result((pair,rows),21,True,1,10))
        self.assertIsNone(preview_result((pair,rows),10.5,False,1,10))

    def test_latest_pair_skips_backlog(self):
        colors=[(1.,9.5,None),(2.,9.8,None)]
        depths=[(1.01,9.5,None),(2.04,9.8,None)]
        pair=choose_pair(colors,depths,10.,1.,.1,-1)
        self.assertEqual(pair[0][0],2.)
        self.assertIsNone(choose_pair(colors,depths,10.,1.,.1,2.))

    def test_stale_or_unsynchronized_rejected(self):
        self.assertIsNone(choose_pair([(1,5,None)],[(1,5,None)],10,1,.1,-1))
        self.assertIsNone(choose_pair([(1,10,None)],[(2,10,None)],10,1,.1,-1))

    def test_bad_depth_does_not_hide_other_detection(self):
        color=np.zeros((100,100,3),np.uint8)
        depth=np.zeros((100,100),np.uint16);depth[10:30,10:30]=500
        records=[dict(label='l',u=20,v=20,confidence=.8,bbox=[10,10,30,30]),
                 dict(label='s',u=80,v=80,confidence=.8,bbox=[70,70,90,90])]
        info=dict(height=100,width=100,frame='camera',K=np.array([[100,0,50],[0,100,50],[0,0,1]]))
        rows=coordinates(records,((1,1,color),(1,1,depth)),info,np.eye(3),np.zeros(3),
                         dict(parent_frame='base_link',child_frame='camera'),{})
        np.testing.assert_allclose(rows[0]['p_base'],[-.15,-.15,.5])
        self.assertNotIn('p_base',rows[1]);self.assertIn('error',rows[1])


if __name__=='__main__':unittest.main()
