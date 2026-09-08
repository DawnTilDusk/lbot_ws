import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from replay_workpoints import plan, execute, compact_route


def event(kind, t, q=0., valid=True):
    state = {}
    for arm in ('left_arm', 'right_arm'):
        state[arm+'/joint_states'] = {'message': {'name': [f'{arm}_{i}' for i in range(7)],
            'position': [q if arm == 'right_arm' else 0.] * 7, 'header': {'frame_id': 'base_link'}}}
    return dict(type=kind, valid=valid, elapsed_seconds=t, states=state)


def events():
    return [dict(type='metadata', schema_version=1, namespace='/robot1'),
            event('sample', 0., valid=False), event('sample', .1),
            dict(event('waypoint', .15), point_id='point_001', label='start'),
            event('sample', .2, .01), event('sample', .3, .02),
            dict(event('waypoint', .35, .025), point_id='point_002', label='lift')]


class ReplayTests(unittest.TestCase):
    def make_plan(self, es=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.jsonl'
            p.write_text('\n'.join(json.dumps(e) for e in (es or events())))
            return plan(p, 'lift', kwargs.pop('start', None), 'right_arm', **kwargs)

    def test_startup_skip_and_named_segment(self):
        p=self.make_plan()
        self.assertEqual(p['skipped'],1)
        p=self.make_plan(start='start')
        self.assertEqual(len(p['route']),4)
        self.assertEqual(p['skipped'],0)

    def test_invalid_middle_rejected(self):
        es=events();es[4]['valid']=False
        with self.assertRaisesRegex(ValueError,'无效'):self.make_plan(es)

    def test_gap_and_jump_rejected(self):
        with self.assertRaisesRegex(ValueError,'间隔'):self.make_plan(max_gap=.01)
        es=events();es[4]['states']['right_arm/joint_states']['message']['position'][0]=.4
        with self.assertRaisesRegex(ValueError,'跳变'):self.make_plan(es)

    def test_other_arm_moving_rejected(self):
        es=events();es[4]['states']['left_arm/joint_states']['message']['position'][0]=.4
        with self.assertRaisesRegex(ValueError,'另一只臂'):self.make_plan(es)

    def test_changed_names_rejected(self):
        es=events();es[4]['states']['right_arm/joint_states']['message']['name'].reverse()
        with self.assertRaisesRegex(ValueError,'名称'):self.make_plan(es)

    def test_ambiguous_label_and_reverse_rejected(self):
        es=events();es[3]['label']='lift'
        with self.assertRaisesRegex(ValueError,'不唯一'):self.make_plan(es)
        with self.assertRaisesRegex(ValueError,'起点'):self.make_plan(start='point_002')

    def test_compaction_preserves_endpoint_and_jump_limit(self):
        route=[event('sample', i*.1, q) for i,q in enumerate([0., .001, .0015, .201, .202])]
        compact=compact_route(route,'right_arm',.002,.2)
        self.assertIs(compact[0],route[0])
        self.assertIs(compact[-1],route[-1])
        from replay_workpoints import distance, joints
        self.assertTrue(all(distance(joints(a,'right_arm')[0],joints(b,'right_arm')[0]) <= .2 for a,b in zip(compact,compact[1:])))
        self.assertEqual(len(compact_route(route,'right_arm',0.,.2)),len(route))
        repeated=[event('sample', i*.1, 0.) for i in range(50)]
        self.assertEqual(len(compact_route(repeated,'right_arm',.002,.2)),2)

    def run_mock(self, offset=0., success=True, approach=False, other_offset=0., update=True):
        p=self.make_plan(start='start')
        feedback=copy.deepcopy(p['route'][0])
        feedback['errors']=[]
        feedback['states']['right_arm/joint_states']['message']['position']=[offset]*7
        feedback['states']['left_arm/joint_states']['message']['position']=[other_offset]*7
        calls=[]
        class Client:
            def wait_for_service(self, **kw):return True
            def call_async(self, req):
                calls.append(req)
                if update:
                    feedback['states']['right_arm/joint_states']['message']['position']=req.joints
                return NS(done=lambda:True, result=lambda:NS(success=success))
        node=NS(cache={}, create_client=lambda *a:Client(), destroy_node=lambda:None)
        modules={'rclpy':NS(init=lambda:None,ok=lambda:True,shutdown=lambda:None,spin_once=lambda *a,**k:None),
                 'lbot_arm_interfaces':NS(), 'lbot_arm_interfaces.srv':NS(MoveJ=NS(Request=lambda:NS())),
                 'record_workpoints':NS(Recorder=lambda ns:node,snapshot=lambda *a:feedback)}
        args=NS(move_to_start=approach, approach_speed=.15, approach_accel=.15, start_tolerance=.05,max_step=.2,speed=.15,accel=.15,timeout=1.,reached_tolerance=.03)
        with patch.dict(sys.modules,modules):
            try:execute(p,args)
            except RuntimeError as e:return calls,str(e)
        return calls,None

    def test_start_mismatch_sends_nothing(self):
        calls,error=self.run_mock(offset=1.)
        self.assertEqual(calls,[])
        self.assertIn('起点',error)

    def test_approach_precedes_replay(self):
        calls,error=self.run_mock(offset=1.,approach=True)
        self.assertIsNone(error)
        self.assertEqual([c.joints[0] for c in calls],[0.,.01,.02,.025])
        self.assertEqual(calls[0].speed,.15)

    def test_failed_approach_does_not_replay(self):
        calls,error=self.run_mock(offset=1.,approach=True,success=False)
        self.assertEqual(len(calls),1)
        self.assertIn('失败',error)

    def test_other_arm_mismatch_prevents_approach(self):
        calls,error=self.run_mock(offset=1.,approach=True,other_offset=1.)
        self.assertEqual(calls,[])
        self.assertIn('另一只臂',error)

    def test_approach_flag_at_start_adds_no_motion(self):
        calls,error=self.run_mock(approach=True)
        self.assertIsNone(error)
        self.assertEqual(len(calls),3)

    def test_approach_requires_measured_arrival(self):
        calls,error=self.run_mock(offset=1.,approach=True,update=False)
        self.assertEqual(len(calls),1)
        self.assertIn('未到目标',error)

    def test_service_failure_stops_following_requests(self):
        calls,error=self.run_mock(success=False)
        self.assertEqual(len(calls),1)
        self.assertIn('失败',error)

    def test_mock_execution_uses_all_recorded_targets(self):
        calls,error=self.run_mock()
        self.assertIsNone(error)
        self.assertEqual([req.joints[0] for req in calls],[.01,.02,.025])
        self.assertTrue(all(req.block for req in calls))


if __name__=='__main__':unittest.main()
