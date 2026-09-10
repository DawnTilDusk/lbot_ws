import json
from pathlib import Path
import tempfile
import unittest
from record_workpoints import KEYS, Session, snapshot, FeedTracker


def cache():
    data = {}
    for key in KEYS:
        msg = {'header': {'frame_id': 'base_link', 'stamp': {'sec': 1, 'nanosec': 0}}}
        if key.endswith('joint_states'):
            msg.update(name=[f'j{i}' for i in range(7)], position=[0.] * 7,
                       velocity=[0.] * 7, effort=[0.] * 7)
        else:
            msg['pose'] = {'position': dict(x=0., y=0., z=0.),
                           'orientation': dict(x=0., y=0., z=0., w=1.)}
        data[key] = (msg, 10.)
    return data


class FeedTrackerTests(unittest.TestCase):
    def test_frozen_vs_live_joints(self):
        tracker = FeedTracker()
        key = [k for k in KEYS if k.endswith('joint_states')][0]
        tracker.update(key, {'position': [1.] * 7}, 10.0)
        # Identical payload at a later time: feedback frozen although time advanced.
        tracker.update(key, {'position': [1.] * 7}, 11.0)
        self.assertEqual(tracker.frozen_age(key, 13.0), 3.0)
        # A single encoder-LSB change means live feedback.
        p = [1.] * 7
        p[3] += .0001
        tracker.update(key, {'position': p}, 13.5)
        self.assertEqual(tracker.frozen_age(key, 13.5), 0.0)

    def test_frozen_vs_live_pose(self):
        tracker = FeedTracker()
        key = [k for k in KEYS if k.endswith('pose_states')][0]
        msg = {'pose': {'position': dict(x=1., y=2., z=3.),
                        'orientation': dict(x=0., y=0., z=0., w=1.)}}
        tracker.update(key, msg, 5.0)
        tracker.update(key, msg, 6.0)
        self.assertEqual(tracker.frozen_age(key, 9.0), 4.0)
        msg2 = {'pose': {'position': dict(x=1.001, y=2., z=3.),
                         'orientation': dict(x=0., y=0., z=0., w=1.)}}
        tracker.update(key, msg2, 9.5)
        self.assertEqual(tracker.frozen_age(key, 9.5), 0.0)


class RecordingTests(unittest.TestCase):
    def test_valid_snapshot_is_copy(self):
        c = cache()
        s = snapshot(c, 10.1, .5, .15)
        self.assertTrue(s['valid'])
        c[KEYS[0]][0]['position'][0] = 2.
        self.assertEqual(s['states'][KEYS[0]]['message']['position'][0], 0.)

    def test_missing_and_stale(self):
        self.assertFalse(snapshot({}, 10.1, .5, .15)['valid'])
        self.assertFalse(snapshot(cache(), 11., .5, .15)['valid'])

    def test_skew(self):
        c = cache()
        c[KEYS[0]] = (c[KEYS[0]][0], 9.8)
        self.assertIn('topic reception skew exceeds limit', snapshot(c, 10.1, .5, .15)['errors'])

    def test_invalid_values(self):
        for value in (float('nan'), float('inf')):
            c = cache()
            c[KEYS[0]][0]['position'][0] = value
            self.assertFalse(snapshot(c, 10.1, .5, .15)['valid'])
        c = cache()
        c[KEYS[1]][0]['pose']['orientation']['w'] = 0.
        self.assertFalse(snapshot(c, 10.1, .5, .15)['valid'])

    def test_snapshot_subset_ignores_unwatched_topics(self):
        c = cache()
        del c[KEYS[1]]  # a pose_states topic missing
        joint_keys = [k for k in KEYS if k.endswith('joint_states')]
        self.assertTrue(snapshot(c, 10.1, .5, .15, joint_keys)['valid'])
        self.assertFalse(snapshot(c, 10.1, .5, .15)['valid'])

    def test_wrong_joint_shape(self):
        c = cache()
        c[KEYS[0]][0]['position'] = [0.]
        self.assertFalse(snapshot(c, 10.1, .5, .15)['valid'])

    def test_manual_sequence_and_replay_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp) / 'run', {'schema_version': 3, 'namespace': '/robot1'})
            good = snapshot(cache(), 10.1, .5, .15)
            bad = snapshot({}, 10.1, .5, .15)
            self.assertFalse(session.stop())
            self.assertFalse(session.mark(bad, 0.))
            self.assertTrue(session.mark(good, 1.))
            good['states']['right_arm/joint_states']['message']['position'][0] = .8
            self.assertTrue(session.mark(good, 40.))
            session.stop()
            self.assertFalse(session.mark(good, 41.))
            good['states']['right_arm/joint_states']['message']['position'][0] = 1.2
            self.assertIsNone(session.name(' '))
            self.assertEqual(session.name('lift'), 'lift_001')
            self.assertTrue(session.mark(good, 100.))
            session.stop()
            self.assertEqual(session.name('lift'), 'lift_002')
            session.close()
            session.close()
            path=session.directory/'events.jsonl'
            es=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sum(e['type']=='waypoint' for e in es),3)
            self.assertFalse(any(e['type']=='sample' for e in es))
            from replay_workpoints import plan, joints
            p=plan(path,'lift_001',None,'right_arm')
            self.assertTrue(p['manual'])
            self.assertEqual([joints(e,'right_arm')[0][0] for e in p['route']],[0.,.8])
            self.assertEqual(len(plan(path,'lift_002',None,'right_arm')['route']),1)

    def test_quit_before_naming_preserves_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            session=Session(Path(tmp)/'run',{'schema_version':3,'namespace':'/robot1'})
            session.mark(snapshot(cache(),10.1,.5,.15),1.)
            session.stop()
            session.close()
            es=[json.loads(l) for l in (session.directory/'events.jsonl').read_text().splitlines()]
            self.assertFalse(any(e['type']=='sequence' for e in es))
            self.assertEqual(es[-1]['unfinished_point_ids'],['point_001'])

    def test_commands_quote_labels_and_paths(self):
        from record_workpoints import replay_commands
        import shlex
        path=Path('/tmp/a b/记录/events.jsonl')
        label="lift ' $(touch /tmp/unsafe)_001"
        preview,run=replay_commands(path,label,'left')
        self.assertEqual(shlex.split(preview)[2],str(path))
        self.assertEqual(shlex.split(preview)[6],label)
        self.assertNotIn('--execute',shlex.split(preview))
        self.assertEqual(shlex.split(run)[-1],'--execute')

    def test_console_immediate_q_and_restoration(self):
        import os, pty, termios
        from unittest.mock import patch
        from record_workpoints import Console
        master,slave=pty.openpty()
        stream=os.fdopen(os.dup(slave),'r')
        before=termios.tcgetattr(slave)
        try:
            with patch('sys.stdin',stream):
                with Console() as console:
                    os.write(master,b'q')
                    import select
                    select.select([slave],[],[],1.)
                    self.assertEqual(console.poll(),'q')
            self.assertEqual(termios.tcgetattr(slave),before)
        finally:
            stream.close();os.close(master);os.close(slave)


if __name__ == '__main__':
    unittest.main()
