import json
from pathlib import Path
import tempfile
import unittest
from record_workpoints import KEYS, Session, snapshot


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
