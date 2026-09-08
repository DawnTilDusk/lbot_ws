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

    def test_two_enter_records_and_frozen_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp) / 'run', {'schema_version': 2, 'namespace': '/robot1'})
            good = snapshot(cache(), 10.1, .5, .15)
            bad = snapshot({}, 10.1, .5, .15)
            session.sample(good, 0.)
            self.assertEqual(session.samples, 0)
            self.assertFalse(session.begin(bad, .1))
            self.assertTrue(session.begin(good, .2))
            session.sample(good, .3)
            session.stop(good, .4)
            good['states'][KEYS[0]]['message']['position'][0] = .01
            session.sample(good, .5)
            self.assertEqual(session.samples, 1)
            self.assertIsNone(session.name(' '))
            self.assertEqual(session.name('lift'), 'lift_001')
            session.sample(good, .6)
            self.assertTrue(session.begin(good, 20.))
            session.sample(good, 20.1)
            session.stop(good, 20.2)
            self.assertEqual(session.name('lift'), 'lift_002')
            session.close()
            session.close()
            from replay_workpoints import plan
            path = session.directory/'events.jsonl'
            points = [e for e in map(json.loads, path.read_text().splitlines()) if e['type']=='waypoint']
            self.assertEqual(points[1]['states'][KEYS[0]]['message']['position'][0], 0.)
            self.assertEqual(points[1]['elapsed_seconds'], .4)
            self.assertEqual(len(plan(path, 'lift_002', None, 'right_arm')['route']), 3)
            self.assertEqual(plan(path, 'lift_002', None, 'right_arm')['route'][0]['elapsed_seconds'], 20.)
            with self.assertRaises(ValueError):
                plan(path, 'lift_002', 'record_001_start', 'right_arm')

    def test_stop_on_bad_data_and_unfinished_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp) / 'run', {'schema_version': 2, 'namespace': '/robot1'})
            good = snapshot(cache(), 10.1, .5, .15)
            bad = snapshot({}, 10.1, .5, .15)
            session.begin(good, .1)
            session.sample(bad, .2)
            self.assertTrue(session.stop(bad, .3))
            self.assertEqual(session.mode, 'naming')
            session.name('bad')
            session.begin(good, .4)
            session.sample(good, .5)
            session.close()
            path=session.directory/'events.jsonl'
            es=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(es[-1]['unfinished_record_id'], 'record_002')
            from replay_workpoints import plan
            with self.assertRaises(ValueError):plan(path, 'bad_001', None, 'right_arm')

    def test_nonfinite_is_saved_as_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(Path(tmp) / 'run', {})
            session.begin(snapshot(cache(), 10.1, .5, .15), 0.)
            c = cache()
            c[KEYS[0]][0]['position'][0] = float('nan')
            session.sample(snapshot(c, 10.1, .5, .15), 0.)
            session.close()
            raw = (session.directory / 'events.jsonl').read_text()
            self.assertNotIn('NaN', raw)
            events = [json.loads(line) for line in raw.splitlines()]
            self.assertIsNone(next(e for e in events if e['type']=='sample')['states'][KEYS[0]]['message']['position'][0])


if __name__ == '__main__':
    unittest.main()
