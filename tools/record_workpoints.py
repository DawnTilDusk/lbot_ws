#!/usr/bin/env python3
"""Read-only dual-arm waypoint and trajectory recorder. See WORKPOINTS.md."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import select
import sys
import time
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from rosidl_runtime_py.convert import message_to_ordereddict


KEYS = tuple(f'{arm}/{kind}' for arm in ('left_arm', 'right_arm')
             for kind in ('joint_states', 'pose_states'))


def snapshot(cache, now, max_age, max_skew):
    """Freshness means local reception freshness, not controller health."""
    states, errors, received = {}, [], []
    for key in KEYS:
        entry = cache.get(key)
        if entry is None:
            errors.append(f'{key}: no data')
            continue
        message, reception = entry
        age = now - reception
        states[key] = {'age_seconds': age, 'message': copy.deepcopy(message)}
        received.append(reception)
        if age > max_age:
            errors.append(f'{key}: stale ({age:.3f}s)')
        if key.endswith('joint_states'):
            if len(message['name']) != 7 or len(message['position']) != 7:
                errors.append(f'{key}: expected 7 joint names and positions')
            if len(set(message['name'])) != len(message['name']):
                errors.append(f'{key}: duplicate joint names')
            for field in ('position', 'velocity', 'effort'):
                values = message[field]
                if field != 'position' and len(values) not in (0, 7):
                    errors.append(f'{key}: invalid {field} length')
                if any(not math.isfinite(v) for v in values):
                    errors.append(f'{key}: nonfinite {field}')
        else:
            pose = message['pose']
            values = list(pose['position'].values()) + list(pose['orientation'].values())
            if not all(math.isfinite(v) for v in values):
                errors.append(f'{key}: nonfinite pose')
            elif abs(sum(v*v for v in pose['orientation'].values()) - 1) > 0.05:
                errors.append(f'{key}: invalid quaternion')
        if not message['header']['frame_id']:
            errors.append(f'{key}: empty frame_id')
    if len(received) == len(KEYS) and max(received) - min(received) > max_skew:
        errors.append('topic reception skew exceeds limit')
    return {'valid': not errors, 'errors': errors, 'states': states}


def clean_json(value):
    # Invalid numeric feedback must still be loggable as strict JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


class Session:
    def __init__(self, directory, metadata):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.file = (directory / 'events.jsonl').open('x', encoding='utf-8')
        self.samples = 0
        self.points = 0  # Number of named records.
        self.records = 0
        self.mode = 'idle'
        self.pending = None
        self.invalid_samples = 0
        self.emit({'type': 'metadata', **metadata}, durable=True)

    def emit(self, event, durable=False):
        self.file.write(json.dumps(clean_json(event), ensure_ascii=False, allow_nan=False) + '\n')
        self.file.flush()
        if durable:
            os.fsync(self.file.fileno())

    def begin(self, state, elapsed):
        if self.mode != 'idle' or not state['valid']:
            return False
        self.records += 1
        self.record_id = f'record_{self.records:03d}'
        self.start_id = self.record_id + '_start'
        self.segment_start = self.samples + 1
        self.invalid_samples = 0
        self.emit({'type': 'waypoint', 'point_id': self.start_id,
                   'label': self.start_id, 'role': 'start',
                   'record_id': self.record_id, 'elapsed_seconds': elapsed,
                   **copy.deepcopy(state)}, durable=True)
        self.mode = 'recording'
        return True

    def sample(self, state, elapsed):
        if self.mode != 'recording':
            return
        self.samples += 1
        self.invalid_samples += int(not state['valid'])
        self.emit({'type': 'sample', 'record_id': self.record_id,
                   'sample_id': self.samples, 'elapsed_seconds': elapsed, **state})

    def stop(self, state, elapsed):
        if self.mode != 'recording':
            return False
        self.pending = {'type': 'waypoint', 'point_id': self.record_id + '_end',
                        'role': 'end', 'record_id': self.record_id,
                        'elapsed_seconds': elapsed,
                        'segment': {'from_point_id': self.start_id,
                                    'first_sample_id': self.segment_start if self.samples >= self.segment_start else None,
                                    'last_sample_id': self.samples if self.samples >= self.segment_start else None,
                                    'invalid_samples': self.invalid_samples},
                        **copy.deepcopy(state)}
        # Persist the stop snapshot before naming; later feedback must not change it.
        self.emit({**self.pending, 'type': 'record_stop'}, durable=True)
        self.mode = 'naming'
        return True

    def name(self, label):
        label = label.strip()
        if self.mode != 'naming' or not label:
            return None
        full_label = f'{label}_{self.records:03d}'
        self.emit({**self.pending, 'label': full_label, 'base_label': label}, durable=True)
        self.points += 1
        self.mode = 'idle'
        self.pending = None
        return full_label

    def close(self):
        if not self.file.closed:
            self.emit({'type': 'end', 'named_records': self.points,
                       'started_records': self.records, 'samples': self.samples,
                       'unfinished_record_id': self.record_id if self.mode != 'idle' else None,
                       'unfinished_status': self.mode if self.mode != 'idle' else None}, durable=True)
            self.file.close()


class Recorder(Node):
    def __init__(self, namespace):
        super().__init__('workpoint_recorder')
        self.cache = {}
        self.subscriptions_ = []
        for key in KEYS:
            msg_type = JointState if key.endswith('joint_states') else PoseStamped
            self.subscriptions_.append(self.create_subscription(
                msg_type, f'{namespace}/{key}',
                lambda msg, key=key: self.receive(key, msg), qos_profile_sensor_data))

    def receive(self, key, msg):
        self.cache[key] = (message_to_ordereddict(msg), time.monotonic())


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='/robot1')
    parser.add_argument('--rate', type=positive, default=10.0, help='trajectory samples per second')
    parser.add_argument('--max-age', type=positive, default=0.5)
    parser.add_argument('--max-skew', type=positive, default=0.15)
    parser.add_argument('--output', type=Path, default=Path('recordings'))
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error('run in an interactive terminal')
    namespace = '/' + args.namespace.strip('/') if args.namespace.strip('/') else ''
    directory = args.output.expanduser().resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session = Session(directory, {
        'schema_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'namespace': namespace, 'rate_hz': args.rate,
        'max_age_seconds': args.max_age, 'max_reception_skew_seconds': args.max_skew,
        'topics': [f'{namespace}/{key}' for key in KEYS],
        'units': {'joint_position': 'rad', 'pose_position': 'm', 'orientation': 'quaternion_xyzw'},
        'limitations': ['Latest topic values, not hardware-synchronized snapshots.',
                       'Driver stamps cached feedback at publication: fresh reception does not prove fresh hardware feedback.',
                       'No enable/fault/temperature/hand feedback or tool configuration is published by this driver.',
                       'Waypoints are observations, not calibrated safe workspace boundaries or replay commands.']})
    node = None
    initialized = False
    try:
        rclpy.init()
        initialized = True
        node = Recorder(namespace)
        start = time.monotonic()
        next_sample = start
        last_valid = None
        print(f'记录文件：{directory / "events.jsonl"}', flush=True)
        print('仅记录，不使能/掉使能、不发送运动。回车开始 → 回车停止 → 输入名称回车保存；可连续记录，:q 退出。', flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=min(0.02, 1 / args.rate))
            now = time.monotonic()
            state = snapshot(node.cache, now, args.max_age, args.max_skew)
            if state['valid'] != last_valid:
                print('状态就绪。等待开始时按回车；记录中按回车停止。' if state['valid'] else
                      '状态不可用：' + '; '.join(state['errors']), flush=True)
                last_valid = state['valid']
            if now >= next_sample:
                session.sample(state, now - start)
                next_sample = now + 1 / args.rate
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line or line.strip() == ':q':
                    break
                text = line.strip()
                if session.mode == 'naming':
                    label = session.name(text)
                    if label:
                        print(f'已保存 {label}。按回车开始下一条记录。', flush=True)
                    else:
                        print('名称不能为空，请输入名称后回车。', flush=True)
                elif text:
                    print('请直接按回车开始/停止；停止之后再输入名称。', flush=True)
                elif session.mode == 'idle':
                    if session.begin(state, now - start):
                        next_sample = now
                        print(f'开始第 {session.records} 条记录，按回车停止。', flush=True)
                    else:
                        print('拒绝开始：' + '; '.join(state['errors']), flush=True)
                else:
                    session.stop(state, now - start)
                    print('已停止采样，请输入本条名称后回车（自动追加序号）。', flush=True)
                    if not state['valid'] or session.invalid_samples:
                        print('本段含无效反馈，仍保留原始记录，但回放会拒绝。', flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()
        print(f'记录结束：{session.points} 条已命名记录，{session.samples} 个采样。文件：{directory}', flush=True)


if __name__ == '__main__':
    main()
