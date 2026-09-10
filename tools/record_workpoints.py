#!/usr/bin/env python3
"""Read-only dual-arm waypoint and trajectory recorder. See WORKPOINTS.md."""
import argparse
import codecs
import shlex
import termios
import tty
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


class FeedTracker:
    """Track bit-identical feedback per topic.

    A live resting encoder still jitters by encoder LSBs, so exact payload equality
    while header stamps advance means the driver is republishing cached feedback
    (controller offline / arm powered down / e-stop), not a stationary arm.
    """

    @staticmethod
    def signature(key, message):
        if key.endswith('joint_states'):
            return ('j', tuple(message.get('position', ())))
        pose = message.get('pose', {})
        return ('p', tuple(pose.get('position', {}).values()),
                tuple(pose.get('orientation', {}).values()))

    def __init__(self):
        self._sig = {}
        self.last_change = {}

    def update(self, key, message, now):
        sig = self.signature(key, message)
        if self._sig.get(key) != sig:
            self.last_change[key] = now
        self._sig[key] = sig

    def frozen_age(self, key, now):
        return now - self.last_change.get(key, now)


def snapshot(cache, now, max_age, max_skew, keys=KEYS):
    """Freshness means local reception freshness, not controller health."""
    states, errors, received = {}, [], []
    for key in keys:
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
    if len(received) == len(keys) and max(received) - min(received) > max_skew:
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


def waypoint_summary(state, arm):
    """One-line end-effector xyz of the arm that will be replayed (pose already logged)."""
    try:
        msg = state['states'][f'{arm}_arm/pose_states']['message']
        pos = msg['pose']['position']
        return (f"{arm}臂末端 xyz=({pos['x']:+.3f}, {pos['y']:+.3f}, {pos['z']:+.3f}) m "
                f"[{msg['header']['frame_id']}]")
    except (KeyError, TypeError):
        return ''


class Session:
    def __init__(self, directory, metadata):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.file = (directory / 'events.jsonl').open('x', encoding='utf-8')
        self.points = 0
        self.records = 0
        self.mode = 'collecting'
        self.point_ids = []
        self.emit({'type': 'metadata', **metadata})

    def emit(self, event):
        self.file.write(json.dumps(clean_json(event), ensure_ascii=False, allow_nan=False) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())

    def mark(self, state, elapsed):
        if self.mode != 'collecting' or not state['valid']:
            return False
        point_id = f'point_{self.points+1:03d}'
        self.emit({'type': 'waypoint', 'point_id': point_id,
                   'label': point_id, 'elapsed_seconds': elapsed,
                   **copy.deepcopy(state)})
        self.point_ids.append(point_id)
        self.points += 1
        return True

    def stop(self):
        if self.mode != 'collecting' or not self.point_ids:
            return False
        self.emit({'type': 'sequence_stop', 'point_ids': self.point_ids[:]})
        self.mode = 'naming'
        return True

    def name(self, label):
        label = label.strip()
        if self.mode != 'naming' or not label:
            return None
        self.records += 1
        full_label = f'{label}_{self.records:03d}'
        self.emit({'type': 'sequence', 'sequence_id': f'sequence_{self.records:03d}',
                   'label': full_label, 'base_label': label,
                   'point_ids': self.point_ids[:]})
        self.point_ids.clear()
        self.mode = 'collecting'
        return full_label

    def close(self):
        if not self.file.closed:
            self.emit({'type': 'end', 'named_sequences': self.records,
                       'points': self.points, 'unfinished_point_ids': self.point_ids[:],
                       'unfinished_status': self.mode if self.point_ids else None})
            self.file.close()


def replay_commands(path, label, arm):
    script = Path(__file__).resolve().with_name('replay_workpoints.py')
    args = ['/usr/bin/python3', str(script), str(path.resolve()),
            '--arm', arm, '--to', label, '--move-to-start']
    return shlex.join(args), shlex.join(args + ['--execute'])


class Console:
    """Immediate q/Enter capture; restore terminal even on failure or Ctrl+C."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        return self

    def poll(self):
        if select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 4096)
            return self.decoder.decode(data) if data else '\x04'
        return ''

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)


class Recorder(Node):
    def __init__(self, namespace):
        super().__init__('workpoint_recorder')
        self.cache = {}
        self.tracker = FeedTracker()
        self.subscriptions_ = []
        for key in KEYS:
            msg_type = JointState if key.endswith('joint_states') else PoseStamped
            self.subscriptions_.append(self.create_subscription(
                msg_type, f'{namespace}/{key}',
                lambda msg, key=key: self.receive(key, msg), qos_profile_sensor_data))

    def receive(self, key, msg):
        now = time.monotonic()
        message = message_to_ordereddict(msg)
        self.cache[key] = (message, now)
        self.tracker.update(key, message, now)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='/robot1')
    parser.add_argument('--arm', choices=('left', 'right'), default='right', help='生成回放命令所选机械臂；始终记录双臂反馈')
    parser.add_argument('--max-age', type=positive, default=0.5)
    parser.add_argument('--max-skew', type=positive, default=0.15)
    parser.add_argument('--output', type=Path, default=Path('recordings'))
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error('run in an interactive terminal')
    namespace = '/' + args.namespace.strip('/') if args.namespace.strip('/') else ''
    directory = args.output.expanduser().resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session = Session(directory, {
        'schema_version': 3, 'recording_mode': 'manual_waypoints', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'namespace': namespace, 'replay_arm': args.arm,
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
        last_valid = None
        last_frozen_warn = 0.
        last_marked_q = None
        active_joint_key = f'{args.arm}_arm/joint_states'
        name_buffer = ''
        print(f'记录文件：{directory / "events.jsonl"}', flush=True)
        print(f'每按回车记录一个点；直接按 q 停止并命名；Ctrl+C 退出。回放命令默认选择{args.arm}臂。', flush=True)
        with Console() as console:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=.02)
                now = time.monotonic()
                state = snapshot(node.cache, now, args.max_age, args.max_skew)
                if session.mode == 'collecting' and state['valid'] != last_valid:
                    print('状态就绪，按回车记录点。' if state['valid'] else
                          '状态不可用：' + '; '.join(state['errors']), flush=True)
                    last_valid = state['valid']
                if session.mode == 'collecting' and state['valid'] and now - last_frozen_warn > 2.:
                    frozen = node.tracker.frozen_age(active_joint_key, now)
                    if frozen > 2.:
                        print(f'警告：{args.arm}臂关节反馈已 {frozen:.1f} s 逐位未变；时间戳照常推进通常意味着'
                              '驱动在重发缓存（控制器掉线/下电/急停）。先活动一下臂，确认下方 xyz 跟随变化后再录。', flush=True)
                        last_frozen_warn = now
                keys = console.poll()
                if '\x04' in keys:
                    break
                for key in keys:
                    if session.mode == 'naming':
                        if key in ('\n', '\r'):
                            print()
                            label = session.name(name_buffer)
                            if label:
                                print(f'已保存动作序列：{label}', flush=True)
                                preview, run = replay_commands(directory/'events.jsonl', label, args.arm)
                                print('预览命令（不运动）：\n' + preview, flush=True)
                                print('执行命令（会先接近起点，再依次运动；需确认路径无碰撞）：\n' + run, flush=True)
                                print('可继续按回车记录下一条序列，或 Ctrl+C 退出。', flush=True)
                                name_buffer = ''
                                last_marked_q = None
                            else:
                                print('名称不能为空，请重新输入：', end='', flush=True)
                        elif key in ('\x7f', '\b'):
                            if name_buffer:
                                name_buffer = name_buffer[:-1]
                                print('\b \b', end='', flush=True)
                        elif key.isprintable():
                            name_buffer += key
                            print(key, end='', flush=True)
                    elif key.lower() == 'q':
                        if session.stop():
                            name_buffer = ''
                            print('已停止记录。请输入动作名称后回车（追加本次序号）：', end='', flush=True)
                        else:
                            print('还没有点位，请先按回车记录。', flush=True)
                    elif key in ('\n', '\r'):
                        if session.mark(state, now-start):
                            summary = waypoint_summary(state, args.arm)
                            q_now = state['states'][active_joint_key]['message']['position']
                            note = ''
                            if last_marked_q is not None:
                                delta = max(abs(a-b) for a, b in zip(last_marked_q, q_now))
                                if delta == 0.:
                                    note = '  ⚠️ 与上一点关节角完全相同：若你刚移动过臂，说明反馈冻结（驱动缓存/控制器掉线），此点无效！'
                                elif delta < .002:
                                    note = f'  ⚠️ 与上一点几乎相同（Δmax={delta:.4f} rad）。'
                            last_marked_q = q_now[:]
                            print(f'已记录 {session.point_ids[-1]}，当前序列共 {len(session.point_ids)} 个点。'
                                  + (f' {summary}' if summary else '') + note, flush=True)
                        else:
                            print('拒绝记录：' + '; '.join(state['errors']), flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()
        print(f'记录结束：{session.records} 条已命名序列，{session.points} 个手动点。文件：{directory}', flush=True)


if __name__ == '__main__':
    main()
