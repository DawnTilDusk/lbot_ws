#!/usr/bin/env python3
"""Preview or explicitly execute a recorded single-arm joint path."""
import argparse
import json
import math
from pathlib import Path
import sys
import time


def distance(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def joints(event, arm):
    if not event.get('valid'):
        raise ValueError('轨迹包含无效反馈')
    try:
        m = event['states'][arm + '/joint_states']['message']
        q, names = m['position'], m['name']
        frame = m['header']['frame_id']
        if (len(q) != 7 or len(names) != 7 or len(set(names)) != 7 or
                not all(isinstance(n, str) and n for n in names) or not frame or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in q)):
            raise ValueError('无效的关节位置、名称或坐标系')
        return q, names, frame
    except (KeyError, TypeError) as exc:
        raise ValueError('记录缺少关节状态') from exc


def pick(points, selector):
    by_id = [e for e in points if e['point_id'] == selector]
    matches = by_id or [e for e in points if e['label'] == selector]
    if len(matches) != 1:
        raise ValueError(f'点位 {selector!r} 不存在或名称不唯一，请使用 point_id')
    return matches[0]


def plan(path, target, start, arm, max_step=.2, max_gap=.5, other_tolerance=.05):
    events = []
    with path.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            try:
                events.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f'第 {number} 行不是完整 JSON；先正常退出记录器') from exc
    if not events or events[0].get('type') != 'metadata' or events[0].get('schema_version') not in (1, 2):
        raise ValueError('不支持的记录格式')
    points = [e for e in events if e.get('type') == 'waypoint']
    end = pick(points, target)
    end_index = events.index(end)
    if events[0]['schema_version'] == 2:
        if end.get('role') != 'end':
            raise ValueError('目标必须是已命名记录的终点')
        paired_start = end['segment']['from_point_id']
        if start and pick(points, start)['point_id'] != paired_start:
            raise ValueError('新格式只回放同一条记录的起终点，不能跨段')
        start = paired_start
    begin = pick(points, start) if start else None
    begin_index = events.index(begin) if begin else 0
    if begin_index >= end_index:
        raise ValueError('起点必须在终点之前；不自动反向或跳转')
    segment = events[begin_index + 1:end_index]
    samples = [e for e in segment if e.get('type') == 'sample']
    skipped = 0
    if begin is None:
        while samples and not samples[0].get('valid'):
            skipped += 1
            samples.pop(0)
    if not samples and begin is None:
        raise ValueError('目标点前没有有效轨迹，无法按记录路径接近')
    route = ([begin] if begin else []) + samples + [end]
    other = 'left_arm' if arm == 'right_arm' else 'right_arm'
    first, names, frame = joints(route[0], arm)
    other_first, other_names, other_frame = joints(route[0], other)
    previous = first
    previous_time = route[0]['elapsed_seconds']
    for e in route:
        q, ns, fr = joints(e, arm)
        oq, ons, ofr = joints(e, other)
        if (ns, fr, ons, ofr) != (names, frame, other_names, other_frame):
            raise ValueError('轨迹中关节名称顺序或坐标系发生变化')
        if distance(oq, other_first) > other_tolerance:
            raise ValueError('另一只臂在记录中也明显移动；此脚本不支持双臂协同回放')
        t = e['elapsed_seconds']
        if not math.isfinite(t) or t < previous_time or t - previous_time > max_gap:
            raise ValueError('轨迹采样时间倒退或间隔过大')
        if distance(q, previous) > max_step:
            raise ValueError(f'相邻记录关节跳变 {distance(q, previous):.3f} rad 超过 {max_step:.3f} rad（{previous_time:.3f}s → {t:.3f}s），拒绝跨越')
        previous, previous_time = q, t
    return {'namespace': events[0]['namespace'], 'arm': arm, 'other': other,
            'route': route, 'names': names, 'frame': frame,
            'other_names': other_names, 'other_frame': other_frame,
            'other_start': other_first, 'skipped': skipped,
            'target': end['point_id'], 'label': end['label']}


def compact_route(route, arm, min_step, max_step):
    """Drop near-static samples; retain endpoints and bound retained point jumps."""
    if min_step == 0 or len(route) <= 2:
        return route
    kept = [route[0]]
    for index, e in enumerate(route[1:], 1):
        q = joints(e, arm)[0]
        delta = distance(joints(kept[-1], arm)[0], q)
        if delta > max_step:
            # The original path has already passed max_step validation.
            prior = route[index - 1]
            if kept[-1] is not prior:
                kept.append(prior)
        if index == len(route)-1 or distance(joints(kept[-1], arm)[0], q) >= min_step:
            kept.append(e)
    return kept


def execute(p, args):
    # ROS imports and clients exist only with --execute.
    import rclpy
    from lbot_arm_interfaces.srv import MoveJ
    from record_workpoints import Recorder, snapshot
    node = None
    initialized = False
    try:
        rclpy.init()
        initialized = True
        node = Recorder(p['namespace'])
        client = node.create_client(MoveJ, f"{p['namespace']}/{p['arm']}/move_joint")
        if not client.wait_for_service(timeout_sec=5.):
            raise RuntimeError('运动服务不可用')

        def state():
            current = snapshot(node.cache, time.monotonic(), .5, .15)
            if not current['valid']:
                raise RuntimeError('反馈无效：' + '; '.join(current['errors']))
            q, names, frame = joints(current, p['arm'])
            oq, ons, ofr = joints(current, p['other'])
            if (names, frame, ons, ofr) != (p['names'], p['frame'], p['other_names'], p['other_frame']):
                raise RuntimeError('实物关节名称/坐标系与记录不一致')
            if distance(oq, p['other_start']) > args.start_tolerance:
                raise RuntimeError('另一只臂偏离记录姿态，停止继续下发')
            return q

        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if snapshot(node.cache, time.monotonic(), .5, .15)['valid']:
                break
        current = state()
        origin = joints(p['route'][0], p['arm'])[0]
        if distance(current, origin) > args.start_tolerance:
            raise RuntimeError(f'当前姿态与轨迹起点相差 {distance(current, origin):.3f} rad，超过 {args.start_tolerance:.3f} rad；不会自动移动到起点')
        previous = origin
        for index, event in enumerate(p['route'][1:], 1):
            rclpy.spin_once(node, timeout_sec=.02)
            current = state()
            if distance(current, previous) > args.start_tolerance:
                raise RuntimeError('执行前反馈偏离上一步目标')
            q = joints(event, p['arm'])[0]
            if distance(q, current) > args.max_step + args.start_tolerance:
                raise RuntimeError('当前状态到目标的跳变过大')
            req = MoveJ.Request()
            req.joints = [float(v) for v in q]
            req.speed = args.speed
            req.acce = args.accel
            req.block = True
            print(f'下发 {index}/{len(p["route"])-1}', flush=True)
            future = client.call_async(req)
            deadline = time.monotonic() + args.timeout
            while not future.done():
                rclpy.spin_once(node, timeout_sec=.02)
                state()
                if time.monotonic() > deadline:
                    raise RuntimeError('运动服务超时；不再下发，当前运动可能仍在执行')
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError('控制器报告运动失败')
            deadline = time.monotonic() + 2.
            while True:
                rclpy.spin_once(node, timeout_sec=.02)
                if distance(state(), q) <= args.reached_tolerance:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('服务成功，但反馈未到目标容差内')
            previous = q
        print(f'已沿记录关节点到达 {p["target"]} ({p["label"]})。')
    finally:
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('参数必须为正数')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--to', required=True, help='目标 point_id 或唯一标签')
    parser.add_argument('--from', dest='start', help='起点 point_id 或唯一标签；默认首条有效采样')
    parser.add_argument('--arm', choices=('left', 'right'), required=True)
    parser.add_argument('--execute', action='store_true', help='实际发送运动；默认只预览')
    parser.add_argument('--speed', type=positive, default=.3, help='rad/s，最大 0.5')
    parser.add_argument('--accel', type=positive, default=.5, help='rad/s^2，最大 0.5')
    parser.add_argument('--start-tolerance', type=positive, default=.05)
    parser.add_argument('--reached-tolerance', type=positive, default=.03)
    parser.add_argument('--max-step', type=positive, default=.2)
    parser.add_argument('--max-gap', type=positive, default=.5)
    parser.add_argument('--timeout', type=positive, default=30.)
    parser.add_argument('--min-step', type=float, default=.002, help='合并静止附近采样的关节阈值(rad)，0 保留全部点')
    args = parser.parse_args()
    if not math.isfinite(args.min_step) or not 0 <= args.min_step <= min(.01, args.max_step):
        parser.error('--min-step 必须在 0 到 min(0.01, max-step) 之间')
    if args.speed > .5 or args.accel > .5:
        parser.error('本脚本将速度和加速度限制在 0.5 以内')
    try:
        p = plan(args.file, args.to, args.start, args.arm + '_arm', args.max_step, args.max_gap)
        original_count = len(p['route'])
        p['route'] = compact_route(p['route'], p['arm'], args.min_step, args.max_step)
        route = p['route']
        print(f'原始状态 {original_count} 个 → 下发目标 {len(route)-1} 个；速度 {args.speed} rad/s，加速度 {args.accel} rad/s²')
        print(f'目标：{p["target"]} ({p["label"]})；控制 {p["arm"]}；命名空间 {p["namespace"]}')
        print(f'路径状态数：{len(route)}；记录时长：{route[-1]["elapsed_seconds"]-route[0]["elapsed_seconds"]:.2f}s；忽略启动无效采样：{p["skipped"]}')
        print('起点关节(rad)：', joints(route[0], p['arm'])[0])
        print('终点关节(rad)：', joints(route[-1], p['arm'])[0])
        print('逐点 MoveJ 回放，不复现原始时序；没有碰撞规划。不会自动使能或自动回到起点。')
        if args.execute:
            print('执行模式。Ctrl+C/异常只停止后续下发，不保证撤销在途运动；紧急情况使用实体急停。', flush=True)
            execute(p, args)
        else:
            print('仅预览：未连接机器人、未发送指令。实际执行需添加 --execute。')
        return 0
    except KeyboardInterrupt:
        print('已中断后续下发；如有在途运动，请现场确认或使用实体急停。', file=sys.stderr)
        return 130
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'已停止：{exc}', file=sys.stderr)
        if args.execute:
            print('未自动掉使能或急停；已下发的在途运动未必停止，请现场确认。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
