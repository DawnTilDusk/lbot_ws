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


def pose_xyz(event, arm):
    """End-effector position (x, y, z in metres); None if this record lacks pose feedback."""
    try:
        pos = event['states'][arm + '/pose_states']['message']['pose']['position']
        xyz = (pos['x'], pos['y'], pos['z'])
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xyz):
            return None
        return xyz
    except (KeyError, TypeError):
        return None


def fmt_xyz(xyz):
    return f'({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m' if xyz else '本记录无末端坐标'


def pick(points, selector):
    by_id = [e for e in points if e['point_id'] == selector]
    matches = by_id or [e for e in points if e['label'] == selector]
    if len(matches) != 1:
        raise ValueError(f'点位 {selector!r} 不存在或名称不唯一，请使用 point_id')
    return matches[0]


def plan(path, target, start, arm, max_step=.2, max_gap=.5, other_tolerance=.05, check_other=True):
    events = []
    with path.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            try:
                events.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f'第 {number} 行不是完整 JSON；先正常退出记录器') from exc
    if not events or events[0].get('type') != 'metadata' or events[0].get('schema_version') not in (1, 2, 3):
        raise ValueError('不支持的记录格式')
    manual = events[0]['schema_version'] == 3
    if manual:
        sequences = [e for e in events if e.get('type') == 'sequence']
        matches = [e for e in sequences if e.get('sequence_id') == target]
        matches = matches or [e for e in sequences if e.get('label') == target]
        if not matches:
            base = [e for e in sequences if e.get('base_label') == target]
            if len(base) == 1:
                matches = base
        if len(matches) != 1:
            available = '；'.join(f"{e.get('label')} ({e.get('sequence_id')})" for e in sequences) or '无已命名序列'
            raise ValueError(f'动作序列 {target!r} 不存在或名称不唯一。可用序列：{available}')
        sequence = matches[0]
        if start:
            raise ValueError('手动序列从其第一个点开始，不使用 --from')
        points = [e for e in events if e.get('type') == 'waypoint']
        ids = sequence['point_ids']
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('动作序列为空或含重复点编号')
        route = [pick(points, point_id) for point_id in ids]
        end = {**route[-1], 'point_id': sequence['sequence_id'], 'label': sequence['label']}
        skipped = 0
    else:
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
        if check_other and distance(oq, other_first) > other_tolerance:
            raise ValueError('另一只臂在记录中也明显移动；此脚本不支持双臂协同回放（确认仅回放单臂动作可加 --ignore-other-arm）')
        t = e['elapsed_seconds']
        if not math.isfinite(t) or t < previous_time or (not manual and t - previous_time > max_gap):
            raise ValueError('轨迹采样时间倒退或间隔过大')
        if not manual and distance(q, previous) > max_step:
            raise ValueError(f'相邻记录关节跳变 {distance(q, previous):.3f} rad 超过 {max_step:.3f} rad（{previous_time:.3f}s → {t:.3f}s），拒绝跨越')
        previous, previous_time = q, t
    return {'manual': manual, 'namespace': events[0]['namespace'], 'arm': arm, 'other': other,
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
    try:
        import rclpy
        from lbot_arm_interfaces.srv import MoveJ
        from record_workpoints import Recorder, snapshot
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            '缺少 ROS 依赖（%s）。请先在本终端 source 环境后再运行：\n'
            '  source /opt/ros/jazzy/setup.bash\n'
            '  source install/setup.bash   （在 lbot_ws 目录下；若报 not found 需先 colcon build）'
            % exc.name) from exc
    node = None
    initialized = False
    try:
        rclpy.init()
        initialized = True
        node = Recorder(p['namespace'])
        client = node.create_client(MoveJ, f"{p['namespace']}/{p['arm']}/move_joint")
        if not client.wait_for_service(timeout_sec=5.):
            raise RuntimeError('运动服务不可用')

        ignore_other = getattr(args, 'ignore_other_arm', False)
        # 执行只需关节话题；pose_states 运动时发布更慢，纳入新鲜度检查会误报 skew。
        watched = [p['arm'] + '/joint_states']
        if not ignore_other:
            watched.append(p['other'] + '/joint_states')
        if ignore_other:
            print('警告：已跳过另一只臂姿态校验；请自行确认双臂全程无干涉、另一只臂已固定。', flush=True)

        def fresh_snapshot():
            return snapshot(node.cache, time.monotonic(), args.max_age, args.max_skew, watched)

        def state():
            current = fresh_snapshot()
            if not current['valid']:
                raise RuntimeError('反馈无效：' + '; '.join(current['errors']))
            q, names, frame = joints(current, p['arm'])
            if (names, frame) != (p['names'], p['frame']):
                raise RuntimeError('实物关节名称/坐标系与记录不一致')
            if not ignore_other:
                oq, ons, ofr = joints(current, p['other'])
                if (ons, ofr) != (p['other_names'], p['other_frame']):
                    raise RuntimeError('实物关节名称/坐标系与记录不一致')
                if distance(oq, p['other_start']) > args.start_tolerance:
                    raise RuntimeError('另一只臂偏离记录姿态，停止继续下发（确认安全可加 --ignore-other-arm 跳过）')
            return q

        def move(q, speed, accel, tolerance, max_attempts=3):
            # 伺服低速时稳态可能停在容差外几毫弧度量级；同一目标重发一次通常即可补齐
            # （等价于手动再跑一遍脚本），所以这里自动重试，而不是直接放弃整条轨迹。
            q_before = state()
            joint_delta = distance(q_before, q)
            q_now = q_before
            for attempt in range(1, max_attempts + 1):
                # 重发时按当前残差重新计算等待时间。
                attempt_delta = joint_delta if attempt == 1 else distance(q_now, q)
                budget = min(args.timeout, max(2., attempt_delta / speed + 2.))
                req = MoveJ.Request()
                req.joints = [float(v) for v in q]
                req.speed = speed
                req.acce = accel
                req.block = True
                future = client.call_async(req)
                deadline = time.monotonic() + args.timeout
                while not future.done():
                    rclpy.spin_once(node, timeout_sec=.02)
                    state()
                    if time.monotonic() > deadline:
                        raise RuntimeError('运动服务超时；不再下发，当前运动可能仍在执行')
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(f'控制器报告运动失败：请检查{p["arm"]}是否已使能、急停是否解除、有无在途运动或故障，'
                                       f'然后重试（使能：ros2 service call {p["namespace"]}/{p["arm"]}/set_enable '
                                       'lbot_arm_interfaces/srv/SetEnable "{enable: true}"）')
                # block=true 的返回时刻可能早于物理到位（低速大角度尤其明显），
                # 按关节行程/速度给到位等待，并留 2s 加减速余量。
                q_at_return = state()
                wait_end = time.monotonic() + budget
                while True:
                    rclpy.spin_once(node, timeout_sec=.02)
                    q_now = state()
                    residual = distance(q_now, q)
                    if residual <= tolerance:
                        return
                    if time.monotonic() > wait_end:
                        break
                per_joint = [round(abs(a-b), 3) for a, b in zip(q_now, q)]
                # 只有“差一点点”才重发同一目标；偏差很大说明没动或被挡住，重试无意义。
                if attempt < max_attempts and residual <= max(.1, 3 * tolerance):
                    print(f'到位残差 {residual:.3f} rad 略超容差 {tolerance} rad，重发同一目标补点 '
                          f'({attempt + 1}/{max_attempts})：{per_joint}', flush=True)
                    continue
                moved_total = distance(q_before, q_now)
                crept = distance(q_at_return, q_now)
                if moved_total < .02:
                    reason = '臂基本没有移动，检查使能/抱闸/是否被挡住或控制器有在途运动'
                elif crept > tolerance:
                    reason = (f'服务返回后臂仍在运动（等待窗口内又走了 {crept:.3f} rad），'
                              'block 返回早于实际到位，可增大 --timeout 或降低速度后重试')
                else:
                    reason = ('臂已走完绝大部分但伺服稳态停在容差外，重发同一目标仍未补齐；'
                              '可适当放宽 --reached-tolerance，或检查该姿态是否受力/接近奇异')
                raise RuntimeError(f'服务成功，但 {budget:.1f}s 内反馈未到目标容差：最大残差 {residual:.3f} rad，'
                                   f'各关节残差 {per_joint}（容差 {tolerance} rad）。{reason}')

        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if fresh_snapshot()['valid']:
                break
        current = state()
        origin = joints(p['route'][0], p['arm'])[0]
        if distance(current, origin) > args.start_tolerance:
            if not args.move_to_start:
                raise RuntimeError(f'当前姿态与轨迹起点相差 {distance(current, origin):.3f} rad，超过 {args.start_tolerance:.3f} rad；需要 --move-to-start 才会移动到起点')
            print(f'先移动到起点：最大关节差 {distance(current, origin):.3f} rad，速度 {args.approach_speed} rad/s。', flush=True)
            move(origin, args.approach_speed, args.approach_accel,
                 min(args.start_tolerance, args.reached_tolerance))
            print('已确认到达起点，开始轨迹回放。', flush=True)
        previous = origin
        for index, event in enumerate(p['route'][1:], 1):
            rclpy.spin_once(node, timeout_sec=.02)
            current = state()
            if distance(current, previous) > args.start_tolerance:
                raise RuntimeError('执行前反馈偏离上一步目标')
            q = joints(event, p['arm'])[0]
            if not p.get('manual') and distance(q, current) > args.max_step + args.start_tolerance:
                raise RuntimeError('当前状态到目标的跳变过大')
            print(f'下发 {index}/{len(p["route"])-1}', flush=True)
            move(q, args.speed, args.accel, args.reached_tolerance)
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
    parser.add_argument('--move-to-start', action='store_true', help='执行时先以 MoveJ 低速接近起点；需确认额外路径无碰撞')
    parser.add_argument('--ignore-other-arm', action='store_true',
                        help='跳过另一只臂的姿态校验（默认校验，防止双臂干涉）；仅在已确认另一只臂全程无碰撞风险时使用')
    parser.add_argument('--approach-speed', type=positive, default=.15, help='接近起点速度 rad/s，最大 0.3')
    parser.add_argument('--approach-accel', type=positive, default=.15, help='接近起点加速度 rad/s²，最大 0.3')
    parser.add_argument('--speed', type=positive, default=.3, help='rad/s，最大 0.5')
    parser.add_argument('--accel', type=positive, default=.5, help='rad/s^2，最大 0.5')
    parser.add_argument('--start-tolerance', type=positive, default=.05)
    parser.add_argument('--reached-tolerance', type=positive, default=.03)
    parser.add_argument('--max-step', type=positive, default=.2)
    parser.add_argument('--max-gap', type=positive, default=.5)
    parser.add_argument('--timeout', type=positive, default=30.)
    parser.add_argument('--max-age', type=positive, default=.5, help='执行时关节反馈允许的最大接收延迟(s)')
    parser.add_argument('--max-skew', type=positive, default=.15, help='执行时左右关节话题接收时间允许的最大差(s)')
    parser.add_argument('--min-step', type=float, default=.002, help='合并静止附近采样的关节阈值(rad)，0 保留全部点')
    args = parser.parse_args()
    if not math.isfinite(args.min_step) or not 0 <= args.min_step <= min(.01, args.max_step):
        parser.error('--min-step 必须在 0 到 min(0.01, max-step) 之间')
    if args.approach_speed > .3 or args.approach_accel > .3:
        parser.error('接近起点速度和加速度限制在 0.3 以内')
    if args.speed > .5 or args.accel > .5:
        parser.error('本脚本将速度和加速度限制在 0.5 以内')
    try:
        p = plan(args.file, args.to, args.start, args.arm + '_arm', args.max_step, args.max_gap,
                 check_other=not args.ignore_other_arm)
        original_count = len(p['route'])
        if not p.get('manual'):
            p['route'] = compact_route(p['route'], p['arm'], args.min_step, args.max_step)
        route = p['route']
        print(f'原始状态 {original_count} 个 → 下发目标 {len(route)-1} 个；速度 {args.speed} rad/s，加速度 {args.accel} rad/s²')
        print(f'目标：{p["target"]} ({p["label"]})；控制 {p["arm"]}；命名空间 {p["namespace"]}')
        print(f'路径状态数：{len(route)}；记录时长：{route[-1]["elapsed_seconds"]-route[0]["elapsed_seconds"]:.2f}s；忽略启动无效采样：{p["skipped"]}')
        if p.get('manual'):
            print('手动点序列：保留每个点，点间由 MoveJ 插值；未记录点间路径，不套用连续采样间隔/跳变阈值。')
        print('起点关节(rad)：', joints(route[0], p['arm'])[0])
        print('终点关节(rad)：', joints(route[-1], p['arm'])[0])
        print('起点末端坐标：', fmt_xyz(pose_xyz(route[0], p['arm'])))
        print('终点末端坐标：', fmt_xyz(pose_xyz(route[-1], p['arm'])))
        print('逐点 MoveJ 回放，不复现原始时序；没有碰撞规划。不会自动使能。')
        print('起点修正：启用；执行时从实时姿态以 MoveJ 接近，非记录路径，没有碰撞规划。' if args.move_to_start else '起点修正：关闭；起点不匹配时拒绝运动。')
        print('另一只臂校验：已跳过（--ignore-other-arm），需自行确认双臂无干涉。' if args.ignore_other_arm
              else f'另一只臂校验：启用；执行时{p["other"]}须停在记录姿态，偏差超过 {args.start_tolerance} rad 即中止。')
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
