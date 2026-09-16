#!/usr/bin/env python3
"""对一个 base_link 笛卡尔目标点做多种子逆解探针（**只调 IK 服务，不动臂**）。

用途：`move_via_ik` 报"需要换臂型（最大单关节变化 X°）"时，先离线看清这个点
到底有没有解、有几个分支、差在哪个关节、以及差额是不是 360° 角度环绕造成的假象。

用法：
    python3 tools/probe_ik_point.py --xyz 254.7,282.4,-190 --eul 40,-6.7,-95

    # 不写 --eul 就用 yaml 里 left.grasp_orientation 的姿态
    python3 tools/probe_ik_point.py --xyz 254.7,282.4,-190 --arm left

输出每个种子的解、与当前关节的逐关节差（原始 + 环绕归一化后），
并标出最大差关节。种子来自：当前关节角 + 左臂各录段的 pt0/末点。
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nut_robot import DEFAULT_CONFIG, TaskConfig  # noqa: E402
from nut_sequences import load_leg  # noqa: E402


def parse3(text, where, scale=1.0):
    nums = [float(v) for v in str(text).replace(',', ' ').split()]
    if len(nums) != 3:
        raise SystemExit(f'{where} 需要 3 个数，收到 {text!r}')
    return np.array(nums) * scale


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--arm', choices=('left', 'right'), default='left')
    p.add_argument('--xyz', required=True, help='目标位置 mm，逗号分隔（base_link 系）')
    p.add_argument('--eul', default=None,
                   help='目标姿态 度，逗号分隔（xyz 欧拉）；不给就用 yaml 的抓取姿态')
    args = p.parse_args()

    cfg = TaskConfig(args.config)
    xyz = parse3(args.xyz, '--xyz', 1e-3)
    if args.eul:
        eul = np.radians(parse3(args.eul, '--eul'))
    else:
        from nut_pick_place import resolve_grasp_euler
        from nut_robot import PoseStore
        store = PoseStore(cfg.poses_path)
        eul, src = resolve_grasp_euler(cfg, store, 'l', {})
        print(f'姿态取自 {src}')

    import rclpy
    from nut_robot import RobotClient

    rclpy.init()
    node = rclpy.create_node('probe_ik_point')
    robot = RobotClient(node, cfg.namespace, args.arm)
    if not robot.wait_state(3.0):
        print('⚠ 没收到关节/位姿反馈，只能用录段种子')

    print(f'目标 [{xyz[0]*1000:+.1f}, {xyz[1]*1000:+.1f}, {xyz[2]*1000:+.1f}] mm  '
          f'euler(deg)={np.round(np.degrees(eul), 1)}')

    seeds = []
    if robot.joints is not None:
        seeds.append(('当前关节角', np.asarray(robot.joints, float)))
    for i, spec in enumerate(cfg.left_legs if args.arm == 'left' else []):
        leg = load_leg(args.arm, spec, cfg.check_other)
        seeds.append((f'{leg.target} pt0', np.asarray(leg.joints[0], float)))
        seeds.append((f'{leg.target} 末点', np.asarray(leg.joints[-1], float)))
    if not seeds:
        raise SystemExit('没有可用种子（无反馈且无录段）')

    ref = np.asarray(cfg and (robot.joints if robot.joints is not None
                              else seeds[0][1]), float)
    names = robot.joint_names or [f'J{i}' for i in range(7)]
    print(f'参考臂型 = {seeds[0][0]}')
    print(f'{"关节":34s} {"参考":>9s} {"IK解":>9s} {"原始差":>9s} {"环绕差":>9s}')
    any_ok = False
    for name, seed in seeds:
        status, joints = robot.ik_try_full(xyz, eul, seed)
        if status != 'ok' or joints is None:
            print(f'  {name:32s} 逆解 {status}')
            continue
        any_ok = True
        q = np.asarray(joints, float)
        print(f'  ── 种子「{name}」逆解成功 ──')
        raw = np.abs(q - ref)
        wrap = np.minimum(raw, 2 * np.pi - raw)
        for i, jn in enumerate(names):
            flag = ''
            if wrap[i] < raw[i] - 1e-6:
                flag = f'   <- 360°环绕（实际只差 {np.degrees(wrap[i]):.1f}°）'
            print(f'  {jn:34s} {np.degrees(ref[i]):+8.1f}° {np.degrees(q[i]):+8.1f}° '
                  f'{np.degrees(raw[i]):8.1f}° {np.degrees(wrap[i]):8.1f}°{flag}')
        print(f'  最大原始差 {np.degrees(raw).max():.1f}°  '
              f'最大环绕差 {np.degrees(wrap).max():.1f}°  '
              f'(关节 {names[int(raw.argmax())]})')
    if not any_ok:
        print('全部种子逆解失败 —— 这个点在当前姿态下确实不可达。')

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
