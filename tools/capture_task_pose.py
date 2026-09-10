#!/usr/bin/env python3
"""把机械臂【当前实到位姿】存入 task_poses.yaml，供抓放任务按名取用。

双臂交接版只需要采左臂一个姿态（先把左臂摇到抓螺母时要保持的姿态）：
  python3 tools/capture_task_pose.py --arm left --name left_grasp_init
  python3 tools/capture_task_pose.py --list
  python3 tools/capture_task_pose.py --arm left --name left_grasp_init --force  # 覆盖
  python3 tools/capture_task_pose.py --delete left_grasp_init

注意：left_grasp_init 只取三个旋转角（视觉抓取全程姿态保持不变），xyz 由视觉反算覆盖；
所有固定段（离场/中央放置/中央重抓/入盒）都由预录关节序列承担，不再采位姿。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nut_robot import DEFAULT_CONFIG, PoseStore, TaskConfig, TaskError


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--arm', choices=('left', 'right'), default=None,
                   help='默认取 nut_task.yaml 里的 arm')
    p.add_argument('--name', help='位姿名，如 ready / grasp_init / place_l')
    p.add_argument('--list', action='store_true', help='列出已保存的全部位姿后退出')
    p.add_argument('--delete', metavar='NAME', help='删除指定位姿后退出')
    p.add_argument('--force', action='store_true', help='允许覆盖同名位姿')
    p.add_argument('--timeout', type=float, default=3.0)
    args = p.parse_args()

    try:
        cfg = TaskConfig(args.config, arm_override=args.arm)
    except TaskError as exc:
        sys.exit(f'配置错误：{exc}')

    store = PoseStore(cfg.poses_path)

    if args.list:
        for arm in ('left', 'right'):
            names = store.names(arm)
            mark = ' *' if arm == cfg.arm_for_capture else ''
            print(f'[{arm}{mark}] {" ".join(names) if names else "(空)"}')
        print(f'\n文件：{cfg.poses_path}（* = 配置当前选用臂 {cfg.arm_for_capture}）')
        return 0

    if args.delete:
        try:
            store.delete(cfg.arm_for_capture, args.delete)
            store.save()
            print(f'已删除 {cfg.arm_for_capture} 臂位姿 {args.delete!r}。')
        except TaskError as exc:
            sys.exit(str(exc))
        return 0

    if not args.name:
        p.error('采集需要 --name（或用 --list / --delete）')

    if args.name in store.names(cfg.arm_for_capture) and not args.force:
        sys.exit(f'{cfg.arm_for_capture} 臂已存在同名位姿 {args.name!r}；覆盖加 --force')

    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import PoseStamped
        from scipy.spatial.transform import Rotation as Rot
    except ImportError as exc:
        sys.exit(f'缺少 ROS 依赖（{exc.name}）。请先 source /opt/ros/jazzy/setup.bash 和 install/setup.bash')

    rclpy.init()
    node = rclpy.create_node('capture_task_pose')
    state = {'j': None, 'p': None, 'q': None, 'frame': None}
    base = f'{cfg.namespace}/{cfg.arm_for_capture}_arm'
    node.create_subscription(JointState, f'{base}/joint_states',
                             lambda m: state.update(j=[float(x) for x in m.position]),
                             qos_profile_sensor_data)
    node.create_subscription(PoseStamped, f'{base}/pose_states',
                             lambda m: state.update(
                                 p=[m.pose.position.x, m.pose.position.y, m.pose.position.z],
                                 q=[m.pose.orientation.x, m.pose.orientation.y,
                                    m.pose.orientation.z, m.pose.orientation.w],
                                 frame=m.header.frame_id),
                             qos_profile_sensor_data)

    t0 = time.monotonic()
    while time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        if state['j'] is not None and state['p'] is not None:
            break
    if state['j'] is None or state['p'] is None:
        sys.exit(f'{args.timeout}s 内未收到 {cfg.arm_for_capture}_arm 的 joint_states/pose_states，驱动是否启动？')
    if len(state['j']) != 7:
        sys.exit(f'joint_states 长度 {len(state["j"])}，预期 7')

    eul = Rot.from_quat(state['q']).as_euler('xyz', degrees=True)
    store.put(cfg.arm_for_capture, args.name, state['p'], eul, state['j'],
              frame_id=state['frame'] or 'base_link')
    store.save()

    print(f'已保存 {cfg.arm_for_capture} 臂位姿 {args.name!r} -> {cfg.poses_path}')
    print(f'  position_m : [{", ".join(f"{v:+.4f}" for v in state["p"])}]')
    print(f'  euler_deg  : [{", ".join(f"{v:+.2f}" for v in eul)}]')
    print('提示：grasp_init 只取三个旋转角，抓取时保持该姿态；place_* 教格内释放点。')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
