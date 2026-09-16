#!/usr/bin/env python3
"""等步长笛卡尔平移工具：读取机械臂当前末端位姿，MoveJP 到「当前位姿 + delta」。

用法：
    python3 tools/step_move.py --arm left --dx 0.02            # 向 +X 挪 2cm
    python3 tools/step_move.py --arm left --dy -0.01 --dz -0.01
    python3 tools/step_move.py --arm right --dx 0.01 --speed 0.2

- 单位：米；姿态保持不变（用当前实时位姿的欧拉角）。
- 不会自动上使能；臂必须在使能状态（先 ros2 service call .../set_enable）。
- base_link 系：+X 机器人前方，+Z 向上；Y 轴方向若与直觉相反下次反号即可。
"""
import argparse
import time

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as Rot
from lbot_arm_interfaces.srv import MoveJP


def quat_to_euler_xyz(q):
    return Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('left', 'right'), required=True)
    p.add_argument('--namespace', default='/robot1')
    p.add_argument('--dx', type=float, default=0.0, help='X 方向偏移（米）')
    p.add_argument('--dy', type=float, default=0.0, help='Y 方向偏移（米）')
    p.add_argument('--dz', type=float, default=0.0, help='Z 方向偏移（米）')
    p.add_argument('--speed', type=float, default=0.3)
    p.add_argument('--acce', type=float, default=0.3)
    p.add_argument('--timeout', type=float, default=120.0)
    args = p.parse_args()
    delta = np.array([args.dx, args.dy, args.dz], float)
    if not np.any(delta):
        print('dx/dy/dz 全为 0，无移动。用法：--dx/--dy/--dz 给米制偏移，如 --dx 0.02')
        return 1

    rclpy.init()
    node = Node('step_move')
    ns = '/' + args.namespace.strip('/')
    cli = node.create_client(MoveJP, f'{ns}/{args.arm}_arm/move_pose')
    pos_q = []

    def on_pose(msg):
        pos_q.append(msg)

    node.create_subscription(PoseStamped, f'{ns}/{args.arm}_arm/pose_states', on_pose, 10)

    print(f'等待服务 {cli.srv_name} ...')
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15:
        rclpy.spin_once(node, timeout_sec=0.1)
        if cli.service_is_ready() and pos_q:
            break
    if not cli.service_is_ready():
        print('move_pose 服务 15s 内不可用，驱动是否已启动？')
        return 1
    if not pos_q:
        print('没有收到 pose_states 反馈，请确认驱动在运行。')
        return 1

    cur = pos_q[-1].pose
    eul = quat_to_euler_xyz(cur.orientation)
    new_pos = np.array([cur.position.x, cur.position.y, cur.position.z]) + delta

    req = MoveJP.Request()
    req.position.x, req.position.y, req.position.z = [float(v) for v in new_pos]
    req.euler.x, req.euler.y, req.euler.z = [float(v) for v in eul]
    req.speed = float(args.speed)
    req.acce = float(args.acce)
    req.block = True

    print(f'当前末端 (m): [{cur.position.x:.4f} {cur.position.y:.4f} {cur.position.z:.4f}]')
    print(f'偏移     (m): [{args.dx:+.4f} {args.dy:+.4f} {args.dz:+.4f}]')
    print(f'目标     (m): [{new_pos[0]:.4f} {new_pos[1]:.4f} {new_pos[2]:.4f}]')
    print('正在发送 MoveJP ...', flush=True)

    fut = cli.call_async(req)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        if fut.done():
            try:
                res = fut.result()
            except Exception as exc:  # noqa
                print(f'服务调用异常：{exc}')
                return 1
            if res.success:
                print('✅ 移动成功。')
                return 0
            print('❌ 移动失败（success=false）——目标可能不可达，或臂未使能/有在途运动。')
            return 1
    print('⏱ 等待响应超时；请现场确认臂状态（可用 pgrep -f lib/lbot_driver 查驱动）。')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())