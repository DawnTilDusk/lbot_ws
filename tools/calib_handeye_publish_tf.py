#!/usr/bin/env python3
"""读取 gemini2_extrinsics.yaml 并发布静态 TF（base_link -> camera_color_optical_frame）。

用法：
    python3 tools/calib_handeye_publish_tf.py
    python3 tools/calib_handeye_publish_tf.py --extrinsics 开发资源/calibration/gemini2_extrinsics.yaml

保持该进程运行，TF 即常驻。注意：
  - 若 Orbbec 驱动以 publish_tf:=true 启动，它会发 camera_link->camera_color_optical_frame，
    与本节点的 base_link->camera_color_optical_frame 产生光学帧两个父级。二选一：
    把驱动改为 publish_tf:=false，或让本节点发布到 camera_link（见 --child-frame）。
"""
import argparse
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import TransformStamped
import yaml


def load_extrinsics(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    for key in ('parent_frame', 'child_frame', 'translation', 'rotation_xyzw'):
        if key not in data:
            raise ValueError(f'外参文件缺少字段 {key}：{path}')
    return data


class StaticTfPublisher(Node):
    def __init__(self, data, period=1.0):
        super().__init__('handeye_static_tf_publisher')
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(TransformStamped, '/tf_static', qos)
        self.msg = TransformStamped()
        self.msg.header.frame_id = data['parent_frame']
        self.msg.child_frame_id = data['child_frame']
        t, q = data['translation'], data['rotation_xyzw']
        self.msg.transform.translation.x = float(t[0])
        self.msg.transform.translation.y = float(t[1])
        self.msg.transform.translation.z = float(t[2])
        self.msg.transform.rotation.x = float(q[0])
        self.msg.transform.rotation.y = float(q[1])
        self.msg.transform.rotation.z = float(q[2])
        self.msg.transform.rotation.w = float(q[3])
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.create_timer(period, self._send)
        self._send()

    def _send(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--extrinsics', type=Path,
                   default=Path(__file__).resolve().parent.parent /
                   '开发资源' / 'calibration' / 'gemini2_extrinsics.yaml')
    p.add_argument('--child-frame', default=None,
                   help='覆盖 child_frame（例如 camera_link）；默认用外参文件中的值')
    args = p.parse_args()

    if not args.extrinsics.exists():
        print(f'未找到外参文件 {args.extrinsics}，请先运行 calib_handeye_solve.py。')
        sys.exit(1)
    data = load_extrinsics(args.extrinsics)
    if args.child_frame:
        data['child_frame'] = args.child_frame

    t, q = data['translation'], data['rotation_xyzw']
    print(f'发布静态 TF：{data["parent_frame"]} -> {data["child_frame"]}')
    print(f'  translation = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}] m')
    print(f'  rotation    = [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}] (xyzw)')
    print('\n等价命令行：')
    print(f'  ros2 run tf2_ros static_transform_publisher \\')
    print(f'    --x {t[0]:.6f} --y {t[1]:.6f} --z {t[2]:.6f} \\')
    print(f'    --qx {q[0]:.6f} --qy {q[1]:.6f} --qz {q[2]:.6f} --qw {q[3]:.6f} \\')
    print(f'    --frame-id {data["parent_frame"]} --child-frame-id {data["child_frame"]}')
    print('\n核验：ros2 run tf2_ros tf2_echo {} {}'.format(
        data['parent_frame'], data['child_frame']))
    print('Ctrl+C 退出（退出后静态 TF 消失）。\n')

    rclpy.init()
    node = StaticTfPublisher(data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
