#!/usr/bin/env python3
"""采集用于螺母标注的原始彩色 PNG；仅订阅相机，不控制机器人。

默认手动拍照：空格/s 保存，a 切换定时采集，q/ESC 退出。
--interval 3 启动后每 3 秒保存；--no-preview 可在无图形界面时定时采集。
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('必须是有限正数')
    return value


def save_frame(directory, frame, stamp, topic, index):
    """保存未叠加预览文字的原图及其来源记录。"""
    import cv2
    filename = f'{directory.name}_{index:06d}.png'
    path = directory / 'images' / filename
    if not cv2.imwrite(str(path), frame):
        raise OSError(f'图片写入失败：{path}')
    record = dict(file=f'images/{filename}', topic=topic, stamp=stamp,
                  captured_at=datetime.now().isoformat(),
                  width=frame.shape[1], height=frame.shape[0])
    with (directory / 'frames.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/home/dawntildusk/nut_vision'))
    parser.add_argument('--color-topic', default='/camera/color/image_raw')
    parser.add_argument('--interval', type=positive, help='自动拍照间隔秒数；不传则手动')
    parser.add_argument('--max-age', type=positive, default=1., help='允许的接收帧龄（秒）')
    parser.add_argument('--scale', type=positive, default=0.75, help='仅缩放预览，保存原图')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--count', type=int, default=0, help='保存指定张数后退出，0 不限')
    args = parser.parse_args()
    if args.count < 0:
        parser.error('--count 不能为负数')
    if args.no_preview and args.interval is None:
        parser.error('--no-preview 必须配合 --interval')

    import cv2
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    session = args.output.expanduser() / 'raw' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (session / 'images').mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = rclpy.create_node('capture_nut_images')
    bridge = CvBridge()
    latest = {}

    def receive(message):
        try:
            frame = bridge.imgmsg_to_cv2(message, 'bgr8')
        except Exception as exc:
            node.get_logger().warning(f'图像转换失败：{exc}')
            return
        latest.update(frame=frame, received=time.monotonic(),
                      stamp=dict(sec=message.header.stamp.sec,
                                 nanosec=message.header.stamp.nanosec))

    subscription = node.create_subscription(Image, args.color_topic, receive,
                                            qos_profile_sensor_data)
    automatic = args.interval is not None
    interval = args.interval or 3.
    next_capture = time.monotonic()
    last_saved_received = None
    last_saved_stamp = None
    count = 0
    next_warning = 0.
    print(f'保存目录：{session / "images"}', flush=True)
    print(f'订阅：{args.color_topic}；空格/s 拍照，a 切换自动（{interval:g}s），q 退出。', flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            now = time.monotonic()
            fresh = bool(latest) and now - latest['received'] <= args.max_age
            key = -1
            if not args.no_preview:
                if latest:
                    preview = cv2.resize(latest['frame'], None, fx=args.scale, fy=args.scale)
                    text = f'{"AUTO" if automatic else "MANUAL"}  saved={count}'
                    if not fresh:
                        text += '  STALE - NOT SAVING'
                    cv2.putText(preview, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 255, 255), 2)
                    cv2.imshow('nut capture: space/s save, a auto, q quit', preview)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('a'):
                automatic = not automatic
                next_capture = now + interval
                print(f'自动采集：{"开启" if automatic else "暂停"}', flush=True)
            requested = key in (ord('s'), ord(' ')) or (automatic and now >= next_capture)
            stamp = latest.get('stamp')
            duplicate = (latest.get('received') == last_saved_received
                         or (stamp and any(stamp.values()) and stamp == last_saved_stamp))
            if requested and fresh and not duplicate:
                path = save_frame(session, latest['frame'], stamp, args.color_topic, count + 1)
                count += 1
                last_saved_received = latest['received']
                last_saved_stamp = stamp.copy()
                next_capture = now + interval
                print(f'已保存 {count}：{path.name}', flush=True)
                if args.count and count >= args.count:
                    break
            if not fresh and now >= next_warning:
                print('等待新鲜彩色图；请检查相机驱动、话题和 ROS 网络配置。', flush=True)
                next_warning = now + 5.
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if not args.no_preview:
            cv2.destroyAllWindows()
        print(f'本次保存 {count} 张：{session / "images"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ImportError as exc:
        sys.exit(f'缺少依赖：{exc}。zsh 请 source /opt/ros/jazzy/setup.zsh；'
                 'bash 请 source /opt/ros/jazzy/setup.bash。使用 /usr/bin/python3。')
    except (OSError, RuntimeError) as exc:
        sys.exit(f'采图失败：{exc}')
