#!/usr/bin/env python3
"""相机画面专用查看器（只看不控臂）。

订阅 Orbbec Gemini2 的彩色 + 对齐深度（可选 IR），开一个窗口双画面观看：

  +----------+----------+
  |  COLOR   |  DEPTH   |   （按 i 右栏在 深度伪彩 / IR 之间切换）
  +----------+----------+

操作：
  鼠标左键   在画面上取点，显示该像素深度（米，0 = 无效）
  s          保存快照到 tools/snapshots/<时间戳>/（彩色 + 深度伪彩 + 16位毫米深度PNG）
  i          右栏切换 深度伪彩 / IR
  h          切换深度量程档（近距 0.25~1.3m / 全量程自动）
  q / ESC    退出

用法：
  source install/setup.bash
  python3 tools/camera_view.py
  python3 tools/camera_view.py --color-only          # 只看彩色
  python3 tools/camera_view.py --color-topic /camera/color/image_raw
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def depth_to_m(depth):
    """深度图(16UC1=mm / 32FC=m) -> 米，无效为 0。"""
    scale = 1000.0 if depth.dtype == np.uint16 else 1.0
    z = depth.astype(np.float64) / scale
    z[~np.isfinite(z)] = 0.0
    return z


def colorize_depth(z, near=0.25, far=1.30, auto_range=False):
    """深度(米) -> BGR 伪彩（近红远蓝，无效黑）。"""
    import cv2
    valid = z > 0
    vis = np.zeros(z.shape, np.uint8)
    lo, hi = near, far
    if auto_range and valid.any():
        p2, p98 = np.percentile(z[valid], (2, 98))
        lo = float(np.clip(p2, 0.05, 5.0))
        hi = float(np.clip(p98, lo + 0.1, 10.0))
    if valid.any():
        n = np.clip((z - lo) / (hi - lo + 1e-9), 0, 1)
        vis = (n * 255).astype(np.uint8)
        vis[~valid] = 0
    col = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    col[~valid] = (0, 0, 0)
    cv2.putText(col, f'{lo:.2f}-{hi:.2f} m' + (' AUTO' if auto_range else ''),
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return col


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--color-topic', default='/camera/color/image_raw')
    p.add_argument('--depth-topic', default='/camera/depth/image_raw')
    p.add_argument('--ir-topic', default='/camera/ir/image_raw')
    p.add_argument('--color-only', action='store_true', help='只订阅/显示彩色画面')
    p.add_argument('--no-ir', action='store_true', help='不订阅 IR（按 i 也不切换）')
    p.add_argument('--scale', type=float, default=0.6,
                   help='单画面显示缩放（默认 0.6，1280x720 输入时双栏约 1536 宽）')
    args = p.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        import cv2
    except ImportError as e:
        print(f'[camera_view] 依赖缺失：{e}\n'
              '请先 source /opt/ros/jazzy/setup.bash 和工作区 install/setup.bash', file=sys.stderr)
        return 1

    class ViewNode(Node):
        def __init__(self):
            super().__init__('camera_view')
            self.bridge = CvBridge()
            self.color = None
            self.depth = None       # 原始深度（uint16 mm 或 float32 m）
            self.ir = None
            self._t_color, self._t_depth, self._t_ir = [], [], []
            self.fps_color = self.fps_depth = self.fps_ir = 0.0
            self.create_subscription(Image, args.color_topic,
                                     lambda m: self._img(m, 'color'), qos_profile_sensor_data)
            if not args.color_only:
                self.create_subscription(Image, args.depth_topic,
                                         lambda m: self._img(m, 'depth'), qos_profile_sensor_data)
                if not args.no_ir:
                    self.create_subscription(Image, args.ir_topic,
                                             lambda m: self._img(m, 'ir'), qos_profile_sensor_data)

        def _fps(self, buf, now):
            buf.append(now)
            while buf and now - buf[0] > 3.0:
                buf.pop(0)
            return (len(buf) - 1) / (now - buf[0] + 1e-9) if len(buf) > 1 else 0.0

        def _img(self, m, kind):
            try:
                if kind == 'color':
                    self.color = self.bridge.imgmsg_to_cv2(m, 'bgr8')
                    self.fps_color = self._fps(self._t_color, time.monotonic())
                elif kind == 'depth':
                    self.depth = self.bridge.imgmsg_to_cv2(m, 'passthrough')
                    self.fps_depth = self._fps(self._t_depth, time.monotonic())
                else:
                    self.ir = self.bridge.imgmsg_to_cv2(m, 'mono8')
                    self.fps_ir = self._fps(self._t_ir, time.monotonic())
            except Exception:
                # mono8 转换在 Y16 编码 IR 上会失败，回落 passthrough 后自行归一化
                try:
                    raw = self.bridge.imgmsg_to_cv2(m, 'passthrough')
                    if raw.dtype != np.uint8:
                        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    self.ir = raw
                    self.fps_ir = self._fps(self._t_ir, time.monotonic())
                except Exception:
                    pass

    rclpy.init()
    node = ViewNode()

    import threading
    spin_t = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_t.start()

    print('[camera_view] 等待画面 ...（q/ESC 退出，s 保存，i 切 IR，h 切深度量程）')

    show_ir = False
    auto_range = False
    click = None  # (u, v)
    PW = PH = None

    cv2.namedWindow('camera_view', cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal click
        if event == cv2.EVENT_LBUTTONDOWN and PW is not None:
            click = (min(x, PW - 1), y)

    cv2.setMouseCallback('camera_view', on_mouse)

    def waiting_panel(h=360, w=640, text='WAITING FOR IMAGE ...'):
        img = np.zeros((h, w, 3), np.uint8)
        cv2.putText(img, text, (w // 6, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (200, 200, 200), 2)
        return img

    snap_dir = Path(__file__).resolve().parent / 'snapshots'

    try:
        while True:
            color = node.color
            depth = node.depth
            ir = node.ir

            if color is None and depth is None and ir is None:
                cv2.imshow('camera_view', waiting_panel())
                if cv2.waitKey(100) & 0xFF in (ord('q'), 27):
                    break
                continue

            # ---- 左：彩色 ----
            if color is not None:
                p_color = cv2.resize(color, None, fx=args.scale, fy=args.scale)
                tag = f'COLOR {color.shape[1]}x{color.shape[0]} {node.fps_color:.1f}fps'
            else:
                p_color = waiting_panel(text='NO COLOR')
                tag = 'COLOR N/A'
            PH, PW = p_color.shape[:2]
            cv2.putText(p_color, tag, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            # ---- 右：深度伪彩 / IR ----
            if args.color_only:
                p_right = waiting_panel(PH, PW, 'COLOR-ONLY MODE')
                right_tag = ''
            elif show_ir and not args.no_ir:
                if ir is not None:
                    ir_r = cv2.resize(ir, (PW, PH))
                    p_right = cv2.cvtColor(ir_r, cv2.COLOR_GRAY2BGR)
                    right_tag = f'IR {ir.shape[1]}x{ir.shape[0]} {node.fps_ir:.1f}fps'
                else:
                    p_right = waiting_panel(PH, PW, 'NO IR')
                    right_tag = 'IR N/A'
            elif depth is not None:
                z = depth_to_m(depth)
                p_right = cv2.resize(colorize_depth(z, auto_range=auto_range), (PW, PH))
                right_tag = f'DEPTH {depth.shape[1]}x{depth.shape[0]} {node.fps_depth:.1f}fps'
            else:
                p_right = waiting_panel(PH, PW, 'NO DEPTH')
                right_tag = 'DEPTH N/A'
            if right_tag:
                cv2.putText(p_right, right_tag, (8, PH - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            # ---- 取点标注 + 深度读数 ----
            if click is not None and not args.color_only:
                u, v = click
                d_str = '--'
                if depth is not None:
                    # 显示坐标缩放过，换回原图坐标取值
                    u0 = int(round(u / args.scale))
                    v0 = int(round(v / args.scale))
                    if 0 <= v0 < depth.shape[0] and 0 <= u0 < depth.shape[1]:
                        z_m = depth_to_m(depth[v0, v0])
                        d_str = f'{z_m:.3f} m' if z_m > 0 else 'invalid'
                        # 深度画面也按缩放显示，标注坐标同比例
                        cv2.drawMarker(p_right, (u, v), (255, 255, 255),
                                       cv2.MARKER_CROSS, 20, 1)
                for pan in (p_color, p_right):
                    cv2.drawMarker(pan, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.putText(p_color, f'({u},{v}) {d_str}', (u + 10, max(v - 10, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            combo = np.hstack([p_color, p_right])
            cv2.imshow('camera_view', combo)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('i') and not args.color_only and not args.no_ir:
                show_ir = not show_ir
                print(f'[camera_view] 右栏 -> {"IR" if show_ir else "深度伪彩"}')
            elif key == ord('h') and not args.color_only:
                auto_range = not auto_range
                print(f'[camera_view] 深度量程 -> {"自动" if auto_range else "近距 0.25-1.3m"}')
            elif key == ord('s'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                out = snap_dir / ts
                out.mkdir(parents=True, exist_ok=True)
                saved = []
                if color is not None:
                    cv2.imwrite(str(out / 'color.png'), color)
                    saved.append('color.png')
                if depth is not None:
                    z = depth_to_m(depth)
                    cv2.imwrite(str(out / 'depth_color.png'),
                                colorize_depth(z, auto_range=auto_range))
                    cv2.imwrite(str(out / 'depth_mm.png'),
                                (z * 1000).astype(np.uint16))  # 16位毫米深度，无效=0
                    saved += ['depth_color.png', 'depth_mm.png']
                if ir is not None:
                    cv2.imwrite(str(out / 'ir.png'), ir)
                    saved.append('ir.png')
                print(f'[camera_view] 快照已保存: {out}  ({", ".join(saved)})')
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
