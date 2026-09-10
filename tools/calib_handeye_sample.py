#!/usr/bin/env python3
"""Eye-to-Hand 手眼标定采样节点（只读订阅，不发任何运动指令）。

同步记录：
  - 机械臂末端位姿 B_T_E：/robot1/<arm>/pose_states（geometry_msgs/PoseStamped）
  - 标定板在相机中位姿 C_T_M：对 /camera/color/image_raw 做 ChArUco 位姿估计
采集足够样本后，用 calib_handeye_solve.py 求解外参。

用法：
    # 终端1：启动机器人驱动 + Orbbec 相机
    ros2 launch orbbec_camera gemini2.launch.py depth_registration:=true publish_tf:=true
    # 终端2：采样
    python3 tools/calib_handeye_sample.py
    python3 tools/calib_handeye_sample.py --arm right --save-images

交互：预览窗口聚焦后，【空格/回车】记录当前样本，【q】结束。
采样时机械臂务必停稳（脚本有静止检测），标定板完全可见、多角度分布，建议 15~25 个姿态。
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from calib_common import (BOARD_CONFIG, RECORDINGS_DIR, create_board,
                          load_board_config, write_jsonl_event, clean_json,
                          quat_xyzw_to_euler_deg)


class CalibSampler(Node):
    def __init__(self, args, board, cfg):
        super().__init__('handeye_calib_sampler')
        self.args = args
        self.board = board
        self.cfg = cfg
        self.bridge = CvBridge()
        from cv2 import aruco
        self.aruco = aruco
        self.detector_params = aruco.DetectorParameters_create()

        # 缓存最新数据 + 接收时刻
        self.img_msg = None
        self.img_bgr = None
        self.img_recv = None
        self.cam_info = None
        self.cam_recv = None
        self.pose_recv = None
        self.pose_history = deque()   # (t, pos(3,), quat(4,))

        ns = '/' + args.namespace.strip('/')
        pose_topic = f'{ns}/{args.arm}_arm/pose_states'
        self.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._on_caminfo, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, qos_profile_sensor_data)
        self.get_logger().info(f'订阅图像 {args.image_topic}')
        self.get_logger().info(f'订阅内参 {args.camera_info_topic}')
        self.get_logger().info(f'订阅末端位姿 {pose_topic}')

    # ---- 回调 ----
    def _on_image(self, msg):
        try:
            self.img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.img_msg = msg
            self.img_recv = time.monotonic()
        except Exception as e:  # CvBridgeError 等
            self.get_logger().warn(f'图像转换失败：{e}', throttle_duration_sec=2)

    def _on_caminfo(self, msg):
        self.cam_info = msg
        self.cam_recv = time.monotonic()

    def _on_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        self.pose_recv = time.monotonic()
        self.pose_frame = msg.header.frame_id
        self.latest_pose = (np.array([p.x, p.y, p.z]),
                            np.array([q.x, q.y, q.z, q.w]))
        self.pose_history.append((self.pose_recv, self.latest_pose[0], self.latest_pose[1]))
        cutoff = self.pose_recv - self.args.still_window
        while self.pose_history and self.pose_history[0][0] < cutoff:
            self.pose_history.popleft()

    # ---- 检测 ----
    def detect_board(self):
        """返回 (ok, rvec, tvec, corners_count, debug_img, errors, code)。
        code 为 ASCII 短码，供画面叠加（cv2.putText 无法渲染中文）。"""
        import cv2
        errors = []
        if self.img_bgr is None:
            errors.append('未收到图像')
            return False, None, None, 0, None, errors, 'NO IMAGE'
        if self.cam_info is None:
            errors.append('未收到相机内参 camera_info')
            return False, None, None, 0, self.img_bgr.copy(), errors, 'NO CAMERA INFO'

        gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        K = np.array(self.cam_info.k, dtype=np.float64).reshape(3, 3)
        D = np.array(self.cam_info.d, dtype=np.float64)

        corners, ids, _ = self.aruco.detectMarkers(gray, self.board.dictionary,
                                                   parameters=self.detector_params)
        debug = self.img_bgr.copy()
        if ids is None or len(ids) == 0:
            errors.append('未检测到 ArUco 码（板是否在视野内？）')
            return False, None, None, 0, debug, errors, 'NO ARUCO MARKER (board in view?)'
        self.aruco.drawDetectedMarkers(debug, corners, ids)
        n_markers = len(ids)

        retval, charuco_corners, charuco_ids = self.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board)
        n_corners = 0 if charuco_corners is None else len(charuco_corners)
        if n_corners < self.args.min_corners:
            errors.append(f'ChArUco 角点不足（{n_corners}/{self.args.min_corners}）')
            code = f'CORNERS {n_corners}/{self.args.min_corners} (markers {n_markers}) - too far/tilted?'
            return False, None, None, n_corners, debug, errors, code

        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        ok = self.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board, K, D, rvec, tvec)
        if not ok:
            errors.append('板位姿估计失败')
            return False, None, None, n_corners, debug, errors, 'POSE ESTIMATE FAIL'

        axis_len = float(self.cfg['square_length_m']) * 2.0
        cv2.drawFrameAxes(debug, K, D, rvec, tvec, axis_len)
        code = f'OK corners {n_corners} dist {np.linalg.norm(tvec):.2f}m'
        return True, np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3), n_corners, debug, errors, code

    # ---- 质量门 ----
    def check_stationary(self):
        """返回 (ok, errors, pos_dev_m, rot_dev_deg)。"""
        errors = []
        now = time.monotonic()
        if self.pose_recv is None:
            return False, ['未收到机械臂位姿'], None, None
        age = now - self.pose_recv
        if age > self.args.max_age:
            errors.append(f'机械臂位姿过期（{age:.2f}s）')
        if len(self.pose_history) < 5:
            errors.append('机械臂位姿样本不足，等待稳定')
            return False, errors, None, None
        span = self.pose_history[-1][0] - self.pose_history[0][0]
        if span < self.args.still_window * 0.8:
            errors.append(f'位姿观测时长不足（{span:.2f}s）')
            return False, errors, None, None

        poss = np.array([p for _, p, _ in self.pose_history])
        quats = np.array([q for _, _, q in self.pose_history])
        pos_mean = poss.mean(axis=0)
        pos_dev = np.linalg.norm(poss - pos_mean, axis=1).max()
        # 平均四元数（带符号对齐）后求最大偏差角
        q_sign = quats * np.sign(quats @ quats[0])[:, None]
        q_mean = q_sign.mean(axis=0)
        q_mean /= np.linalg.norm(q_mean)
        dots = np.clip(np.abs(quats @ q_mean), -1, 1)
        rot_dev = np.degrees(2 * np.arccos(dots)).max()
        if pos_dev > self.args.still_pos:
            errors.append(f'机械臂未停稳（平移抖动 {pos_dev*1000:.2f}mm）')
        if rot_dev > self.args.still_rot:
            errors.append(f'机械臂未停稳（姿态抖动 {rot_dev:.3f}°）')
        return (len(errors) == 0), errors, pos_dev, rot_dev


def arm_status_ascii(node, ok, pos_dev, rot_dev):
    """机械臂静止判定的 ASCII 原因码（cv2.putText 画不了中文）。"""
    now = time.monotonic()
    a = node.args
    if node.pose_recv is None:
        return 'NO ARM POSE DATA'
    age = now - node.pose_recv
    parts = []
    if age > a.max_age:
        parts.append(f'POSE STALE {age:.1f}s')
    if len(node.pose_history) < 5:
        parts.append('ARM SETTLING...')
        return ' / '.join(parts)
    span = node.pose_history[-1][0] - node.pose_history[0][0]
    if span < a.still_window * 0.8:
        parts.append('ARM SETTLING...')
    if pos_dev is not None and pos_dev > a.still_pos:
        parts.append(f'POS JITTER {pos_dev*1000:.1f}mm (need <{a.still_pos*1000:.1f})')
    if rot_dev is not None and rot_dev > a.still_rot:
        parts.append(f'ROT JITTER {rot_dev:.2f}deg (need <{a.still_rot:.1f})')
    return ' / '.join(parts) if parts else 'STILL'


def positive(v):
    v = float(v)
    if v <= 0:
        raise argparse.ArgumentTypeError('必须为正数')
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--namespace', default='/robot1')
    p.add_argument('--arm', choices=('left', 'right'), default='right')
    p.add_argument('--image-topic', default='/camera/color/image_raw')
    p.add_argument('--camera-info-topic', default='/camera/color/camera_info')
    p.add_argument('--board', type=Path, default=BOARD_CONFIG, help='ChArUco 板配置 YAML')
    p.add_argument('--min-corners', type=int, default=6, help='采样所需最少 ChArUco 角点数')
    p.add_argument('--max-age', type=positive, default=0.5, help='数据新鲜度阈值 s')
    p.add_argument('--still-window', type=positive, default=0.5, help='静止判定窗口 s')
    p.add_argument('--still-pos', type=positive, default=0.001, help='静止平移阈值 m（1mm，需高于反馈噪声）')
    p.add_argument('--still-rot', type=positive, default=0.3, help='静止姿态阈值 度（需高于反馈噪声~0.1°）')
    p.add_argument('--save-images', action='store_true', help='同时保存每帧图像（便于复核）')
    p.add_argument('--output', type=Path, default=RECORDINGS_DIR)
    args = p.parse_args()

    if not args.board.exists():
        print(f'未找到板配置 {args.board}，请先运行 calib_charuco_board.py 生成并实测回填尺寸。')
        sys.exit(1)
    cfg = load_board_config(args.board)
    board = create_board(cfg)
    print(f'已加载标定板：{cfg["dictionary"]}，方格 {cfg["squares_x"]}x{cfg["squares_y"]}，'
          f'方格 {cfg["square_length_m"]*1000:.2f}mm，marker {cfg["marker_length_m"]*1000:.2f}mm')
    print('请确认 square_length_m 已按实测回填，否则外参尺度会错！')

    session_dir = args.output.expanduser().resolve() / ('calib_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    img_dir = session_dir / 'images'
    session_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        img_dir.mkdir(exist_ok=True)
    fh = (session_dir / 'samples.jsonl').open('w', encoding='utf-8')

    rclpy.init()
    node = CalibSampler(args, board, cfg)
    import cv2
    cv2.namedWindow('handeye_calib', cv2.WINDOW_NORMAL)

    count = 0
    meta_written = False
    start = time.monotonic()
    last_status = ''
    try:
        print('\n窗口聚焦后：空格/回车=记录样本， q=结束。建议 15~25 个姿态、多角度、板铺满视野。\n')
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            ok_board, rvec, tvec, n_corners, debug, board_errors, bcode = node.detect_board()
            if debug is None:
                debug = np.zeros((480, 640, 3), dtype=np.uint8)
            ok_still, still_errors, pos_dev, rot_dev = node.check_stationary()

            errors = board_errors + still_errors
            ready = ok_board and ok_still
            acode = arm_status_ascii(node, ok_still, pos_dev, rot_dev)

            # 叠加状态（ASCII，cv2.putText 无法渲染中文）
            lines = [f'samples: {count}',
                     f'board: {bcode}',
                     f'arm:   {acode}']
            if pos_dev is not None:
                lines.append(f'jitter: {pos_dev*1000:.2f} mm, {rot_dev:.3f} deg')
            lines.append('READY - press SPACE' if ready else 'not ready')
            color = (0, 200, 0) if ready else (0, 0, 255)
            for i, line in enumerate(lines):
                cv2.putText(debug, line, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)

            # 当前末端位姿（实时反馈；移动机械臂时这些数值应随之变化，不变说明反馈冻结）
            ly = 28 + len(lines) * 26 + 8
            latest = getattr(node, 'latest_pose', None)
            if latest is not None:
                p_pos, p_quat = latest
                eul = quat_xyzw_to_euler_deg(p_quat)
                pose_lines = [
                    f'EE pos(mm): {p_pos[0]*1000:8.1f} {p_pos[1]*1000:8.1f} {p_pos[2]*1000:8.1f}',
                    f'EE rpy(deg): {eul[0]:7.1f} {eul[1]:7.1f} {eul[2]:7.1f}',
                ]
                age = time.monotonic() - node.pose_recv if node.pose_recv else float('nan')
                pose_lines.append(f'pose age: {age:.2f} s')
                for j, pl in enumerate(pose_lines):
                    cv2.putText(debug, pl, (10, ly + j * 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(debug, 'EE pose: no data', (10, ly), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('handeye_calib', debug)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):
                break
            if key in (32, 13, 10):  # space / enter
                if not ready:
                    print('拒绝记录：' + '; '.join(errors), flush=True)
                    continue
                if not meta_written:
                    ci = node.cam_info
                    write_jsonl_event(fh, {
                        'type': 'metadata', 'schema_version': 1, 'mode': 'eye_to_hand',
                        'created_utc': datetime.now(timezone.utc).isoformat(),
                        'namespace': '/' + args.namespace.strip('/'), 'arm': args.arm,
                        'base_frame': getattr(node, 'pose_frame', None),
                        'camera_frame': ci.header.frame_id if ci else 'camera_color_optical_frame',
                        'topics': {'image': args.image_topic,
                                   'camera_info': args.camera_info_topic,
                                   'pose': f'/{args.namespace.strip("/")}/{args.arm}_arm/pose_states'},
                        'board': dict(cfg),
                        'camera_info': None if ci is None else {
                            'frame_id': ci.header.frame_id, 'width': ci.width, 'height': ci.height,
                            'distortion_model': ci.distortion_model,
                            'K': list(ci.k), 'D': list(ci.d), 'P': list(ci.p)},
                        'units': {'pose_position': 'm', 'orientation': 'quaternion_xyzw',
                                  'board_rvec': 'rad', 'board_tvec': 'm'},
                        'limitations': ['图像与机械臂位姿非硬件同步；靠停稳后采样消除运动误差。',
                                        '板尺寸必须为打印实测值。'],
                    })
                    meta_written = True

                pos, quat = node.latest_pose
                point_id = f'point_{count+1:03d}'
                img_rel = None
                if args.save_images:
                    img_rel = f'images/{point_id}.jpg'
                    cv2.imwrite(str(session_dir / img_rel), node.img_bgr)
                write_jsonl_event(fh, {
                    'type': 'sample', 'point_id': point_id,
                    'elapsed_seconds': time.monotonic() - start,
                    'ee': {'position': [float(v) for v in pos],
                           'orientation_xyzw': [float(v) for v in quat],
                           'frame_id': getattr(node, 'pose_frame', None)},
                    'board': {'rvec': [float(v) for v in rvec],
                              'tvec': [float(v) for v in tvec],
                              'corners': int(n_corners),
                              'frame_id': node.cam_info.header.frame_id},
                    'image': img_rel,
                })
                count += 1
                print(f'已记录 {point_id}（角点 {n_corners}，板距 {np.linalg.norm(tvec):.3f}m），共 {count} 个。', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f'\n采样结束：{count} 个样本，目录 {session_dir}')
    if count >= 5:
        print(f'求解外参：python3 tools/calib_handeye_solve.py {session_dir}')
    elif count > 0:
        print('样本偏少，建议 15 个以上再求解以获得稳定外参。')
    else:
        print('未记录任何样本。')


if __name__ == '__main__':
    main()
