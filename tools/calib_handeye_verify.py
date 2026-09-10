#!/usr/bin/env python3
"""手眼标定核验。

在线（默认）：实时检测 ChArUco 板，用两条独立路径估计标定板原点在 base_link 下的位置并比对：
  - 视觉路径：p_vis = B_T_C · (板在相机中的 tvec)
  - 运动学路径：p_kin = B_T_E · E_T_M · 0   （机械臂位姿 × 末端->板常量）
两条路径都成立且差值小（数毫米内），说明外参 B_T_C 与末端->板 E_T_M 都正确。

离线：--session 指定采样目录，对每个样本输出上述比对表。

用法：
    # 先发布静态 TF 的等价外参已在文件里（本脚本直接读文件，不依赖 /tf）
    python3 tools/calib_handeye_verify.py
    python3 tools/calib_handeye_verify.py --session recordings/calib_xxx
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from calib_common import (BOARD_CONFIG, CALIB_DIR, T_from_pose, T_from_rvec_tvec,
                          create_board, invert_T, load_board_config, make_T,
                          quat_xyzw_to_mat, read_jsonl)

EXTRINSICS = CALIB_DIR / 'gemini2_extrinsics.yaml'
GRIPPER_BOARD = CALIB_DIR / 'gemini2_gripper_to_board.yaml'


def load_transform_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    return make_T(quat_xyzw_to_mat(d['rotation_xyzw']), d['translation'])


def board_origin_via_vision(Z, CTM):
    """板原点在 base：Z @ CTM @ 0。"""
    return (Z @ CTM)[:3, 3]


def board_origin_via_kin(BTE, X):
    """板原点在 base：BTE @ X @ 0。"""
    return (BTE @ X)[:3, 3]


# ---------- 离线 ----------
def offline(session_dir):
    events = read_jsonl(Path(session_dir) / 'samples.jsonl')
    Z = load_transform_yaml(EXTRINSICS)
    X = load_transform_yaml(GRIPPER_BOARD) if GRIPPER_BOARD.exists() else None
    rows = []
    for e in events:
        if e.get('type') != 'sample':
            continue
        BTE = T_from_pose(e['ee']['position'], e['ee']['orientation_xyzw'])
        CTM = T_from_rvec_tvec(e['board']['rvec'], e['board']['tvec'])
        p_vis = board_origin_via_vision(Z, CTM)
        if X is not None:
            p_kin = board_origin_via_kin(BTE, X)
        else:
            # 无 X 文件时，用本样本反推 X 原点（应恒为 E_T_M，此处退化为展示一致性）
            X_i = invert_T(BTE) @ Z @ CTM
            p_kin = board_origin_via_kin(BTE, X_i)
        err = np.linalg.norm(p_vis - p_kin) * 1000
        rows.append((e['point_id'], p_vis, p_kin, err))
    if not rows:
        print('该目录无样本。')
        return
    print(f'{"样本":<12}{"视觉路径 base 位置 (m)":<34}{"运动学路径 (m)":<34}{"差值(mm)":>10}')
    errs = []
    for pid, pv, pk, err in rows:
        errs.append(err)
        print(f'{pid:<12}[{pv[0]: .3f} {pv[1]: .3f} {pv[2]: .3f}]      '
              f'[{pk[0]: .3f} {pk[1]: .3f} {pk[2]: .3f}]      {err:8.2f}')
    print(f'\n板原点位置差：均值 {np.mean(errs):.2f} mm，最大 {np.max(errs):.2f} mm')
    if X is None:
        print('（未找到 gemini2_gripper_to_board.yaml，运动学路径按逐样本反推，仅供参考。）')
    print('判据：差值稳定在数毫米内说明外参可用；偏大检查板尺寸、采样姿态多样性、夹持是否松动。')


# ---------- 在线 ----------
def online(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
    import cv2
    from cv2 import aruco

    for f, name in ((EXTRINSICS, '外参 gemini2_extrinsics.yaml'),):
        if not f.exists():
            print(f'缺少 {name}，请先运行 calib_handeye_solve.py。')
            sys.exit(1)
    cfg = load_board_config(args.board)
    board = create_board(cfg)
    Z = load_transform_yaml(EXTRINSICS)
    X = load_transform_yaml(GRIPPER_BOARD) if GRIPPER_BOARD.exists() else None
    if X is None:
        print('提示：未找到 gemini2_gripper_to_board.yaml，将只输出视觉路径的板原点 base 坐标。')

    class Verifier(Node):
        def __init__(self):
            super().__init__('handeye_verify')
            self.bridge = CvBridge()
            self.params = aruco.DetectorParameters_create()
            self.img = None
            self.K = None
            self.D = None
            self.BTE = None
            ns = '/' + args.namespace.strip('/')
            self.create_subscription(Image, args.image_topic,
                                     lambda m: setattr(self, 'img', self._to_bgr(m)),
                                     qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_info_topic, self._cam, qos_profile_sensor_data)
            self.create_subscription(PoseStamped, f'{ns}/{args.arm}_arm/pose_states',
                                     self._pose, qos_profile_sensor_data)

        def _to_bgr(self, m):
            try:
                return self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')
            except Exception:
                return None

        def _cam(self, m):
            self.K = np.array(m.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(m.d, dtype=np.float64)

        def _pose(self, m):
            p, q = m.pose.position, m.pose.orientation
            self.BTE = T_from_pose([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])

    rclpy.init()
    node = Verifier()
    cv2.namedWindow('handeye_verify', cv2.WINDOW_NORMAL)
    last_print = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.img is None or node.K is None:
                time.sleep(0.02)
                continue
            gray = cv2.cvtColor(node.img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, board.dictionary, parameters=node.params)
            debug = node.img.copy()
            if ids is not None:
                aruco.drawDetectedMarkers(debug, corners, ids)
                ret, cc, cid = aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if cc is not None and len(cc) >= args.min_corners:
                    ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        cc, cid, board, node.K, node.D,
                        np.zeros((3, 1)), np.zeros((3, 1)))
                    if ok:
                        CTM = T_from_rvec_tvec(rvec, tvec)
                        cv2.drawFrameAxes(debug, node.K, node.D, rvec, tvec,
                                          float(cfg['square_length_m']) * 2)
                        p_vis = board_origin_via_vision(Z, CTM)
                        msg = f'vis  base: [{p_vis[0]:.3f} {p_vis[1]:.3f} {p_vis[2]:.3f}] m'
                        cv2.putText(debug, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
                        if X is not None and node.BTE is not None:
                            p_kin = board_origin_via_kin(node.BTE, X)
                            err = np.linalg.norm(p_vis - p_kin) * 1000
                            msg2 = f'kin  base: [{p_kin[0]:.3f} {p_kin[1]:.3f} {p_kin[2]:.3f}] m'
                            msg3 = f'diff: {err:.1f} mm'
                            col = (0, 200, 0) if err < 5 else (0, 0, 255)
                            cv2.putText(debug, msg2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                            cv2.putText(debug, msg3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                            if time.monotonic() - last_print > 0.5:
                                print(f'视觉 [{p_vis[0]:.3f} {p_vis[1]:.3f} {p_vis[2]:.3f}]  '
                                      f'运动学 [{p_kin[0]:.3f} {p_kin[1]:.3f} {p_kin[2]:.3f}]  '
                                      f'差值 {err:.2f} mm', flush=True)
                                last_print = time.monotonic()
            cv2.imshow('handeye_verify', debug)
            if (cv2.waitKey(20) & 0xFF) in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--session', type=Path, default=None, help='离线核验：指定采样目录')
    p.add_argument('--namespace', default='/robot1')
    p.add_argument('--arm', choices=('left', 'right'), default='right')
    p.add_argument('--image-topic', default='/camera/color/image_raw')
    p.add_argument('--camera-info-topic', default='/camera/color/camera_info')
    p.add_argument('--board', type=Path, default=BOARD_CONFIG)
    p.add_argument('--min-corners', type=int, default=6)
    args = p.parse_args()

    if args.session:
        offline(args.session)
    else:
        online(args)


if __name__ == '__main__':
    main()
