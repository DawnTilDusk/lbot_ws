#!/usr/bin/env python3
"""Eye-to-Hand 手眼标定【半自动】采样节点 —— 会【真实驱动机械臂运动】。

流程：上使能 → 依次移动到 auto_waypoints.yaml 里的目标点（MoveJP，低速）→
停稳且板被检测到 → 【回车】记录当前场景 → 自动前往下一目标点。

与 calib_handeye_sample.py 写完全相同的 samples.jsonl，可直接用 calib_handeye_solve.py 求解。

⚠️ 安全：
  - 本脚本会让机械臂自主运动！运行前请清空机械臂活动范围内的人和物，手持急停。
  - 标定板必须【刚性固定在法兰/夹爪上】（不能用手拿着或胶带软固定），否则外参差。
  - 速度默认很低（speed=acce=0.5，示例程序标准值为 1），可用 --speed/--acce 调。
  - 第一个目标点相对当前位姿可能有较大移动，注意观察；异常立即按急停或 q 退出。

用法：
    # 终端1：驱动 + 相机已启动
    python3 tools/calib_handeye_auto.py                       # 用默认目标点文件
    python3 tools/calib_handeye_auto.py --speed 0.3 --acce 0.3
    python3 tools/calib_handeye_auto.py --no-ik               # 跳过逆解可达性预检
交互（预览窗口聚焦，默认“就绪后自动连采”）：
  到位后检到板且停稳，持续 --auto-delay（默认0.6s）即【自动记录并去下一个位姿】，无需按键；
  空格/回车=【直接记录并立刻去下一个】（当前帧有板就强制采、无视停稳门；没板则直接跳过）；
  s=跳过本帧；q=结束；
  到位后 --ready-timeout（默认12s）仍未就绪：窗口橙色 TIMEOUT，空格/s=跳到下一个位姿，r=再等。
  想回到“完全手动、不自动采”：加 --auto-delay 0。
  注意：空格强制采会跳过停稳判定，请在臂到位后再按，运动中按会录到运动位姿。
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from calib_common import (BOARD_CONFIG, CALIB_DIR, RECORDINGS_DIR, create_board,
                          load_board_config, write_jsonl_event, quat_xyzw_to_euler_deg)

WAYPOINTS_PATH = CALIB_DIR / 'auto_waypoints.yaml'


def call_service(node, client, request, timeout, spin_hz=20):
    """异步调用 service，期间持续 spin 节点（保持图像/位姿刷新）。超时返回 None。"""
    fut = client.call_async(request)
    t0 = time.monotonic()
    dt = 1.0 / spin_hz
    while time.monotonic() - t0 < timeout:
        rclpy_spin_once(node, timeout_sec=dt)
        if fut.done():
            try:
                return fut.result()
            except Exception as e:  # noqa
                node.get_logger().warn(f'service 调用异常：{e}')
                return None
    return None


def rclpy_spin_once(node, timeout_sec=0.05):
    import rclpy
    rclpy.spin_once(node, timeout_sec=timeout_sec)


class AutoCalibNode:
    def __init__(self, args, board, cfg, waypoints):
        import rclpy
        from rclpy.node import Node
        # 复用手动采样节点的订阅/检测/静止判定逻辑
        from calib_handeye_sample import CalibSampler, arm_status_ascii
        self.arm_status_ascii = arm_status_ascii
        from lbot_arm_interfaces.srv import MoveJP, SetEnable, InverseKinematics

        self.rclpy = rclpy
        self.args = args
        self.board = board
        self.cfg = cfg
        self.waypoints = waypoints
        self.MoveJP = MoveJP
        self.SetEnable = SetEnable
        self.InverseKinematics = InverseKinematics

        self.node = CalibSampler(args, board, cfg)
        n = self.node
        ns = '/' + args.namespace.strip('/')
        arm = args.arm
        self.move_cli = n.create_client(MoveJP, f'{ns}/{arm}_arm/move_pose')
        self.enable_cli = n.create_client(SetEnable, f'{ns}/{arm}_arm/set_enable')
        self.ik_cli = n.create_client(InverseKinematics, f'{ns}/{arm}_arm/inverse_kinematics')

    def log(self, msg):
        self.node.get_logger().info(msg)

    def wait_services(self):
        for name, cli in (('set_enable', self.enable_cli), ('move_pose', self.move_cli)):
            self.log(f'等待服务 {cli.srv_name} ...')
            ok = False
            t0 = time.monotonic()
            while time.monotonic() - t0 < 15:
                rclpy_spin_once(self.node, 0.1)
                if cli.service_is_ready():
                    ok = True
                    break
            if not ok:
                raise SystemExit(f'服务 {name} ({cli.srv_name}) 15s 内不可用，驱动是否已启动？')
        if not self.args.no_ik:
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5:
                rclpy_spin_once(self.node, 0.1)
                if self.ik_cli.service_is_ready():
                    break
            self.ik_ok = self.ik_cli.service_is_ready()
            if not self.ik_ok:
                self.log('逆解服务不可用，跳过可达性预检（目标点本身来自已到达位姿，风险低）。')
        else:
            self.ik_ok = False

    def set_enable(self, enable):
        req = self.SetEnable.Request()
        req.enable = bool(enable)
        res = call_service(self.node, self.enable_cli, req, timeout=10)
        return bool(res is not None and getattr(res, 'success', False))

    def check_reach(self, wp):
        if not self.ik_ok:
            return True
        req = self.InverseKinematics.Request()
        req.joints = []  # 空 → 驱动用当前关节角作为种子
        req.position.x, req.position.y, req.position.z = wp['position']
        req.euler.x, req.euler.y, req.euler.z = wp['euler']
        res = call_service(self.node, self.ik_cli, req, timeout=8)
        if res is None:
            self.log('逆解服务超时，跳过预检继续。')
            return True
        if not res.success:
            return False
        return True

    def move_to(self, wp):
        req = self.MoveJP.Request()
        req.position.x, req.position.y, req.position.z = wp['position']
        req.euler.x, req.euler.y, req.euler.z = wp['euler']
        req.speed = float(self.args.speed)
        req.acce = float(self.args.acce)
        req.block = True
        res = call_service(self.node, self.move_cli, req, timeout=self.args.move_timeout)
        if res is None:
            self.log('move_pose 超时！请检查机械臂是否卡住/急停。')
            return False
        if not res.success:
            self.log('move_pose 返回 success=false（目标可能不可达或被保护停止）。')
            return False
        return True


def overlay(base_img, lines, color):
    import cv2
    for i, line in enumerate(lines):
        cv2.putText(base_img, line, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)


def run():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--namespace', default='/robot1')
    p.add_argument('--arm', choices=('left', 'right'), default='right')
    p.add_argument('--image-topic', default='/camera/color/image_raw')
    p.add_argument('--camera-info-topic', default='/camera/color/camera_info')
    p.add_argument('--board', type=Path, default=BOARD_CONFIG)
    p.add_argument('--waypoints', type=Path, default=WAYPOINTS_PATH)
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--acce', type=float, default=0.5)
    p.add_argument('--move-timeout', type=float, default=60.0)
    p.add_argument('--ready-timeout', type=float, default=12.0,
                   help='到位后等待“停稳+检到板”的秒数；超时后回车/s 跳到下一个位姿，r 继续等')
    p.add_argument('--auto-delay', type=float, default=0.6,
                   help='就绪后自动采集前需持续稳定的秒数（防检测闪烁误采）；0=关闭自动采集，改回纯回车触发')
    p.add_argument('--no-ik', action='store_true', help='跳过逆解可达性预检')
    p.add_argument('--min-corners', type=int, default=6)
    p.add_argument('--max-age', type=float, default=0.5)
    p.add_argument('--still-window', type=float, default=0.5)
    p.add_argument('--still-pos', type=float, default=0.001, help='静止平移阈值 m（默认1mm，需高于反馈噪声）')
    p.add_argument('--still-rot', type=float, default=0.3, help='静止姿态阈值 度（默认0.3，反馈噪声约0.1°）')
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--output', type=Path, default=RECORDINGS_DIR)
    p.add_argument('--yes', action='store_true', help='跳过启动确认（脚本化用，谨慎）')
    args = p.parse_args()

    if not args.board.exists():
        sys.exit('未找到板配置，请先 calib_charuco_board.py 生成并实测回填尺寸。')
    if not args.waypoints.exists():
        sys.exit(f'未找到目标点文件 {args.waypoints}，请先运行 calib_gen_waypoints.py。')
    cfg = load_board_config(args.board)
    board = create_board(cfg)
    wdata = yaml.safe_load(open(args.waypoints, encoding='utf-8'))
    waypoints = wdata['waypoints']
    if wdata.get('speed') and args.speed == 0.5:
        args.speed = wdata['speed']
    if wdata.get('acce') and args.acce == 0.5:
        args.acce = wdata['acce']

    print(f'标定板：{cfg["dictionary"]} {cfg["squares_x"]}x{cfg["squares_y"]}，'
          f'方格 {cfg["square_length_m"]*1000:.2f}mm')
    print(f'目标点 {len(waypoints)} 个，speed={args.speed} acce={args.acce}（示例程序标准值=1）')
    print('\n⚠️  机械臂将自主运动。请确认：活动范围内无人无物、急停在手、标定板刚性固定在法兰上。')
    for i, w in enumerate(waypoints, 1):
        print(f'  {w["name"]:<7} pos={np.round(w["position"],3)}  '
              f'rpy(deg)={np.round(np.degrees(w["euler"]),1)}')
    if not args.yes:
        ans = input('\n确认上使能并开始自动标定？输入 yes 回车：').strip().lower()
        if ans != 'yes':
            print('已取消。')
            return

    import rclpy
    import cv2
    rclpy.init()
    app = AutoCalibNode(args, board, cfg, waypoints)
    app.wait_services()

    print('上使能中...（电机会抱闸，请勿触碰机械臂）')
    if not app.set_enable(True):
        sys.exit('上使能失败，退出。')
    print('已上使能。')

    session_dir = args.output.expanduser().resolve() / ('calib_auto_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    img_dir = session_dir / 'images'
    session_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        img_dir.mkdir(exist_ok=True)
    fh = (session_dir / 'samples.jsonl').open('w', encoding='utf-8')

    cv2.namedWindow('handeye_calib_auto', cv2.WINDOW_NORMAL)
    count = 0
    meta_written = False
    start = time.monotonic()
    stop = False

    def write_meta(n):
        ci = n.cam_info
        write_jsonl_event(fh, {
            'type': 'metadata', 'schema_version': 1, 'mode': 'eye_to_hand', 'automated': True,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'namespace': '/' + args.namespace.strip('/'), 'arm': args.arm,
            'base_frame': getattr(n, 'pose_frame', None),
            'camera_frame': ci.header.frame_id if ci else 'camera_color_optical_frame',
            'waypoints_file': str(args.waypoints),
            'speed': args.speed, 'acce': args.acce,
            'topics': {'image': args.image_topic, 'camera_info': args.camera_info_topic,
                       'pose': f'/{args.namespace.strip("/")}/{args.arm}_arm/pose_states'},
            'board': dict(cfg),
            'camera_info': None if ci is None else {
                'frame_id': ci.header.frame_id, 'width': ci.width, 'height': ci.height,
                'distortion_model': ci.distortion_model,
                'K': list(ci.k), 'D': list(ci.d), 'P': list(ci.p)},
            'units': {'pose_position': 'm', 'orientation': 'quaternion_xyzw',
                      'board_rvec': 'rad', 'board_tvec': 'm', 'waypoint_euler': 'rad_intrinsic_xyz'},
            'limitations': ['自动 MoveJP 到位后停稳采样；板必须刚性固定在法兰。'],
        })

    try:
        for idx, wp in enumerate(waypoints):
            if stop:
                break
            print(f'\n[{idx+1}/{len(waypoints)}] 前往 {wp["name"]} ...', flush=True)
            if not app.check_reach(wp):
                print(f'  逆解判定 {wp["name"]} 不可达，跳过。', flush=True)
                continue
            if not app.move_to(wp):
                print(f'  {wp["name"]} 运动失败，跳过。', flush=True)
                continue
            expect = wp.get('expected', 'visible')
            print(f'  到位，等待停稳 + 检测标定板 ...（预期 {expect}，{args.ready_timeout:.0f}s 超时可跳过）', flush=True)

            # 到位后：就绪(检到板+停稳)持续 auto_delay 秒 -> 自动记录并去下一个；
            # 回车=立即记录；s=跳过；r=超时后继续等；q=退出。
            deadline = time.monotonic() + args.ready_timeout
            outcome = None  # 'record' | 'skip' | 'quit'
            ready_since = None

            def do_record(n_corners, tvec):
                nonlocal count, meta_written
                if not meta_written:
                    write_meta(app.node)
                    meta_written = True
                pos, quat = app.node.latest_pose
                point_id = f'point_{count+1:03d}'
                img_rel = None
                if args.save_images:
                    img_rel = f'images/{point_id}.jpg'
                    cv2.imwrite(str(session_dir / img_rel), app.node.img_bgr)
                write_jsonl_event(fh, {
                    'type': 'sample', 'point_id': point_id, 'waypoint': wp['name'],
                    'elapsed_seconds': time.monotonic() - start,
                    'ee': {'position': [float(v) for v in pos],
                           'orientation_xyzw': [float(v) for v in quat],
                           'frame_id': getattr(app.node, 'pose_frame', None)},
                    'board': {'rvec': [float(v) for v in rvec_cur],
                              'tvec': [float(v) for v in tvec],
                              'corners': int(n_corners),
                              'frame_id': app.node.cam_info.header.frame_id},
                    'image': img_rel,
                })
                count += 1
                print(f'  已记录 {point_id}（角点 {n_corners}，板距 {np.linalg.norm(tvec):.3f}m），前往下一帧。', flush=True)
                return 'record'

            while outcome is None:
                rclpy.spin_once(app.node, timeout_sec=0.02)
                ok_board, rvec_cur, tvec_cur, n_corners, debug, berr, bcode = app.node.detect_board()
                if debug is None:
                    debug = np.zeros((480, 640, 3), np.uint8)
                ok_still, serr, pos_dev, rot_dev = app.node.check_stationary()
                acode = app.arm_status_ascii(app.node, ok_still, pos_dev, rot_dev)
                ready = ok_board and ok_still
                now = time.monotonic()
                remain = deadline - now
                timed_out = remain <= 0

                # 就绪持续计时（用于自动采集）
                if ready:
                    if ready_since is None:
                        ready_since = now
                    stable_for = now - ready_since
                else:
                    ready_since = None
                    stable_for = 0.0
                auto_on = args.auto_delay > 0
                auto_fire = auto_on and ready and stable_for >= args.auto_delay

                lines = [f'waypoint: {wp["name"]}  ({idx+1}/{len(waypoints)})  expect:{expect}',
                         f'samples: {count}',
                         f'board: {bcode}',
                         f'arm:   {acode}']
                if pos_dev is not None:
                    lines.append(f'jitter: {pos_dev*1000:.2f} mm, {rot_dev:.3f} deg')
                latest = getattr(app.node, 'latest_pose', None)
                if latest is not None:
                    pp, pq = latest
                    eu = quat_xyzw_to_euler_deg(pq)
                    lines += [
                        f'EE pos(mm): {pp[0]*1000:8.1f} {pp[1]*1000:8.1f} {pp[2]*1000:8.1f}',
                        f'EE rpy(deg): {eu[0]:7.1f} {eu[1]:7.1f} {eu[2]:7.1f}']
                if ready and auto_on:
                    cd = max(0.0, args.auto_delay - stable_for)
                    lines.append(f'READY: AUTO in {cd:4.1f}s | SPACE=capture&next  s=skip  q=quit')
                    col = (0, 200, 0)
                elif ready:
                    lines.append('READY: SPACE=record&next   s=skip   q=quit')
                    col = (0, 200, 0)
                elif timed_out:
                    lines.append('TIMEOUT: SPACE/s=next pose   r=wait more   q=quit')
                    col = (0, 165, 255)
                else:
                    lines.append(f'waiting... {remain:4.1f}s   SPACE/s=next   q=quit')
                    col = (0, 0, 255)
                overlay(debug, lines, col)
                cv2.imshow('handeye_calib_auto', debug)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord('q'), 27):
                    outcome = 'quit'
                    break
                if key in (ord('s'), ord('S')):
                    outcome = 'skip'
                    print('  手动跳过本帧。', flush=True)
                    break
                if key in (ord('r'), ord('R')):
                    deadline = time.monotonic() + args.ready_timeout
                    ready_since = None
                    print(f'  继续等待 {args.ready_timeout:.0f}s ...', flush=True)
                    continue
                if key in (32, 13, 10):  # space / enter：直接记录并下一个（有板就强制采，无板则跳过）
                    if ok_board:
                        print('  手动强制采集（无视停稳门）->', flush=True)
                        outcome = do_record(n_corners, tvec_cur)
                    else:
                        outcome = 'skip'
                        print('  当前帧未检测到标定板，空格直接跳到下一个位姿。', flush=True)
                elif auto_fire:
                    print(f'  检测成功且已稳定 {args.auto_delay:.1f}s，自动采集 ->', flush=True)
                    outcome = do_record(n_corners, tvec_cur)
            if outcome == 'quit':
                stop = True
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        cv2.destroyAllWindows()
        # 保持使能（抱住位姿，避免下坠）；由用户决定是否下使能
        try:
            app.node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    print(f'\n自动标定结束：记录 {count} 个样本，目录 {session_dir}')
    print('机械臂仍处于使能/抱闸状态；如需下使能请在平台操作或运行 set_enable(false)。')
    if count >= 5:
        print(f'求解外参：python3 tools/calib_handeye_solve.py {session_dir}')
    else:
        print('样本偏少，建议 15 个以上。')


if __name__ == '__main__':
    run()
