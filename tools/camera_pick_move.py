#!/usr/bin/env python3
"""相机点选/坐标输入 -> base_link -> move_pose（会真实驱动机械臂！）。

两种用法：
  1) 画面点选（默认）：订阅彩色+【对齐】深度图，鼠标点目标，显示该点相机坐标/base坐标，
     回车确认 -> 上使能 -> move_pose 到目标正上方（默认抬高 5cm）；q 退出。
  2) 直接给相机坐标：--cam-point x y z（米，camera_color_optical_frame：Z前/Y下/X右），
     只打印 base 坐标；加 --go 直接运动。

链路：像素 (u,v) + 对齐深度 z -> 反投影 p_cam=(X,Y,Z) -> 外参 B_T_C -> p_base -> move_pose。

⚠️ 安全：本工具会让机械臂运动！清空活动范围、手持急停。默认只移动到目标【正上方 5cm】
   （--lift），不直接扎到台面；确认无误后用 --lift 0 / --approach 做贴近。

前置：
  # 相机（深度对齐彩色）+ 机器人驱动已启动，且已标定外参
  ros2 launch orbbec_camera gemini2.launch.py depth_registration:=true
  python3 tools/calib_handeye_publish_tf.py   # 可选，供 RViz 看；本工具直接读外参 yaml

用法：
  python3 tools/camera_pick_move.py
  python3 tools/camera_pick_move.py --lift 0.03
  python3 tools/camera_pick_move.py --cam-point 0.02 -0.05 0.55 --go
  python3 tools/camera_pick_move.py --euler-deg 0 -90 0     # 指定姿态；默认工具朝下自动搜 yaw
界面：横向三联 = 彩色(可点选) | 伪彩对齐深度 | 反投影点云(3/4 视角，选中点黄十字)。
交互（只在最左彩色面板点选）：鼠标左键=选点；回车/空格=确认前往；s=清除；q=退出。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from calib_common import (CALIB_DIR, EXTRINSICS_PATH, COLOR_CAMERA_INFO_PATH,
                          quat_xyzw_to_mat)

# 默认抓取朝向：工具竖直向下（厂商右臂示例 euler = 0,-90,0）。绕工具轴的 yaw 会自动搜索。
DEFAULT_GRASP_EULER_DEG = [0.0, -90.0, 0.0]
# 工具朝下时，绕竖直轴搜索的腕部 yaw（度），按顺序试，取第一个逆解成功的
YAW_SEARCH_DEG = [0, -90, 90, 180, -45, 45, -135, 135]


def load_extrinsics(path):
    d = yaml.safe_load(open(path, encoding='utf-8'))
    R = quat_xyzw_to_mat(d['rotation_xyzw'])
    t = np.array(d['translation'], dtype=float)
    return R, t, d


def load_camera_K(path):
    """从标定保存的 camera_info yaml 读内参 K（3x3）；无文件/无 K 返回 None。"""
    if path is None or not Path(path).exists():
        return None
    d = yaml.safe_load(open(path, encoding='utf-8'))
    K = d.get('K')
    if not K:
        return None
    K = np.array(K, float).reshape(3, 3)
    if K[0, 0] <= 0:  # 占位/无效内参
        return None
    return K


def deproject(u, v, z, K):
    """像素+深度(m)->相机光学系 3D 点。"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def sample_depth(depth, u, v, win=4, max_r=14):
    """取 (u,v) 附近有效深度的中位数（mm 或 m 自动归一化到米）；无有效深度返回 None。"""
    h, w = depth.shape[:2]
    scale = 1000.0 if depth.dtype == np.uint16 else 1.0  # 16UC1 通常是 mm
    for r in (win, max_r):
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch = depth[v0:v1, u0:u1].astype(np.float64).reshape(-1)
        valid = patch[patch > 0]
        if np.isfinite(valid).any():
            valid = valid[np.isfinite(valid)]
            z = float(np.median(valid)) / scale
            if 0.05 < z < 3.0:
                return z
    return None


def _depth_m(depth):
    """深度图(16UC1=mm / 32FC=m) -> 米，无效为 0。"""
    scale = 1000.0 if depth.dtype == np.uint16 else 1.0
    z = depth.astype(np.float64) / scale
    z[~np.isfinite(z)] = 0.0
    return z


def colorize_depth(depth, sel=None, lo=0.25, hi=1.30):
    """深度图 -> BGR 伪彩（近红远蓝，无效黑），并标出选点。"""
    import cv2
    z = _depth_m(depth)
    valid = z > 0
    vis = np.zeros(z.shape, np.uint8)
    if valid.any():
        # 用 2~98 分位做对比度（夹在合理范围内）
        p2, p98 = np.percentile(z[valid], (2, 98))
        a = float(np.clip(p2, 0.2, 1.0))
        b = float(np.clip(p98, a + 0.1, 2.5))
        a, b = (lo, hi) if not (b > a) else (max(a, lo), min(b, hi))
        n = np.clip((z - a) / (b - a + 1e-9), 0, 1)
        vis = (n * 255).astype(np.uint8)
        vis[~valid] = 0
    col = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    col[~valid] = (0, 0, 0)
    if sel is not None:
        cv2.drawMarker(col, (sel['u'], sel['v']), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(col, (sel['u'], sel['v']), 9, (255, 255, 255), 2)
    return col


def render_cloud(depth, color, K, p_sel=None, step=8, W=560, H=315):
    """对齐深度反投影成点云，虚拟 3/4 视角渲染；选中点高亮。"""
    import cv2
    from scipy.spatial.transform import Rotation as Rot
    canvas = np.zeros((H, W, 3), np.uint8)
    z = _depth_m(depth)
    h, w = z.shape
    vv, uu = np.mgrid[0:h:step, 0:w:step]
    zs = z[vv, uu]
    m = (zs > 0.05) & (zs < 3.0)
    uu, vv, zs = uu[m], vv[m], zs[m]
    cv2.putText(canvas, 'POINTCLOUD (cam frame)', (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if len(zs) < 20:
        cv2.putText(canvas, 'NO DEPTH', (W // 2 - 50, H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        return canvas
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.stack([(uu - cx) * zs / fx, (vv - cy) * zs / fy, zs], axis=1)
    C = color[vv, uu].astype(np.uint8)
    center = p_sel if p_sel is not None else np.median(P, axis=0)
    Rv = Rot.from_euler('xyz', [0.42, -0.62, 0.0]).as_matrix()  # 抬高+侧转的虚拟视角
    Q = (P - center) @ Rv.T
    tz = 1.6
    zc = Q[:, 2] + tz
    f = 520.0
    us = (f * Q[:, 0] / zc + W / 2).astype(int)
    vs2 = (f * Q[:, 1] / zc + H / 2).astype(int)
    inside = (us >= 0) & (us < W) & (vs2 >= 0) & (vs2 < H)
    # 远的先画（painter）
    order = np.argsort(-zc[inside])
    ui, vi, ci = us[inside][order], vs2[inside][order], C[inside][order]
    canvas[vi, ui] = ci
    # 选中点高亮
    if p_sel is not None:
        q = Rv @ (np.asarray(p_sel) - center)
        ux = int(f * q[0] / (q[2] + tz) + W / 2)
        vx = int(f * q[1] / (q[2] + tz) + H / 2)
        if 0 <= ux < W and 0 <= vx < H:
            cv2.drawMarker(canvas, (ux, vx), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(canvas, (ux, vx), 8, (0, 255, 255), 2)
    return canvas


def aligned_depth_and_K(node):
    """取对齐深度；分辨率与彩色不一致时最近邻缩放，并返回应用的内参。"""
    import cv2
    d = node.depth
    if node.color is not None and d is not None and d.shape[:2] != node.color.shape[:2]:
        d = cv2.resize(d, (node.color.shape[1], node.color.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    K = node.K_color if node.K_color is not None else node.K_depth
    return d, K


# ---------------- 点选 GUI ----------------
def run_gui(args, R_BTC, t_BTC, ext):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo, JointState
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
    import cv2
    from lbot_arm_interfaces.srv import MoveJP, MoveL, SetEnable, InverseKinematics

    class PickNode(Node):
        def __init__(self):
            super().__init__('camera_pick_move')
            self.bridge = CvBridge()
            self.color = None
            self.depth = None
            # 优先用最新标定采集保存的内参 yaml（启动即有，不等 camera_info 话题）
            K_file = load_camera_K(args.camera_info)
            self.K_color = K_file
            self.K_src = f'FILE {Path(args.camera_info).name}' if K_file is not None else None
            self.K_depth = None
            self.pose = None  # (pos(3), quat(4))
            self.joints = None  # 当前 7 关节角，用作逆解种子
            self.create_subscription(Image, args.color_topic, self._img, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_topic, self._dep, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.color_info_topic, self._kci, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.depth_info_topic, self._kd, qos_profile_sensor_data)
            ns = '/' + args.namespace.strip('/')
            self.create_subscription(PoseStamped, f'{ns}/{args.arm}_arm/pose_states',
                                     self._pose, qos_profile_sensor_data)
            self.create_subscription(JointState, f'{ns}/{args.arm}_arm/joint_states',
                                     self._joints, qos_profile_sensor_data)
            self.move_cli = self.create_client(MoveJP, f'{ns}/{args.arm}_arm/move_pose')
            self.linear_cli = self.create_client(MoveL, f'{ns}/{args.arm}_arm/move_linear')
            self.enable_cli = self.create_client(SetEnable, f'{ns}/{args.arm}_arm/set_enable')
            self.ik_cli = self.create_client(InverseKinematics, f'{ns}/{args.arm}_arm/inverse_kinematics')
            self.sel = None  # dict(u,v,p_cam,p_base,z,valid,reason,plan)

        def _joints(self, m):
            if len(m.position) == 7:
                self.joints = [float(x) for x in m.position]

        def _img(self, m):
            try:
                self.color = self.bridge.imgmsg_to_cv2(m, 'bgr8')
            except Exception:
                pass

        def _dep(self, m):
            try:
                self.depth = self.bridge.imgmsg_to_cv2(m, 'passthrough')
            except Exception:
                pass

        def _kci(self, m):
            # 标定内参文件存在时以文件为准；文件缺失才回落到实时话题
            if self.K_src is None or not self.K_src.startswith('FILE'):
                self.K_color = np.array(m.k, float).reshape(3, 3)
                self.K_src = 'TOPIC'

        def _kd(self, m):
            self.K_depth = np.array(m.k, float).reshape(3, 3)

        def _pose(self, m):
            p, q = m.pose.position, m.pose.orientation
            self.pose = (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))

    rclpy.init()
    node = PickNode()
    cv2.namedWindow('pick', cv2.WINDOW_NORMAL)
    PW, PH = 560, 315  # 单面板显示尺寸（三联横排，只有最左彩色面板可点选）

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x >= PW or y >= PH:
            return  # 点到深度/点云面板，忽略
        if node.color is None or node.depth is None:
            return
        depth_aligned, K = aligned_depth_and_K(node)
        H0, W0 = depth_aligned.shape[:2]
        u = int(round(x * W0 / PW))
        v = int(round(y * H0 / PH))
        if K is None:
            node.sel = dict(u=u, v=v, valid=False, reason='NO CAMERA INFO')
            return
        z = sample_depth(depth_aligned, u, v)
        if z is None:
            node.sel = dict(u=u, v=v, valid=False, reason='INVALID/EMPTY DEPTH')
            return
        p_cam = deproject(u, v, z, K)
        p_base = R_BTC @ p_cam + t_BTC
        node.sel = dict(u=u, v=v, z=z, p_cam=p_cam, p_base=p_base, valid=True)

    cv2.setMouseCallback('pick', on_mouse)

    def spin_call(cli, req, timeout=60):
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            if fut.done():
                return fut.result()
        return None

    def candidate_eulers():
        # 显式指定就只试该姿态；否则工具竖直向下 (0,-90,yaw)，绕竖直轴搜索可达 yaw
        if args.euler_deg is not None:
            return [np.radians(np.asarray(args.euler_deg, float))]
        return [np.radians([DEFAULT_GRASP_EULER_DEG[0], DEFAULT_GRASP_EULER_DEG[1], yaw])
                for yaw in YAW_SEARCH_DEG]

    def ik_solve(p, eul, timeout=5):
        """单点逆解；优先用当前 7 关节角做种子（数值解更稳、解更接近当前臂型）。"""
        req = InverseKinematics.Request()
        req.position.x, req.position.y, req.position.z = [float(v) for v in p]
        req.euler.x, req.euler.y, req.euler.z = [float(v) for v in eul]
        req.joints = list(node.joints) if node.joints is not None else []
        res = spin_call(node.ik_cli, req, timeout)
        if res is None:
            return False, None
        return bool(res.success), (list(res.joints) if res.success else None)

    def plan_target(p_base):
        """对预抓点（及 --approach 的真实落点）做逆解可达性预检。

        可达性唯一判据是驱动实时逆解 IK，不设任何人为坐标范围框；
        逆解不成功就不动。返回 dict(ok, reason, eul(rad), p_go)。
        """
        p_go = np.asarray(p_base, float) + np.array([0.0, 0.0, args.lift])
        if not node.ik_cli.service_is_ready():
            return dict(ok=False, reason='IK SERVICE UNAVAILABLE (driver?)', eul=None, p_go=p_go)
        if args.keep_down and node.pose is None:
            return dict(ok=False, reason='NO ARM POSE (keep-down need pose_states)',
                        eul=None, p_go=p_go)
        euls = candidate_eulers()
        for i, eul in enumerate(euls):
            ok1, _ = ik_solve(p_go, eul)
            if not ok1:
                continue
            # --approach 会直线贴近到真实点（比预抓点低 lift），该点也要可达
            if args.approach and args.lift > 1e-6:
                ok2, _ = ik_solve(np.asarray(p_base, float), eul)
                if not ok2:
                    continue
            # --keep-down：先原地转正再 MoveL 平移，故“当前位置+朝下姿态”也要可达
            if args.keep_down:
                ok3, _ = ik_solve(node.pose[0], eul)
                if not ok3:
                    continue
            tag = '' if (args.euler_deg is not None) else f' yaw={YAW_SEARCH_DEG[i]}deg'
            return dict(ok=True, reason=f'IK OK{tag}', eul=eul, p_go=p_go)
        tried = 'given euler' if args.euler_deg is not None else f'{len(euls)} yaw candidates'
        print(f'[IK] {p_go.round(3)} 逆解失败（{tried}），该点当前姿态不可达。', flush=True)
        return dict(ok=False, reason=f'IK UNREACHABLE ({tried})', eul=None, p_go=p_go)

    def do_move(sel):
        plan = sel.get('plan')
        if not plan or not plan.get('ok'):
            print(f'不可达，不运动：{plan.get("reason") if plan else "no plan"}', flush=True)
            return False
        p_go, eul = plan['p_go'], plan['eul']
        if not node.enable_cli.service_is_ready() or not node.move_cli.service_is_ready():
            print('set_enable/move_pose 服务不可用，驱动是否启动？', flush=True)
            return False
        print(f'-> base 目标(含抬高{args.lift*1000:.0f}mm) = {p_go.round(3)} m, '
              f'euler(deg)={np.degrees(eul).round(1)}，上使能并运动...', flush=True)
        er = spin_call(node.enable_cli, SetEnable.Request(enable=True), timeout=10)
        if not (er is not None and er.success):
            print('上使能失败。', flush=True)
            return False

        if args.keep_down:
            # 全程保持工具朝下：先在当前位置原地转正（MoveL 同点变姿），
            # 再 MoveL 直线平移到预抓点——直线运动全程姿态恒定。
            from scipy.spatial.transform import Rotation as Rot
            if node.pose is None:
                print('--keep-down 需要当前末端位姿（pose_states 未收到），中止。', flush=True)
                return False
            if not node.linear_cli.service_is_ready():
                print('move_linear 服务不可用，--keep-down 无法执行，中止。', flush=True)
                return False
            R_now = Rot.from_quat(node.pose[1])
            R_tgt = Rot.from_euler('xyz', eul)
            ang = float(np.degrees((R_tgt * R_now.inv()).magnitude()))
            if ang > 8.0:
                req0 = MoveL.Request()  # 笛卡尔直线（同点变姿），保证末端位置不动只转姿态
                p0 = node.pose[0]
                req0.position.x, req0.position.y, req0.position.z = [float(v) for v in p0]
                req0.euler.x, req0.euler.y, req0.euler.z = [float(v) for v in eul]
                req0.speed = min(float(args.speed), 0.3)
                req0.acce = min(float(args.acce), 0.3)
                req0.block = True
                print(f'   [keep-down] 原地转正 {ang:.0f}°（位置不动）...', flush=True)
                r0 = spin_call(node.linear_cli, req0)
                if r0 is None or not r0.success:
                    print('   原地转正失败，中止（未平移）。', flush=True)
                    return False
            req = MoveL.Request()
            req.position.x, req.position.y, req.position.z = [float(v) for v in p_go]
            req.euler.x, req.euler.y, req.euler.z = [float(v) for v in eul]
            req.speed = min(float(args.speed), 0.3)
            req.acce = min(float(args.acce), 0.3)
            req.block = True
            print('   [keep-down] move_linear 水平平移（姿态保持朝下）...', flush=True)
            res = spin_call(node.linear_cli, req)
            if res is None or not res.success:
                print('   直线平移失败/超时。', flush=True)
                return False
        else:
            req = MoveJP.Request()
            req.position.x, req.position.y, req.position.z = [float(v) for v in p_go]
            req.euler.x, req.euler.y, req.euler.z = [float(v) for v in eul]
            req.speed, req.acce, req.block = float(args.speed), float(args.acce), True
            res = spin_call(node.move_cli, req)
            if res is None or not res.success:
                print('move_pose 失败/超时。', flush=True)
                return False
        print('到位。', flush=True)
        if args.approach:
            if not node.linear_cli.service_is_ready():
                print('   move_linear 服务不可用，跳过贴近。', flush=True)
                return True
            down = np.asarray(sel['p_base'], float)  # 落到真实点（move_linear 直线贴近）
            req2 = MoveL.Request()
            req2.position.x, req2.position.y, req2.position.z = [float(v) for v in down]
            req2.euler = req.euler
            req2.speed = min(float(args.speed), 0.3)
            req2.acce = min(float(args.acce), 0.3)
            req2.block = True
            print(f'   --approach：move_linear 降速直线贴近 {down.round(3)} ...', flush=True)
            r2 = spin_call(node.linear_cli, req2)
            print('   贴近完成。' if r2 is not None and r2.success else '   贴近失败。', flush=True)
        return True

    print('点选窗口：左键选点 -> 显示相机/base坐标 -> 回车前往(目标上方)；s 清除；q 退出。')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            have_color = node.color is not None
            frame = node.color.copy() if have_color else np.zeros((720, 1280, 3), np.uint8)
            lines = [f'ext ~{ext.get("residual_reproj_mean_mm", float("nan")):.1f}mm '
                     f'lift={args.lift*1000:.0f}mm sp={args.speed} '
                     f'{"KEEP-DOWN" if args.keep_down else "MoveJP"}',
                     f'depth:{"OK" if node.depth is not None else "WAIT"} '
                     f'K:{node.K_src if node.K_src is not None else "WAIT"} '
                     f'pose:{"OK" if node.pose is not None else "WAIT"}']
            sel = node.sel
            if sel is not None:
                u, v = sel['u'], sel['v']
                cv2.drawMarker(frame, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
                if sel.get('valid'):
                    # 新选点：做一次逆解可达性预检（结果缓存在 sel['plan']）
                    if 'plan' not in sel:
                        sel['plan'] = plan_target(sel['p_base'])
                    plan = sel['plan']
                    pc, pb = sel['p_cam'], sel['p_base']
                    pg = plan['p_go']
                    lines += [
                        f'px({u},{v}) z={sel["z"]:.3f}m',
                        f'cam :{pc[0]*1000:7.0f}{pc[1]*1000:7.0f}{pc[2]*1000:7.0f} mm',
                        f'base:{pb[0]*1000:7.0f}{pb[1]*1000:7.0f}{pb[2]*1000:7.0f} mm',
                        f'hover:{pg[0]*1000:6.0f}{pg[1]*1000:7.0f}{pg[2]*1000:7.0f} mm  {plan["reason"]}']
                    if plan['ok']:
                        ed = np.degrees(plan['eul'])
                        lines += [f'euler(deg):{ed[0]:6.1f}{ed[1]:7.1f}{ed[2]:7.1f}',
                                  'ENTER=move  s=clear  q=quit']
                        cv2.circle(frame, (u, v), 9, (0, 255, 0), 2)
                    else:
                        lines += [plan['reason'][:34],
                                  'IK cannot solve this pose',
                                  'change point/orientation - s=clear']
                        cv2.circle(frame, (u, v), 9, (0, 0, 255), 2)
                else:
                    lines += [f'px({u},{v}) {sel.get("reason","INVALID")}',
                              'click a surface WITH depth']
            else:
                lines.append('LEFT CLICK a target')
            plan_ok = bool(sel and sel.get('valid') and sel.get('plan', {}).get('ok'))
            if plan_ok:
                col = (0, 200, 0)
            elif sel and sel.get('valid'):
                col = (0, 0, 220)
            else:
                col = (0, 200, 255)
            for i, ln in enumerate(lines):
                cv2.putText(frame, ln, (12, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, ln, (12, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            col, 1, cv2.LINE_AA)

            # 三面板：彩色 / 伪彩深度 / 点云（PW/PH 见上方鼠标回调）
            p_color = cv2.resize(frame, (PW, PH))
            if node.depth is not None:
                depth_a, Kd = aligned_depth_and_K(node)
                p_depth = cv2.resize(colorize_depth(depth_a, sel), (PW, PH))
                if Kd is not None:  # 内参到位才反投影点云（启动瞬间 camera_info 可能还没到）
                    base_color = node.color if have_color else np.zeros((depth_a.shape[0], depth_a.shape[1], 3), np.uint8)
                    ps = sel['p_cam'] if (sel and sel.get('valid')) else None
                    p_cloud = render_cloud(depth_a, base_color, Kd, ps, W=PW, H=PH)
                else:
                    p_cloud = np.zeros((PH, PW, 3), np.uint8)
                    cv2.putText(p_cloud, 'CAMERA INFO WAITING...', (90, PH // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                p_depth = np.zeros((PH, PW, 3), np.uint8)
                cv2.putText(p_depth, 'DEPTH WAITING...', (120, PH // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                p_cloud = np.zeros((PH, PW, 3), np.uint8)
                cv2.putText(p_cloud, 'POINTCLOUD WAITING...', (110, PH // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(p_color, 'COLOR', (8, PH - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(p_depth, 'DEPTH(aligned)', (8, PH - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            combo = np.hstack([p_color, p_depth, p_cloud])
            cv2.imshow('pick', combo)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('s'), ord('S')):
                node.sel = None
            if key in (32, 13, 10):
                if plan_ok:
                    do_move(sel)
                elif sel is not None and sel.get('valid'):
                    print(f'该点不可达，不运动：{sel["plan"]["reason"]}', flush=True)
                else:
                    print('请先用左键选一个有效点。', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ---------------- 纯命令行坐标模式 ----------------
def run_cli(args, R_BTC, t_BTC):
    pc = np.array(args.cam_point, float)
    pb = R_BTC @ pc + t_BTC
    print(f'相机坐标 p_cam = {pc.round(4)} m')
    print(f'基座坐标 p_base = {pb.round(4)} m')
    print(f'预抓点(抬高 {args.lift*1000:.0f}mm) = {(pb+np.array([0,0,args.lift])).round(4)} m')
    if not args.go:
        print('仅换算，未运动。加 --go 执行逆解预检 + move_pose。')
        return
    import rclpy
    from lbot_arm_interfaces.srv import MoveJP, MoveL, SetEnable, InverseKinematics
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from scipy.spatial.transform import Rotation as Rot
    rclpy.init()
    node = rclpy.create_node('camera_pick_move_cli')
    ns = '/' + args.namespace.strip('/')
    enable_cli = node.create_client(SetEnable, f'{ns}/{args.arm}_arm/set_enable')
    move_cli = node.create_client(MoveJP, f'{ns}/{args.arm}_arm/move_pose')
    linear_cli = node.create_client(MoveL, f'{ns}/{args.arm}_arm/move_linear')
    ik_cli = node.create_client(InverseKinematics, f'{ns}/{args.arm}_arm/inverse_kinematics')
    need_linear = args.keep_down or args.approach
    for c in (enable_cli, move_cli, ik_cli) + ((linear_cli,) if need_linear else ()):
        if not c.wait_for_service(timeout_sec=5):
            sys.exit('set_enable/move_pose/inverse_kinematics 服务不可用，驱动是否启动？')
    # 取当前 7 关节角作逆解种子（1.5s 内拿不到就用空种子）；keep-down 还要当前位姿
    state = {}
    node.create_subscription(JointState, f'{ns}/{args.arm}_arm/joint_states',
                             lambda m: state.update(j=list(m.position)), 10)
    node.create_subscription(PoseStamped, f'{ns}/{args.arm}_arm/pose_states',
                             lambda m: state.update(pos=[m.pose.position.x, m.pose.position.y,
                                                         m.pose.position.z],
                                                    quat=[m.pose.orientation.x, m.pose.orientation.y,
                                                          m.pose.orientation.z, m.pose.orientation.w]), 10)
    t0 = time.monotonic()
    need = {'j'} | ({'pos', 'quat'} if args.keep_down else set())
    while not need.issubset(state) and time.monotonic() - t0 < 1.5:
        rclpy.spin_once(node, timeout_sec=0.1)
    seed = state.get('j') if len(state.get('j', [])) == 7 else None
    if seed is not None:
        print('逆解种子=当前关节角。')
    cur = None
    if args.keep_down:
        if 'pos' not in state or 'quat' not in state:
            sys.exit('--keep-down 需要当前末端位姿（pose_states 未收到）。')
        cur = (np.array(state['pos'], float), np.array(state['quat'], float))

    p_go = pb + np.array([0, 0, args.lift])

    def call(cli, req, timeout=60):
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            if fut.done():
                return fut.result()
        return None

    # 默认工具竖直向下 (0,-90,yaw) 自动搜可达 yaw；--euler-deg 则只试给定姿态
    if args.euler_deg is not None:
        euls = [np.radians(np.asarray(args.euler_deg, float))]
    else:
        euls = [np.radians([DEFAULT_GRASP_EULER_DEG[0], DEFAULT_GRASP_EULER_DEG[1], yaw])
                for yaw in YAW_SEARCH_DEG]
    eul = None
    for i, e in enumerate(euls):
        rq = InverseKinematics.Request()
        rq.position.x, rq.position.y, rq.position.z = [float(v) for v in p_go]
        rq.euler.x, rq.euler.y, rq.euler.z = [float(v) for v in e]
        rq.joints = list(seed) if seed is not None else []
        rr = call(ik_cli, rq, timeout=5)
        ok_go = rr is not None and rr.success
        ok_final = True
        if ok_go and args.approach and args.lift > 1e-6:
            rq2 = InverseKinematics.Request()
            rq2.position.x, rq2.position.y, rq2.position.z = [float(v) for v in pb]
            rq2.euler = rq.euler
            rq2.joints = list(seed) if seed is not None else []
            rr2 = call(ik_cli, rq2, timeout=5)
            ok_final = rr2 is not None and rr2.success
        if ok_final and args.keep_down:  # 当前位置原地转正也要可达
            rq3 = InverseKinematics.Request()
            rq3.position.x, rq3.position.y, rq3.position.z = [float(v) for v in cur[0]]
            rq3.euler = rq.euler
            rq3.joints = list(seed) if seed is not None else []
            rr3 = call(ik_cli, rq3, timeout=5)
            ok_final = rr3 is not None and rr3.success
        if ok_go and ok_final:
            eul = e
            tag = '' if args.euler_deg is not None else f'（自动 yaw={YAW_SEARCH_DEG[i]}°）'
            print(f'逆解通过{tag}，euler(deg)={np.degrees(e).round(1)}')
            break
    if eul is None:
        sys.exit('所有候选姿态逆解均失败：该点当前姿态不可达，不运动（无人为范围框，仅以逆解为准）。')

    er = call(enable_cli, SetEnable.Request(enable=True), 10)
    if not (er and er.success):
        sys.exit('上使能失败。')

    if args.keep_down:
        # 全程保持工具朝下：先原地转正，再 MoveL 直线平移（姿态恒定）
        R_now = Rot.from_quat(cur[1])
        R_tgt = Rot.from_euler('xyz', eul)
        ang = float(np.degrees((R_tgt * R_now.inv()).magnitude()))
        if ang > 8.0:
            req0 = MoveL.Request()  # 同点变姿：末端位置不动，只把姿态转正朝下
            req0.position.x, req0.position.y, req0.position.z = [float(v) for v in cur[0]]
            req0.euler.x, req0.euler.y, req0.euler.z = [float(v) for v in eul]
            req0.speed = min(float(args.speed), 0.3)
            req0.acce = min(float(args.acce), 0.3)
            req0.block = True
            r0 = call(linear_cli, req0)
            if r0 is None or not r0.success:
                sys.exit(f'原地转正失败（差 {ang:.0f}°），未平移。')
            print(f'已在原地转正 {ang:.0f}°。')
        req = MoveL.Request()
        req.position.x, req.position.y, req.position.z = [float(v) for v in p_go]
        req.euler.x, req.euler.y, req.euler.z = [float(v) for v in eul]
        req.speed = min(float(args.speed), 0.3)
        req.acce = min(float(args.acce), 0.3)
        req.block = True
        res = call(linear_cli, req)
        if res is None or not res.success:
            sys.exit('直线平移失败/超时。')
    else:
        req = MoveJP.Request()
        req.position.x, req.position.y, req.position.z = [float(v) for v in p_go]
        req.euler.x, req.euler.y, req.euler.z = [float(v) for v in eul]
        req.speed, req.acce, req.block = float(args.speed), float(args.acce), True
        res = call(move_cli, req)
        if res is None or not res.success:
            sys.exit('move_pose 失败/超时。')
    print('到位（预抓点）。')
    if args.approach:
        req2 = MoveL.Request()
        req2.position.x, req2.position.y, req2.position.z = [float(v) for v in pb]
        req2.euler = req.euler
        req2.speed = min(float(args.speed), 0.3)
        req2.acce = min(float(args.acce), 0.3)
        req2.block = True
        r2 = call(linear_cli, req2)
        print('贴近完成。' if r2 is not None and r2.success else '贴近失败。')
    node.destroy_node()
    rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--namespace', default='/robot1')
    p.add_argument('--arm', choices=('left', 'right'), default='right')
    p.add_argument('--color-topic', default='/camera/color/image_raw')
    p.add_argument('--color-info-topic', default='/camera/color/camera_info')
    p.add_argument('--depth-topic', default='/camera/depth/image_raw',
                   help='对齐到彩色的深度图（depth_registration:=true 时的 depth/image_raw）')
    p.add_argument('--depth-info-topic', default='/camera/depth/camera_info')
    p.add_argument('--extrinsics', type=Path, default=EXTRINSICS_PATH)
    p.add_argument('--camera-info', type=Path, default=COLOR_CAMERA_INFO_PATH,
                   help='标定保存的彩色内参 yaml（默认用最新采集生成的 gemini2_color_camera_info.yaml）')
    p.add_argument('--cam-point', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                   help='直接给相机光学系坐标(米)：Z前/Y下/X右；不开启 GUI')
    p.add_argument('--go', action='store_true', help='--cam-point 模式下真正运动')
    p.add_argument('--lift', type=float, default=0.05, help='目标上方抬高 m（base +Z），默认 0.05')
    p.add_argument('--approach', action='store_true', help='到位后再 MoveL 贴近到真实点')
    p.add_argument('--keep-down', action='store_true',
                   help='全程保持工具朝下：先原地转正，再 MoveL 直线平移到预抓点（姿态全程不变）')
    p.add_argument('--euler-deg', type=float, nargs=3, metavar=('RX', 'RY', 'RZ'),
                   help='末端姿态 intrinsic XYZ 度；默认工具竖直向下 (0,-90,yaw) 并自动搜索可达 yaw')
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--acce', type=float, default=0.5)
    args = p.parse_args()

    if not args.extrinsics.exists():
        sys.exit(f'未找到外参 {args.extrinsics}，请先运行 calib_handeye_solve.py。')
    R, t, ext = load_extrinsics(args.extrinsics)
    print(f'已加载外参 {args.extrinsics.name}（残差均值 {ext.get("residual_reproj_mean_mm")}mm）')
    K0 = load_camera_K(args.camera_info)
    if K0 is not None:
        print(f'已加载标定内参 {args.camera_info.name}: '
              f'fx={K0[0,0]:.2f} fy={K0[1,1]:.2f} cx={K0[0,2]:.2f} cy={K0[1,2]:.2f}')
    else:
        print(f'未找到标定内参 {args.camera_info}，回退到实时 /camera/color/camera_info。')

    if args.cam_point is not None:
        run_cli(args, R, t)
    else:
        run_gui(args, R, t, ext)


if __name__ == '__main__':
    main()
