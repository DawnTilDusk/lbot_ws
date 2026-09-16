#!/usr/bin/env python3
"""任务二：把螺母叠到玻璃盘中央的铁针上。

现场：玻璃盘中央立着一根铁针，要把白 / 黑大 / 中 / 小 四个螺母依次穿上去叠起来。

本模块只做两件核心事：

  1. detect_needle()   —— 跑 needle 两关键点 Pose 模型，取指定关键点（tip/base）的像素，
                          把它当成一个"检测中心"交给 nut_yolo.locate()，复用它的
                          **中心邻域深度中值 + 反投影**，再用手眼外参变换到 base_link，
                          得到针的 (x0, y0)。z 只作参考。

  2. place_at_needle() —— 右臂从"中央重抓点"（approach 段末点，刚重抓完螺母）出发，
                          竖直上升 -> 水平横移到针正上方 -> 竖直下降到释放高度 ->
                          张手释放 -> 再竖直抬起让开。全程走外部 IK + MoveJ
                          （复用 nut_pick_place.move_via_ik），
                          绕开 lbot_move_pose / lbot_move_linear 的内部 IK。

设计取舍：
  * **只检一次针**。针是固定的；而且第一个螺母叠上去之后针的样子就变了，
    每次重检只会越来越不准。所以开机双臂离场时检一次，之后一直用。
  * **Z 用固定值**。任务要求"z 取比桌面高 20cm"。而且细杆深度本来就不可靠
    （见 weights/needle_pose_report.md："细杆深度可能取到底座或背景，尚未验证"），
    所以 Z 不信视觉。
  * **X/Y 仍然依赖深度**（反投影要 z）。所以 keypoint 默认选 base（根部），
    并可配 fallback_depth 兜底。
"""
from pathlib import Path

import numpy as np

from nut_robot import TaskError


# ---------------- 配置 ----------------
DEFAULTS = dict(model='weights/needle_pose_best.pt', keypoint='base',
                release_height=0.20, transit_clearance=0.10,
                min_confidence=0.5, fallback_depth=None,
                # 叠针全程的目标末端姿态（三欧拉角，度）。null = 用右臂重抓完的
                # **当前抓握姿态**：螺母本来就是平着抓的，直接松手最自然，而且
                # 第一步竖直上升不需要同时大角度换向（换向会和上升叠一起，
                # 单关节变化超过保护阈值就被判"换臂型"中止）。
                release_euler_deg=None,
                # 叠针各段允许的最大单关节变化（度）。外部 IK 每段解都核对，
                # 超过就拒绝抡臂中止。收敛到别的分支时会报这个错。
                max_joint_diff_deg=60.0,
                # 叠放点在针的 (x0,y0) 基础上再补的平移（米，base_link 系）。
                # 视觉给的针位和"螺母实际该落在哪"之间有系统性偏差时用这个微调。
                release_offset_xyz=(0.0, 0.0, 0.0))

# 桌面在 base_link 系的估计高度（与 nut_pick_place 里的 TABLE_Z 保持一致）
TABLE_Z = -0.44


def needle_options(cfg):
    """从 TaskConfig 取 needle 段，缺省值补齐。"""
    raw = dict(getattr(cfg, 'needle_raw', {}) or {})
    opt = dict(DEFAULTS)
    opt.update({k: v for k, v in raw.items() if k in DEFAULTS})
    if opt['keypoint'] not in ('tip', 'base'):
        raise TaskError(f"needle.keypoint 只能是 tip 或 base，当前 {opt['keypoint']!r}")
    deg = opt.get('release_euler_deg')
    if deg is not None:
        deg = np.asarray(deg, float).reshape(-1)
        if deg.size != 3:
            raise TaskError('needle.release_euler_deg 必须是 3 个数（度）或 null，'
                            f'当前 {opt["release_euler_deg"]!r}')
        opt['release_euler_deg'] = deg
    off = np.asarray(opt['release_offset_xyz'], float).reshape(-1)
    if off.size != 3:
        raise TaskError('needle.release_offset_xyz 必须是 3 个数（米），'
                        f'当前 {opt["release_offset_xyz"]!r}')
    opt['release_offset_xyz'] = off
    try:
        mjd = float(opt['max_joint_diff_deg'])
    except (TypeError, ValueError):
        raise TaskError('needle.max_joint_diff_deg 必须是数字（度）')
    if not (0 < mjd <= 180):
        raise TaskError(f'needle.max_joint_diff_deg 要在 (0, 180] 内，当前 {mjd}')
    opt['max_joint_diff_deg'] = mjd
    return opt


def release_target(needle_xy, opt):
    """针的 (x0,y0) -> 释放点 base_link xyz（米）。Z 取桌面 + release_height。"""
    x0, y0 = float(needle_xy[0]), float(needle_xy[1])
    p = np.array([x0, y0, TABLE_Z + float(opt['release_height'])])
    return p + np.asarray(opt.get('release_offset_xyz', (0.0, 0.0, 0.0)), float)


# ---------------- 1. 检测铁针 ----------------
def detect_needle(cfg, node, K, R_BTC, t_BTC, log=print):
    """跑 Pose 模型定位铁针，返回 dict。

    返回 dict(x=, y=, z=, u=, v=, keypoint=, confidence=, depth=, detail=)；
    x/y/z 为 base_link 系米，(u,v) 为原图像素。
    任何一步失败都抛 TaskError —— 叠螺母之前必须先知道针在哪，不能瞎放。
    """
    from nut_yolo import YoloDetector, locate

    opt = needle_options(cfg)

    # 复用螺母 detector 的相机/超时配置，只换模型并放行 pose
    sub = dict(cfg.detector_raw)
    sub['model'] = opt['model']
    sub['allow_pose'] = True
    sub['roi'] = None          # 针在画面中央，不该受螺母 ROI 限制
    sub['_vision'] = {
        'color_topic': cfg.color_topic,
        'color_info_topic': cfg.color_info_topic,
        'depth_topic': cfg.depth_topic,
        'depth_info_topic': cfg.depth_info_topic,
        'camera_info_path': str(cfg.camera_info_path),
        'extrinsics_path': str(cfg.extrinsics_path),
    }

    det = YoloDetector(node, sub, K)
    log(f'  铁针检测：模型 {opt["model"]}，关键点 {opt["keypoint"]} ...')
    records, color, depth, camera = det.detect(expected=None, raw=True)

    if not records:
        raise TaskError('没检测到铁针（needle Pose 模型无输出）—— 检查针是否在视野内、'
                        'needle.model 是否为 needle_pose_best.pt')

    # 取置信度最高的那条 needle 记录
    best = max(records, key=lambda r: float(r.get('confidence', 0.0)))
    if str(best.get('task')) != 'pose' or not best.get('keypoints'):
        raise TaskError(f'needle 模型返回的不是 Pose 结果：{best!r}')

    kps = {str(k.get('name')): k for k in best['keypoints']}
    kp = kps.get(opt['keypoint'])
    if kp is None:
        raise TaskError(f'铁针结果里没有关键点 {opt["keypoint"]!r}，实际有 '
                        f'{sorted(kps)}；检查 needle.keypoint')

    conf = float(kp.get('confidence', 0.0))
    if (not kp.get('valid')) or conf < float(opt['min_confidence']):
        raise TaskError(f'铁针关键点 {opt["keypoint"]} 无效或置信度不足'
                        f'（conf={conf:.3f} < {opt["min_confidence"]}）—— '
                        f'调整光照/角度，或降低 needle.min_confidence')

    u, v = float(kp['u']), float(kp['v'])

    # 把关键点当"检测中心"交给 locate()：中心邻域深度中值 + 反投影
    record = dict(label=f'needle_{opt["keypoint"]}', u=u, v=v,
                  confidence=conf, bbox=best.get('bbox') or [u, v, u, v])
    locate_cfg = dict(cfg.detector_raw)
    locate_cfg['roi'] = None
    try:
        dets = locate([record], depth, camera['K'], color.shape, locate_cfg)
    except TaskError as exc:
        fb = opt.get('fallback_depth')
        if fb is None:
            raise TaskError(
                f'{exc}（细杆深度易失效。可把 needle.keypoint 换成 base，'
                f'或设 needle.fallback_depth 给一个固定相机系深度兜底）') from exc
        log(f'  ⚠ 关键点处深度无效（{exc}），改用 fallback_depth={fb} m')
        z_cam = float(fb)
        from camera_pick_move import deproject
        dets = [type('D', (), {'label': record['label'], 'p_cam': deproject(u, v, z_cam, camera['K'])})()]

    p_cam = np.asarray(dets[0].p_cam, float)
    p_base = np.asarray(R_BTC, float) @ p_cam + np.asarray(t_BTC, float)

    log(f'  铁针 {opt["keypoint"]}: 像素 ({u:.1f}, {v:.1f}) conf={conf:.3f} '
        f'深度 {p_cam[2] * 1000:.0f}mm  ->  base_link '
        f'[{p_base[0] * 1000:+.1f}, {p_base[1] * 1000:+.1f}, {p_base[2] * 1000:+.1f}] mm')

    return dict(x=float(p_base[0]), y=float(p_base[1]), z=float(p_base[2]),
                u=u, v=v, keypoint=opt['keypoint'], confidence=conf,
                depth=float(p_cam[2]), detail=f'Pose/{opt["keypoint"]}')


def grip_euler_rad(robot):
    """右臂当前末端姿态（rad，与记录段 eul_deg 同一 'xyz' 欧拉约定）；无反馈返回 None。

    重抓完的当前姿态就是"螺母平着握在手里"的姿态 —— 用它当叠针全程的目标姿态，
    第一步竖直上升就不需要同时大角度换向，也就不会一头撞上换臂型保护。
    """
    pose = getattr(robot, 'pose', None)
    if pose is None:
        return None
    try:
        from scipy.spatial.transform import Rotation as Rot
        eul = Rot.from_quat(np.asarray(pose[1], float).reshape(4)).as_euler('xyz')
    except Exception:  # noqa: BLE001  宁可回退也不要因为姿态转换挂掉
        return None
    return eul if np.all(np.isfinite(eul)) else None


def release_euler(cfg, robot, fallback_deg, log=print):
    """叠针释放姿态优先级：needle.release_euler_deg > 当前抓握姿态 > 入盒段末点姿态。"""
    opt = needle_options(cfg)
    if opt.get('release_euler_deg') is not None:
        deg = np.asarray(opt['release_euler_deg'], float)
        log(f'  [右] 叠针释放姿态 = needle.release_euler_deg {np.round(deg, 1)}°')
        return np.radians(deg)
    cur = grip_euler_rad(robot)
    if cur is not None:
        log('  [右] 叠针释放姿态 = 当前抓握姿态 '
            f'{np.round(np.degrees(cur), 1)}°（全程不换向）')
        return cur
    log('  [右] ⚠ 没有末端姿态反馈，回退入盒段末点姿态')
    return np.radians(np.asarray(fallback_deg, float))


# ---------------- 2. 送到针上方释放 ----------------
def place_at_needle(runner, robot, target_xyz, eul_release, ik_seeds, seed_names,
                    cfg, label, log=print):
    """右臂把夹着的螺母送到 target_xyz 上方释放。

    起点：右臂当前位姿（approach 段末点，刚完成中央重抓）。
    路径：竖直上升到中转平面 -> 水平横移到针正上方 -> 竖直下降到释放高度
          -> 张手 -> 竖直抬起让开。
    每一段都走 move_via_ik（外部多种子 IK + MoveJ），失败即运动前中止。
    """
    from nut_pick_place import move_via_ik, SIZE_NAMES_CN

    opt = needle_options(cfg)
    target = np.asarray(target_xyz, float)
    clearance = float(opt['transit_clearance'])
    max_jd = float(opt['max_joint_diff_deg'])

    def steps_for(dz):
        return max(2, int(round(abs(float(dz)) / 0.025)))

    if robot.pose is None:
        robot.wait_state()
    cur = np.asarray(robot.pose[0], float)
    transit_z = float(target[2]) + clearance

    log(f'  [右] 叠到铁针：目标 [{target[0] * 1000:+.1f}, {target[1] * 1000:+.1f}, '
        f'{target[2] * 1000:+.1f}] mm（中转平面 z={transit_z * 1000:+.1f}）')

    # ① 竖直上升到中转平面（当前 XY 不动）
    if cur[2] < transit_z - 0.005:
        rise = np.array([cur[0], cur[1], transit_z])
        log(f'  [右] MoveJ 竖直上升到中转平面 [{rise[0] * 1000:+.1f}, '
            f'{rise[1] * 1000:+.1f}, {rise[2] * 1000:+.1f}] mm')
        move_via_ik(runner, robot, rise, eul_release, ik_seeds, seed_names,
                    f'{SIZE_NAMES_CN.get(label, label)}螺母 上升让针', steps=2,
                    max_joint_diff_deg=max_jd)

    # ② 水平横移到针正上方
    over = np.array([target[0], target[1], transit_z])
    log(f'  [右] MoveJ 水平横移到针正上方 [{over[0] * 1000:+.1f}, {over[1] * 1000:+.1f}, '
        f'{over[2] * 1000:+.1f}] mm')
    move_via_ik(runner, robot, over, eul_release, ik_seeds, seed_names,
                f'{SIZE_NAMES_CN.get(label, label)}螺母 横移到铁针上方', steps=4,
                max_joint_diff_deg=max_jd)

    # ③ 竖直下降到释放高度
    drop = abs(float(target[2]) - transit_z)
    log(f'  [右] MoveJ 竖直下降到释放高度 {target[2] * 1000:+.1f} mm'
        f'（{drop * 1000:.1f}mm，分 {steps_for(drop)} 段）')
    move_via_ik(runner, robot, target, eul_release, ik_seeds, seed_names,
                f'{SIZE_NAMES_CN.get(label, label)}螺母 下降到铁针释放点',
                steps=steps_for(drop), max_joint_diff_deg=max_jd)

    # ④ 张手释放
    runner._dwell(cfg.pre_hand_seconds, '释放前')
    log(f'  [右] 张手释放到铁针（{cfg.hand_open_vals}），静置 {cfg.release_seconds:.1f}s')
    robot.hand_open(cfg.hand_open_vals)
    runner._dwell(cfg.release_seconds, '张手释放')

    # ⑤ 竖直抬起让开（避免蹭倒叠好的螺母）
    up = np.array([target[0], target[1], transit_z])
    log(f'  [右] MoveJ 竖直抬起让开到 {up[2] * 1000:+.1f} mm')
    move_via_ik(runner, robot, up, eul_release, ik_seeds, seed_names,
                f'{SIZE_NAMES_CN.get(label, label)}螺母 释放后抬起', steps=2,
                max_joint_diff_deg=max_jd)
