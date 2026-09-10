#!/usr/bin/env python3
"""双臂交接版螺母抓放主流程。

每颗螺母的执行链：
  0. 左右臂初始位姿 = 各自序列 home（第一段第一个点），开工慢速 MoveJ 到位
  1. 双臂离场后检测（点选/json/external）拿到 p_cam -> base_link（两臂共用 base 系）
  2. 左视觉抓：MoveJP 到螺母正上方 hover -> MoveL 竖直下探 -> 按尺寸闭合 -> MoveL 抬起
  3. 左固定段：join+逐点回放 left.legs，末段 open 释放到桌面中央后 retreat 回 home
  4. 右 approach 段：join+回放，段尾 close 重抓
  5. 右 place 段（按尺寸选段）：join+回放，段尾 open 入盒，retreat 回 home

安全机制：视觉点运动前全部 IK 预检；序列段校验反馈新鲜/关节名/frame、另一只臂静止；
缺检测、缺位姿/序列、同尺寸多目标都在运动前中止。默认 dry-run。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nut_robot import (DEFAULT_CONFIG, SIZE_LABELS, SIZE_NAMES_CN, PoseStore,
                       TaskConfig, TaskError, pose_euler_rad)
from nut_sequences import load_leg

HANDOFF_WARN_M = 0.020   # 左释放点 vs 右抓取点距离超过此值 dry-run 警告


# ---------------- 纯函数 ------------------------------------------------------

def cam_to_base(p_cam, R_BTC, t_BTC):
    return R_BTC @ np.asarray(p_cam, float) + t_BTC


def det_base_xyz(det, R_BTC, t_BTC):
    """检测结果 -> base_link 系 xyz。

    检测器给相机系点：走外参变换；检测器直接给 base 系机械臂坐标
    （extra.frame=base_link）：原样使用，跳过手眼链路。
    """
    if det.extra.get('frame') in ('base', 'base_link'):
        return np.asarray(det.p_cam, float)
    return cam_to_base(det.p_cam, R_BTC, t_BTC)


def parse_order(text):
    """'sml' / 's,m,l' -> ['s','m','l']。"""
    chars = text.replace(',', ' ').replace('_', ' ').split()
    if len(chars) == 1 and len(chars[0]) == 3:
        chars = list(chars[0])
    out = [c.lower() for c in chars]
    if any(c not in SIZE_LABELS for c in out):
        raise argparse.ArgumentTypeError(f'顺序只允许 l/m/s，收到 {text!r}')
    return out


def load_all_legs(cfg):
    """加载全部序列段并做跨段一致性检查。返回 (left_legs, approach, places, release_leg)。

    place 段三颗共用时 places 三个键指向同一个 Leg 对象。
    """
    left_legs = [load_leg('left', s, cfg.check_other) for s in cfg.left_legs]
    appr = load_leg('right', cfg.right_approach, cfg.check_other)
    cache = {(appr.file, appr.target): appr}
    places = {}
    for k in SIZE_LABELS:
        spec = cfg.right_place[k]
        key = (spec[0], spec[1])
        if key not in cache:
            cache[key] = load_leg('right', spec, cfg.check_other)
        places[k] = cache[key]

    sides = {'left': left_legs, 'right': [appr] + list(places.values())}
    for side, legs in sides.items():
        ns0, nm0 = legs[0].namespace, legs[0].names
        for leg in legs[1:]:
            if leg.namespace != ns0:
                raise TaskError(f'{side} 各序列 namespace 不一致：{legs[0].target} vs {leg.target}')
            if leg.names != nm0:
                raise TaskError(f'{side} 各序列关节名顺序不一致：{legs[0].target} vs {leg.target}')

    release_leg = next(leg for leg in left_legs if leg.hand_after == 'open')
    return left_legs, appr, places, release_leg


def validate_detections(cfg, detections):
    by_label = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)
    for label, dets in by_label.items():
        if len(dets) > 1:
            raise TaskError(f'{SIZE_NAMES_CN[label]}螺母检测到 {len(dets)} 个目标，无法决定抓哪个')
    missing = [k for k in cfg.order if k not in by_label]
    if missing and cfg.require_all:
        raise TaskError('缺少螺母检测结果：' + '、'.join(SIZE_NAMES_CN[k] for k in missing)
                        + '（require_all=true；配置改 false 可跳过）')
    return {k: by_label[k][0] for k in cfg.order if k in by_label}


# ---------------- dry-run 打印 -----------------------------------------------

def print_leg_table(title, legs, markers=None):
    markers = markers or {}
    print(f'  {title}')
    for leg in legs:
        extra = []
        if leg.hand_after:
            extra.append(f'{leg.hand_after.upper()}@end')
        if leg.retreat:
            extra.append('RETREAT')
        rel = leg.file
        print(f'    段 {leg.target}  [{rel}]  {len(leg.joints)}点'
              + (f"  ({', '.join(extra)})" if extra else ''))
        for i, ps in enumerate(leg.poses):
            tag = markers.get(id(leg), {}).get(i, '')
            mm, deg = ps['xyz'] * 1000, ps['eul_deg']
            print(f'      pt{i} {"*" if tag else " "} xyz(mm)=[{mm[0]:+7.1f} {mm[1]:+7.1f} '
                  f'{mm[2]:+7.1f}] euler=[{deg[0]:+5.0f},{deg[1]:+5.0f},{deg[2]:+5.0f}] {tag}')


def handoff_check(release_leg, appr):
    """关键交接点：左释放点 vs 右 approach 末点(close 处)。只警告不阻塞。"""
    d = float(np.linalg.norm(release_leg.end_pose['xyz'] - appr.end_pose['xyz']))
    flag = '⚠ >20mm，右臂可能抓空，建议重录对齐' if d > HANDOFF_WARN_M else 'ok'
    return [f'    左释放点 vs 右臂重抓点(approach终点): {d*1000:.0f}mm {flag}']


def join_gap_rows(left_legs, appr, places, place_shared):
    """框架在相邻段之间自动补的慢速 MoveJ 接入距离（关节空间直动，无避障规划）。"""
    rows = []

    def gap(tag, a, b, warn_m=0.25):
        d = float(np.linalg.norm(a['xyz'] - b['xyz']))
        flag = '⚠ 距离较大，确认这条直 MoveJ 路径无干涉，或在段内多录过渡点' \
            if d > warn_m else ''
        rows.append(f'    {tag}: {d*1000:.0f}mm {flag}')

    for i in range(len(left_legs) - 1):
        gap(f'左臂 {left_legs[i].target}末 -> {left_legs[i+1].target}首',
            left_legs[i].end_pose, left_legs[i + 1].start_pose)
    gap(f'左臂 {left_legs[-1].target}末(回位) -> 下轮 {left_legs[0].target}首(home)',
        left_legs[-1].end_pose, left_legs[0].start_pose)
    any_place = places['l']
    gap(f'右臂 {appr.target}末(抓螺母) -> place段首',
        appr.end_pose, any_place.start_pose, warn_m=0.30)
    tag = 'place段末(释放) -> 下轮 approach首(home)' if place_shared \
        else 'place_l段末 -> 下轮 approach首(home)'
    gap(tag, any_place.end_pose, appr.start_pose)
    if not place_shared:
        # 三段分立时，各 place 首点应就是中央重抓点
        for k in SIZE_LABELS:
            d = float(np.linalg.norm(appr.end_pose['xyz'] - places[k].start_pose['xyz']))
            flag = '⚠ >20mm，中央点没对齐' if d > HANDOFF_WARN_M else 'ok'
            rows.append(f'    右臂重抓点 vs place_{k}段首: {d*1000:.0f}mm {flag}')
    return rows


def print_plan(cfg, store, left_legs, appr, places, release_leg, detections, R_BTC, t_BTC,
               grasp_eul=None):
    print('=' * 78)
    print(f'双臂交接抓放  顺序={"".join(cfg.order)} '
          f'({"".join(SIZE_NAMES_CN[k] for k in cfg.order)})  hover={cfg.hover_height*1000:.0f}mm')
    print('-' * 78)
    lm = {id(left_legs[0]): {0: '<- HOME 初始位姿'}}
    rm = {id(appr): {0: '<- HOME 初始位姿'}}
    rlm = {id(release_leg): {len(release_leg.joints) - 1: '<- 中央释放点'}}
    print_leg_table('【左臂固定段】视觉抓起后依次回放：', left_legs,
                    {**lm, **rlm})
    print_leg_table('【右臂固定段】approach（每颗都回放，末点闭合重抓）：', [appr], rm)
    if cfg.place_shared:
        shared = places['l']
        print_leg_table('【右臂固定段】place（三颗共用一条）：', [shared],
                        {id(shared): {0: '<- 段间接入点(框架自动MoveJ，螺母已抓起)',
                                      len(shared.joints) - 1: '<- 盒位释放点 OPEN（l/m/s 共用）'}})
    else:
        place_markers = {}
        for k, leg in places.items():
            place_markers[id(leg)] = {0: '<- 中央重抓点',
                                      len(leg.joints) - 1: f'<- {k}格释放点'}
        print_leg_table('【右臂固定段】place（按尺寸三选一）：',
                        list(places.values()), place_markers)
    print('  【交接点核对】')
    for row in handoff_check(release_leg, appr):
        print(row)
    print('  【段间接入距离】相邻段端点不一致时，框架以慢速 MoveJ 直动接入：')
    for row in join_gap_rows(left_legs, appr, places, cfg.place_shared):
        print(row)

    print('  【闭合值】（每颗螺母；顺序[拇指侧摆,拇指弯曲,食,中,无名,小]）')
    for k in cfg.order:
        print(f'    {SIZE_NAMES_CN[k]}({k}): 左 {cfg.close_for("left", k)}  '
              f'右 {cfg.close_for("right", k)}')

    if grasp_eul is not None:
        print(f'  【左臂视觉抓取姿态】取自已示教 {cfg.left_grasp_pose_name}：'
              f'euler(deg)={np.degrees(grasp_eul).round(1)}（xyz 由视觉覆盖）')
    else:
        print(f'  【左臂视觉抓取姿态】⚠ {cfg.left_grasp_pose_name} 尚未采集'
              f'（capture_task_pose.py --arm left --name {cfg.left_grasp_pose_name}）；'
              f'--execute 前必须采，dry-run 段表照常显示')

    if detections:
        by_label = {}
        for det in detections:
            by_label.setdefault(det.label, []).append(det)
        print('  【视觉结果与左臂抓取点】')
        for label in cfg.order:
            dets = by_label.get(label, [])
            if not dets:
                why = 'require_all=true 将中止' if cfg.require_all else '将跳过'
                print(f'    {SIZE_NAMES_CN[label]}螺母({label}): 未检测到（{why}）')
                continue
            if len(dets) > 1:
                print(f'    {SIZE_NAMES_CN[label]}螺母({label}): 检测到 {len(dets)} 个（执行时将中止）')
            det = dets[0]
            in_base = det.extra.get('frame') in ('base', 'base_link')
            pb = det_base_xyz(det, R_BTC, t_BTC)
            hover = pb + np.array([0, 0, cfg.hover_height])
            down = pb + np.array([0, 0, cfg.grasp_z_offset])
            if in_base:
                print(f'    {SIZE_NAMES_CN[label]}螺母({label}) base={np.round(pb*1000,1)} mm'
                      f'（视觉直给，未走外参）')
            else:
                print(f'    {SIZE_NAMES_CN[label]}螺母({label}) p_cam={np.round(det.p_cam*1000,1)} mm')
                print(f'      base={np.round(pb*1000,1)} mm（外参变换）')
            print(f'      MoveJP hover(+{cfg.hover_height*1000:.0f}mm) -> {np.round(hover*1000,1)}')
            print(f'      MoveL down({cfg.grasp_z_offset*1000:+.0f}mm) -> {np.round(down*1000,1)}')
    else:
        print('  【视觉结果】manual 模式 dry-run 无检测结果；--execute 到位后弹窗点选。')
    print('=' * 78)


# ---------------- 外参/内参 ---------------------------------------------------

def load_transforms(cfg):
    from camera_pick_move import load_extrinsics, load_camera_K
    if not cfg.extrinsics_path.exists():
        raise TaskError(f'找不到外参 {cfg.extrinsics_path}')
    R, t, ext = load_extrinsics(cfg.extrinsics_path)
    K = load_camera_K(cfg.camera_info_path)
    print(f'外参 {cfg.extrinsics_path.name}（残差均值 {ext.get("residual_reproj_mean_mm")}mm）')
    if K is None:
        print(f'注意：未读到标定内参 {cfg.camera_info_path.name}，'
              f'manual 检测器将等待 {cfg.color_info_topic}')
    return R, t, K


# ---------------- 实机执行 ---------------------------------------------------

def initial_poses(cfg, runner, left_legs, appr):
    """步骤0：注册 home，慢速 MoveJ 到左右初始位姿（已在容差内则不动）。"""
    print('步骤0：双臂回到初始位姿（各序列 home）...')
    runner.set_home('left', left_legs[0].joints[0])
    runner.set_home('right', appr.joints[0])
    runner.join_to_start(left_legs[0], tag='（左臂初始位姿）')
    runner.join_to_start(appr, tag='（右臂初始位姿）')


def run_one_nut(cfg, runner, left, right, label, det, eul_left, R_BTC, t_BTC,
                left_legs, appr, places):
    pb = det_base_xyz(det, R_BTC, t_BTC)
    hover = pb + np.array([0, 0, cfg.hover_height])
    down = pb + np.array([0, 0, cfg.grasp_z_offset])

    print(f'==== {SIZE_NAMES_CN[label]}螺母({label}) ====')
    print('  [左] MoveJP 到螺母正上方...')
    left.move_pose(hover, eul_left, cfg.speed, cfg.acce, cfg.move_timeout)
    print('  [左] MoveL 竖直下探...')
    left.move_linear(down, eul_left, cfg.linear_speed, cfg.linear_acce, cfg.move_timeout)
    close_vals = cfg.close_for('left', label)
    print(f'  [左] 闭合手 {close_vals}，静置 {cfg.settle_seconds:.1f}s')
    left.hand_close(close_vals, settle=cfg.settle_seconds)
    print('  [左] MoveL 竖直抬起...')
    left.move_linear(hover, eul_left, cfg.linear_speed, cfg.linear_acce, cfg.move_timeout)

    for leg in left_legs:
        print(f'  [左] 回放段 {leg.target}')
        runner.run_leg(leg)

    print(f'  [右] 回放段 {appr.target}（段尾重抓{SIZE_NAMES_CN[label]}螺母）')
    runner.run_leg(appr, label=label)

    leg = places[label]
    print(f'  [右] 回放段 {leg.target}（入盒释放）')
    runner.run_leg(leg)


def execute(cfg, store, R_BTC, t_BTC, K, left_legs, appr, places, release_leg):
    import rclpy
    from nut_robot import RobotClient
    from nut_sequences import SequenceRunner
    from nut_detectors import build_detector

    rclpy.init()
    node = rclpy.create_node('nut_pick_place')
    left = RobotClient(node, cfg.namespace, 'left')
    right = RobotClient(node, cfg.namespace, 'right')
    try:
        for rc in (left, right):
            print(f'等待 {rc.arm} 臂服务与反馈...')
            rc.wait_services()
            if not rc.wait_state(3.0):
                raise TaskError(f'{rc.arm} 臂未收到 joint_states/pose_states 反馈')
        runner = SequenceRunner(left, right, cfg)

        print('上使能，设置手速/手力，双臂张手...')
        left.enable()
        right.enable()
        for rc in (left, right):
            rc.hand_setup(cfg.hand_speed, cfg.hand_force)
            rc.hand_open(cfg.hand_open_vals)

        # 先到初始位姿（双臂离场），再拍视觉快照
        initial_poses(cfg, runner, left_legs, appr)
        print(f'双臂已离场，开始视觉检测（{cfg.detector_type}），需要顺序：'
              f'{"".join(cfg.order)} ...')
        detector = build_detector(cfg, node, K)
        detections = detector.detect(cfg.order)
        print(f'检测到 {len(detections)} 颗：'
              f'{", ".join(SIZE_NAMES_CN[d.label] for d in detections)}')
        by_label = validate_detections(cfg, detections)

        eul_left = pose_euler_rad(store.get('left', cfg.left_grasp_pose_name))
        print(f'视觉抓取点 IK 预检（左臂，{len(by_label)} 颗）...')
        for label in cfg.order:
            if label not in by_label:
                continue
            pb = det_base_xyz(by_label[label], R_BTC, t_BTC)
            for pt, tag in ((pb + [0, 0, cfg.hover_height], 'hover'),
                            (pb + [0, 0, cfg.grasp_z_offset], 'down')):
                if not left.ik_check(pt, eul_left):
                    raise TaskError(
                        f'左臂对{SIZE_NAMES_CN[label]}螺母 {tag} 逆解失败，中止：'
                        f'xyz(mm)={np.round(pt*1000,1)} euler={np.degrees(eul_left).round(1)}')
        print('视觉点全部可达，开始抓放。')
        print_plan(cfg, store, left_legs, appr, places, release_leg,
                   detections, R_BTC, t_BTC, grasp_eul=eul_left)

        for label in cfg.order:
            det = by_label.get(label)
            if det is None:
                print(f'-- 跳过 {SIZE_NAMES_CN[label]}螺母（未检测到）--')
                continue
            run_one_nut(cfg, runner, left, right, label, det, eul_left,
                        R_BTC, t_BTC, left_legs, appr, places)
        print('全部完成，双臂已回 home。')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ---------------- 入口 -------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--arm', choices=('left', 'right'), default=None,
                   help='只影响 capture_task_pose 默认臂；任务固定双臂')
    p.add_argument('--order', type=parse_order, default=None)
    p.add_argument('--detector', choices=('manual', 'json', 'external', 'yolo'), default=None)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()

    try:
        cfg = TaskConfig(args.config, arm_override=args.arm,
                         order_override=args.order, detector_override=args.detector)
        store = PoseStore(cfg.poses_path)
        R, t, K = load_transforms(cfg)
        left_legs, appr, places, release_leg = load_all_legs(cfg)

        try:
            grasp_eul = pose_euler_rad(store.get('left', cfg.left_grasp_pose_name))
        except TaskError:
            grasp_eul = None  # 未采姿态：dry-run 只警告；--execute 仍会硬中止

        if args.execute:
            if grasp_eul is None:
                raise TaskError(
                    f'左臂视觉抓取姿态 {cfg.left_grasp_pose_name} 未采集；先运行 '
                    f'capture_task_pose.py --arm left --name {cfg.left_grasp_pose_name}')
            # 执行前仍打印骨架与交接点核对（检测点到位后再补打印）
            execute(cfg, store, R, t, K, left_legs, appr, places, release_leg)
        else:
            detections = []
            if cfg.detector_type == 'yolo':
                from nut_yolo import detect_once
                detections, _ = detect_once(cfg)
                validate_detections(cfg, detections)
            if cfg.detector_type in ('json', 'external'):
                from nut_detectors import build_detector
                detector = build_detector(cfg, None, K)
                detections = detector.detect(cfg.order)
            print_plan(cfg, store, left_legs, appr, places, release_leg,
                       detections, R, t, grasp_eul=grasp_eul)
            print('\n[dry-run] 未驱动机器人。确认段表/交接点/抓取点后加 --execute 真机执行。')
    except TaskError as exc:
        print(f'\n[中止] {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
