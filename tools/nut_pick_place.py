#!/usr/bin/env python3
"""双臂交接版螺母抓放主流程。

每颗螺母的执行链：
  0. 上使能、张开初始手型；左右臂各按预录 ready 轨迹（left/right.ready）逐点 MoveJ
     回放到末点离场（未配 ready 则慢速 MoveJ 直接补到首任务段 pt0）
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
                       TaskConfig, TaskError, pose_euler_rad, resolve_ws)
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


def grasp_points(cfg, det, R_BTC, t_BTC):
    """一条检测 -> (检测点 base, 腕部抓取目标 base, hover, down)。

    pb_raw 是视觉给出的螺母位置（base_link 系）；腕部目标 = 螺母位置 +
    该尺寸的 grasp_by_size.<label>.offset_xyz（未配则用全局 grasp_offset_xyz，
    默认向机体方向 -X 退 15cm 的腕-指尖偏差补偿）。
    hover/down 都以补偿后的腕部目标为基准：hover 竖直高该尺寸 hover_height，
    down 竖直偏该尺寸 z_offset。dry-run 打印、IK 预检、实机抓取必须共用此函数。
    """
    pb_raw = det_base_xyz(det, R_BTC, t_BTC)
    pb = pb_raw + cfg.grasp_offset_for(det.label)
    hover = pb + np.array([0, 0, cfg.hover_height_for(det.label)])
    down = pb + np.array([0, 0, cfg.grasp_z_offset_for(det.label)])
    return pb_raw, pb, hover, down


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
    """加载全部序列段并做跨段一致性检查。

    返回 (left_legs, approach, places, release_leg, ready_left, ready_right)；
    ready_* 未配置时为 None。place 段三颗共用时 places 三个键指向同一个 Leg 对象。
    """
    cache = {}

    def get(arm, spec):
        key = (arm, spec[0], spec[1])
        if key not in cache:
            cache[key] = load_leg(arm, spec, cfg.check_other)
        return cache[key]

    left_legs = [get('left', s) for s in cfg.left_legs]
    appr = get('right', cfg.right_approach)
    places = {k: get('right', cfg.right_place[k]) for k in SIZE_LABELS}
    ready_left = get('left', cfg.left_ready) if cfg.left_ready else None
    ready_right = get('right', cfg.right_ready) if cfg.right_ready else None

    sides = {'left': left_legs + ([ready_left] if ready_left else []),
             'right': [appr] + list(places.values()) + ([ready_right] if ready_right else [])}
    for side, legs in sides.items():
        ns0, nm0 = legs[0].namespace, legs[0].names
        for leg in legs[1:]:
            if leg.namespace != ns0:
                raise TaskError(f'{side} 各序列 namespace 不一致：{legs[0].target} vs {leg.target}')
            if leg.names != nm0:
                raise TaskError(f'{side} 各序列关节名顺序不一致：{legs[0].target} vs {leg.target}')

    release_leg = next(leg for leg in left_legs if leg.hand_after == 'open')
    return left_legs, appr, places, release_leg, ready_left, ready_right


def validate_detections(cfg, detections):
    """按 order 每类选一颗。同型号多目标按 detector.duplicate_policy 处理：
    abort（缺省）= 中止；random = 随机选一颗；first = 取列表第一颗。"""
    import random
    policy = getattr(cfg, 'duplicate_policy', 'abort')
    by_label = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)
    chosen = {}
    for label in cfg.order:
        dets = by_label.get(label, [])
        if not dets:
            continue
        if len(dets) > 1:
            if policy == 'abort':
                raise TaskError(
                    f'{SIZE_NAMES_CN[label]}螺母检测到 {len(dets)} 个目标，无法决定抓哪个'
                    f'（detector.duplicate_policy=abort；改成 random 可随机抓一颗）')
            if policy == 'random':
                pick = random.choice(dets)
            else:  # first
                pick = dets[0]
            # 不能用 list.index：Detection 含 ndarray，== 比较会触发真值歧义
            idx = next(i for i, d in enumerate(dets) if d is pick) + 1
            print(f'  {SIZE_NAMES_CN[label]}螺母检测到 {len(dets)} 个，按策略 {policy} '
                  f'抓第 {idx}/{len(dets)} 个 p_cam={np.round(pick.p_cam * 1000, 1)}mm')
            chosen[label] = pick
        else:
            chosen[label] = dets[0]
    missing = [k for k in cfg.order if k not in by_label]
    if missing and cfg.require_all:
        raise TaskError('缺少螺母检测结果：' + '、'.join(SIZE_NAMES_CN[k] for k in missing)
                        + '（require_all=true；配置改 false 可跳过）')
    return chosen


def detect_until_complete(cfg, detect_call, what='视觉检测'):
    """整轮检测 -> 缺型号就等一拍重新检测，最多 cfg.detect_attempts 轮。

    detect_call() 每轮重新取快照/重跑识别（无状态假设）；require_all=false 时缺料
    本就允许，不重试，直接走 validate_detections 的跳过逻辑。
    返回 (最后一轮的原始 detections, validate 后每尺寸选一颗的 chosen)。
    """
    import time
    attempts = max(1, int(getattr(cfg, 'detect_attempts', 3)))
    wait_s = max(0.0, float(getattr(cfg, 'missing_retry_seconds', 1.0)))
    detections = []
    for attempt in range(1, attempts + 1):
        detections = detect_call()
        present = {d.label for d in detections}
        missing = [k for k in cfg.order if k not in present]
        if not missing or not cfg.require_all:
            return detections, validate_detections(cfg, detections)
        names = '、'.join(SIZE_NAMES_CN[k] for k in missing)
        if attempt < attempts:
            print(f'{what}第 {attempt}/{attempts} 轮缺少 {names}螺母'
                  f'（已检测到 {len(detections)} 颗），等待 {wait_s:g}s 后重新检测...')
            if wait_s:
                time.sleep(wait_s)
        else:
            print(f'{what}连续 {attempts} 轮都缺少 {names}螺母')
    return detections, validate_detections(cfg, detections)  # 末轮仍缺 -> 抛 TaskError


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


def join_gap_rows(left_legs, appr, places, place_shared, ready_left=None, ready_right=None):
    """框架在相邻段之间自动补的慢速 MoveJ 接入距离（关节空间直动，无避障规划）。"""
    rows = []

    def gap(tag, a, b, warn_m=0.25):
        d = float(np.linalg.norm(a['xyz'] - b['xyz']))
        flag = '⚠ 距离较大，确认这条直 MoveJ 路径无干涉，或在段内多录过渡点' \
            if d > warn_m else ''
        rows.append(f'    {tag}: {d*1000:.0f}mm {flag}')

    if ready_left is not None:
        gap(f'开机 {ready_left.target}末(ready) -> 左臂首段首点'
            f'（视觉 MoveJP 直接从 ready 出发，驱动规划非框架直动）',
            ready_left.end_pose, left_legs[0].start_pose, warn_m=0.30)
    if ready_right is not None:
        gap(f'开机 {ready_right.target}末(ready) -> 右臂 approach 首（框架自动 MoveJ 接入）',
            ready_right.end_pose, appr.start_pose, warn_m=0.30)
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
               grasp_poses=None, ready_left=None, ready_right=None):
    """grasp_poses: {l/m/s: (eul_rad|None, 来源说明)}，由 resolve_grasp_eulers 得到。"""
    grasp_poses = grasp_poses or {}
    print('=' * 78)
    scale_txt = f'  速度倍率={cfg.speed_scale:g}' if cfg.speed_scale != 1.0 else ''
    print(f'双臂交接抓放  顺序={"".join(cfg.order)} '
          f'({"".join(SIZE_NAMES_CN[k] for k in cfg.order)})  '
          f'hover 默认 {cfg.hover_height*1000:.0f}mm'
          f'{scale_txt}')
    print('-' * 78)
    print('  【开机 ready 段】上使能 + 张开初始手型后逐点 MoveJ 回放，末点=视觉检测时的离场 ready 位：')
    for arm_cn, ready, fallback in (('左', ready_left, left_legs[0]),
                                    ('右', ready_right, appr)):
        if ready is None:
            print(f'    {arm_cn}臂：未配 ready 段，开机直接慢速 MoveJ 到任务段 '
                  f'{fallback.target} pt0')
        else:
            print_leg_table(f'{arm_cn}臂 ready：', [ready],
                            {id(ready): {len(ready.joints) - 1: '<- READY 检测时停在这'}})
    lm = {id(left_legs[0]): {0: '<- 任务段首点（retreat home）'}}
    rm = {id(appr): {0: '<- 任务段首点（retreat home）'}}
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
    for row in join_gap_rows(left_legs, appr, places, cfg.place_shared,
                             ready_left, ready_right):
        print(row)

    tol_txt = f'到位容差 {cfg.reached_tolerance}rad'
    if cfg.reached_tolerance_left is not None:
        tol_txt += f'（左臂 {cfg.reached_tolerance_left}）'
    if cfg.reached_tolerance_right is not None:
        tol_txt += f'（右臂 {cfg.reached_tolerance_right}）'
    tol_txt += (f'；等待 {cfg.reached_wait_seconds}s、未到位最多补发 '
                f'{cfg.reached_reissue_count} 次；视觉 MoveJP/MoveL 末端容差 '
                f'{cfg.pose_pos_tolerance * 1000:.0f}mm/{np.degrees(cfg.pose_ori_tolerance):.1f}°')
    print(f'  【{tol_txt}】')
    print('  【视觉抓取点偏移】检测点(螺母位置) -> 腕部目标（base_link 系；'
          '负 X=向机体方向，补偿腕-指尖前后偏差；未标注即全局值）：')
    for k in cfg.order:
        ov = '（覆盖全局）' if 'offset_xyz' in cfg.grasp_by_size.get(k, {}) else ''
        off_mm = np.round(cfg.grasp_offset_for(k) * 1000, 1)
        print(f'    {SIZE_NAMES_CN[k]}({k}): 偏移 {off_mm} mm{ov}，'
              f'hover +{cfg.hover_height_for(k)*1000:.0f}mm，'
              f'down {cfg.grasp_z_offset_for(k)*1000:+.0f}mm')
    print(f'  【手型】张开 {cfg.hand_open_vals}；闭合值（顺序[拇指侧摆,拇指弯曲,食,中,无名,小]）')
    for k in cfg.order:
        print(f'    {SIZE_NAMES_CN[k]}({k}): 左 {cfg.close_for("left", k)}  '
              f'右 {cfg.close_for("right", k)}')

    print('  【左臂视觉抓取姿态】（xyz 由视觉覆盖；姿态可按尺寸分别配置）：')
    for k in cfg.order:
        eul, src = grasp_poses.get(k, (None, ''))
        ov = '（覆盖全局）' if k in cfg.grasp_orientation_by_size else ''
        if eul is not None:
            print(f'    {SIZE_NAMES_CN[k]}({k}): 取自{src}{ov}，'
                  f'euler(deg)={np.degrees(eul).round(1)}')
        else:
            print(f'    {SIZE_NAMES_CN[k]}({k}): ⚠ {src or orientation_source_desc(cfg, k)}'
                  f'；--execute 前必须修好，dry-run 段表照常显示')

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
                policy = getattr(cfg, 'duplicate_policy', 'abort')
                what = {'abort': '执行时将中止', 'random': '执行时随机抓其中一颗',
                        'first': '执行时取第一颗'}[policy]
                print(f'    {SIZE_NAMES_CN[label]}螺母({label}): 检测到 {len(dets)} 个（{what}）')
            det = dets[0]
            in_base = det.extra.get('frame') in ('base', 'base_link')
            pb_raw, pb, hover, down = grasp_points(cfg, det, R_BTC, t_BTC)
            if in_base:
                print(f'    {SIZE_NAMES_CN[label]}螺母({label}) 检测点 base='
                      f'{np.round(pb_raw*1000,1)} mm（视觉直给，未走外参）')
            else:
                print(f'    {SIZE_NAMES_CN[label]}螺母({label}) p_cam={np.round(det.p_cam*1000,1)} mm')
                print(f'      检测点 base={np.round(pb_raw*1000,1)} mm（外参变换）')
            off_mm = np.round(cfg.grasp_offset_for(label) * 1000, 1)
            if np.linalg.norm(off_mm) > 0.1:
                ov = '，按尺寸覆盖' if 'offset_xyz' in cfg.grasp_by_size.get(label, {}) else ''
                print(f'      腕部目标 = 检测点 + 偏移 {off_mm} mm（负 X=向机体退，'
                      f'补偿腕-指尖偏差{ov}）')
            print(f'      腕部抓取目标 -> {np.round(pb*1000,1)}')
            print(f'      MoveJP hover(+{cfg.hover_height_for(label)*1000:.0f}mm) '
                  f'-> {np.round(hover*1000,1)}')
            print(f'      MoveL down({cfg.grasp_z_offset_for(label)*1000:+.0f}mm) '
                  f'-> {np.round(down*1000,1)}')
    else:
        print('  【视觉结果】manual 模式 dry-run 无检测结果；--execute 到位后弹窗点选。')
    print('=' * 78)


def resolve_grasp_euler(cfg, store, label=None, leg_cache=None):
    """视觉抓取姿态(rad)与来源说明。

    label=None 用全局 left.grasp_orientation（兼容旧调用）；传 l/m/s 时优先用
    left.grasp_orientation_by_size.<label>，缺覆盖回退全局。
    记录段映射 -> 取该点；位姿库名 -> PoseStore。leg_cache 在尺寸间复用加载结果。
    """
    if label is None:
        name, rec = cfg.left_grasp_pose_name, cfg.grasp_orientation_rec
        where = 'left.grasp_orientation'
    else:
        name, rec = cfg.grasp_orientation_for(label)
        where = f'left.grasp_orientation_by_size.{label}'
    if rec is not None:
        key = (str(rec['file']), rec['sequence'], rec['point'])
        if leg_cache is not None and key in leg_cache:
            leg = leg_cache[key]
        else:
            file_ = resolve_ws(rec['file']) if rec['file'] else resolve_ws(cfg.left_trace)
            spec = (file_, rec['sequence'], None, False)
            leg = load_leg('left', spec, cfg.check_other)
            if leg_cache is not None:
                leg_cache[key] = leg
        i = rec['point']
        if i >= len(leg.joints):
            raise TaskError(f'{where} 指定的 {rec["sequence"]} '
                            f'pt{i} 不存在（该段仅 {len(leg.joints)} 点）')
        eul = np.radians(leg.poses[i]['eul_deg'])
        src = f'记录段 {rec["sequence"]} pt{i}（{leg.file.name}）'
        return eul, src
    return pose_euler_rad(store.get('left', name)), f'已示教位姿 {name}'


def orientation_source_desc(cfg, label):
    """该尺寸姿态源的配置描述（报错/打印用）：位姿名或 记录段pt。"""
    name, rec = cfg.grasp_orientation_for(label)
    return (f'记录段 {rec["sequence"]} pt{rec["point"]}' if rec is not None else name)


def resolve_grasp_eulers(cfg, store):
    """{l,m,s: (eul_rad|None, 来源说明)}；某尺寸源不可用时 (None, 原因)。

    dry-run 只警告（计划照打）；--execute 在 main 里对 None 硬中止。
    """
    leg_cache = {}
    out = {}
    for k in SIZE_LABELS:
        try:
            out[k] = resolve_grasp_euler(cfg, store, k, leg_cache)
        except TaskError as exc:
            out[k] = (None, f'{orientation_source_desc(cfg, k)} 不可用：{exc}')
    return out


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

def initial_poses(cfg, runner, left_legs, appr, left, right,
                  ready_left=None, ready_right=None):
    """步骤0：注册 retreat home，双臂按预录 ready 轨迹离场（未配 ready 则直接补到任务段 pt0）。"""
    import time
    print('步骤0：双臂先回 ready（预录轨迹），再离场检测...')
    runner.set_home('left', left_legs[0].joints[0])
    runner.set_home('right', appr.joints[0])
    for arm, rc, ready, fallback in (('左', left, ready_left, left_legs[0]),
                                     ('右', right, ready_right, appr)):
        if ready is not None:
            leg, tgt = ready, ready.poses[-1]['xyz']
            print(f'  {arm}臂：回放 ready 段 {leg.target}'
                  f'（{len(leg.joints)} 点，先慢速接入 pt0 再逐点 MoveJ）')
        else:
            leg, tgt = fallback, fallback.poses[0]['xyz']
            print(f'  {arm}臂：未配 ready 段，直接慢速 MoveJ 到任务段 {leg.target} pt0')
        tgt_mm = tgt * 1000
        cur = rc.pose[0] * 1000 if rc.pose is not None else None
        cur_txt = f'，当前 {np.round(cur, 1)} mm' if cur is not None else ''
        print(f'  {arm}臂 ready 末点目标 {np.round(tgt_mm, 1)} mm（{leg.target}）{cur_txt}')
        if ready is not None:
            runner.run_leg(ready)  # join pt0 + 逐点回放；ready 段无手动作/retreat
        else:
            runner.join_to_start(leg, tag=f'（{arm}臂 ready）')
        rc.wait_state(0.5)
        now = rc.pose[0] * 1000 if rc.pose is not None else None
        if now is not None:
            err = np.linalg.norm(now - tgt_mm) * 0.001
            print(f'  {arm}臂 ready 到位，实际 {np.round(now, 1)} mm，位置误差 {err*1000:.0f}mm')
    print(f'  双臂保持 {cfg.ready_hold_seconds:.1f}s 后再开始视觉检测...')
    time.sleep(cfg.ready_hold_seconds)


def run_one_nut(cfg, runner, left, right, label, det, eul_left, R_BTC, t_BTC,
                left_legs, appr, places):
    pb_raw, pb, hover, down = grasp_points(cfg, det, R_BTC, t_BTC)

    def dwell(sec, what):
        runner._dwell(sec, what)

    print(f'==== {SIZE_NAMES_CN[label]}螺母({label}) ====')
    print(f'  检测点(螺母) {np.round(pb_raw * 1000, 1)}mm -> 腕部目标'
          f'{np.round(pb * 1000, 1)}mm（偏移 '
          f'{np.round(cfg.grasp_offset_for(label) * 1000, 1)}mm）')
    print(f'  [左] MoveJP 到螺母正上方 {np.round(hover * 1000, 1)}mm...')
    runner._goto_pose(left, hover, eul_left, f'{SIZE_NAMES_CN[label]}螺母 hover 悬停')
    dwell(cfg.hover_dwell_seconds, '悬停确认，准备下探')
    print(f'  [左] MoveL 竖直下探 {np.round(down * 1000, 1)}mm...')
    runner._goto_pose(left, down, eul_left, f'{SIZE_NAMES_CN[label]}螺母 down 下探',
                      linear=True)
    dwell(cfg.pre_hand_seconds, '闭合前')
    close_vals = cfg.close_for('left', label)
    print(f'  [左] 闭合手 {close_vals}，静置 {cfg.settle_seconds:.1f}s')
    left.hand_close(close_vals, settle=cfg.settle_seconds)
    dwell(cfg.pre_hand_seconds, '抓稳后抬起')
    print('  [左] MoveL 竖直抬起...')
    runner._goto_pose(left, hover, eul_left, f'{SIZE_NAMES_CN[label]}螺母抬起',
                      linear=True)
    dwell(cfg.between_leg_seconds, '进入固定段')

    for leg in left_legs:
        print(f'  [左] 回放段 {leg.target}')
        runner.run_leg(leg)
        dwell(cfg.between_leg_seconds, '左臂段间')

    print(f'  [右] 回放段 {appr.target}（段尾重抓{SIZE_NAMES_CN[label]}螺母）')
    runner.run_leg(appr, label=label)
    dwell(cfg.between_leg_seconds, '右臂段间')

    leg = places[label]
    print(f'  [右] 回放段 {leg.target}（入盒释放）')
    runner.run_leg(leg)


def go_ready(cfg, left_legs, appr, ready_left=None, ready_right=None):
    """只做开机动作：使能 -> 张开初始手型 -> 双臂按 ready 轨迹离场，然后退出（调试用）。"""
    import rclpy
    from nut_robot import RobotClient
    from nut_sequences import SequenceRunner

    rclpy.init()
    node = rclpy.create_node('nut_go_ready')
    left = RobotClient(node, cfg.namespace, 'left')
    right = RobotClient(node, cfg.namespace, 'right')
    try:
        for rc in (left, right):
            print(f'等待 {rc.arm} 臂服务与反馈...')
            rc.wait_services()
            if not rc.wait_state(3.0):
                raise TaskError(f'{rc.arm} 臂未收到 joint_states/pose_states 反馈')
        runner = SequenceRunner(left, right, cfg)
        left.enable()
        right.enable()
        for tag, rc in (('左', left), ('右', right)):
            rc.hand_setup(cfg.hand_speed, cfg.hand_force)
            print(f'  {tag}手张开 {cfg.hand_open_vals}')
            rc.hand_open(cfg.hand_open_vals)
        import time
        time.sleep(1.0)
        initial_poses(cfg, runner, left_legs, appr, left, right,
                      ready_left, ready_right)
        print('双臂已在 ready，--go-ready 结束（不执行抓放）。')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


VERIFIED_GRASP_POSE = (np.array([0.378, 0.326, -0.35]),
                       np.array([0.924, 0.0, -1.506]))  # 2026-09-10 手动 MoveJP 验证过的位姿


def ik_diagnose(robot, seed_leg, fail_pt, fail_eul):
    """预检所有种子都失败时的对照探针：区分 服务异常 / 位置问题 / 姿态问题 / 种子问题。"""
    last = seed_leg.poses[-1]
    p_rec, e_rec = last['xyz'], np.radians(last['eul_deg'])
    q_rec = seed_leg.joints[-1]
    p_ver, e_ver = VERIFIED_GRASP_POSE
    probes = [
        ('录段末点 + 自身记录臂型（IK 服务健全性）', p_rec, e_rec, q_rec),
        ('曾手动验证的 MoveJP 位姿', p_ver, e_ver, q_rec),
        ('失败点位置 + 录段末点姿态（位置可达？）', fail_pt, e_rec, q_rec),
        ('录段末点位置 + 目标姿态（姿态可达？）', p_rec, fail_eul, q_rec),
    ]
    mark = {'ok': 'ok', 'fail': 'fail', 'timeout': 'TIMEOUT'}
    rows = ['  【IK 对照探针】ok=可解  fail=驱动判不可解  TIMEOUT=8s 无响应']
    for name, p, e, seed in probes:
        r = robot.ik_try(p, e, seed)
        rows.append(f'    {mark[r]:>7}  {name}')
        rows.append(f'            xyz(mm)={np.round(p * 1000, 1)} '
                    f'eul(deg)={np.degrees(e).round(1)}')
    return rows


def execute(cfg, store, R_BTC, t_BTC, K, left_legs, appr, places, release_leg,
            grasp_poses, ready_left=None, ready_right=None):
    """grasp_poses: {l/m/s: (eul_rad, 来源说明)}；调用前 main 已保证 order 内尺寸都可用。"""
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

        print('上使能，设置手速/手力，双臂先张开到初始手型...')
        left.enable()
        right.enable()
        for tag, rc in (('左', left), ('右', right)):
            rc.hand_setup(cfg.hand_speed, cfg.hand_force)
            print(f'  {tag}手张开 {cfg.hand_open_vals}')
            rc.hand_open(cfg.hand_open_vals)
        import time
        time.sleep(1.0)  # 等手指张开动作完成再动臂

        # 先按预录 ready 轨迹离场，再拍视觉快照
        initial_poses(cfg, runner, left_legs, appr, left, right,
                      ready_left, ready_right)
        print(f'双臂已离场，开始视觉检测（{cfg.detector_type}），需要顺序：'
              f'{"".join(cfg.order)} ...')
        detector = build_detector(cfg, node, K)
        detections, by_label = detect_until_complete(
            cfg, lambda: detector.detect(cfg.order))
        print(f'检测到 {len(detections)} 颗：'
              f'{", ".join(SIZE_NAMES_CN[d.label] for d in detections)}')

        print('视觉抓取姿态（按尺寸）：')
        for k in cfg.order:
            eul_k, src_k = grasp_poses[k]
            ov = '（覆盖全局）' if k in cfg.grasp_orientation_by_size else ''
            print(f'  {SIZE_NAMES_CN[k]}({k})：{src_k}{ov}，euler(deg)={np.degrees(eul_k).round(1)}')
        # IK 种子：当前关节角之外，加录段里抓取区/低位的真实臂型，防远处种子数值收敛失败
        seed_leg = left_legs[0]
        ik_seeds = [seed_leg.joints[0], seed_leg.joints[-1]]
        seed_names = ['当前关节角', f'{seed_leg.target} pt0（记录臂型）',
                      f'{seed_leg.target} 末点（记录臂型）', '空种子（驱动自读当前角）']
        print(f'视觉抓取点 IK 预检（左臂，{len(by_label)} 颗；姿态按尺寸，位置来自视觉）...')
        first_fail = None
        for label in cfg.order:
            if label not in by_label:
                continue
            det = by_label[label]
            eul_label = grasp_poses[label][0]
            pb_raw, pb, hover, down = grasp_points(cfg, det, R_BTC, t_BTC)
            frame = det.extra.get('frame') in ('base', 'base_link')
            print(f'  {SIZE_NAMES_CN[label]}({label}) 检测点 base='
                  f'{np.round(pb_raw*1000,1)} mm（{"视觉直给" if frame else "外参变换"}）')
            print(f'    腕部目标(+偏移{np.round(cfg.grasp_offset_for(label)*1000,1)}mm)='
                  f'{np.round(pb*1000,1)} mm')
            for pt, tag in ((hover, 'hover'), (down, 'down')):
                ok, used = left.ik_check(pt, eul_label, extra_seeds=ik_seeds)
                if ok:
                    print(f'    {tag} {np.round(pt*1000,1)} 逆解通过（种子：{seed_names[used]}）')
                elif first_fail is None:
                    first_fail = (label, tag, pt, pb, eul_label)

        if first_fail is not None:
            label, tag, pt, pb, eul_fail = first_fail
            lines = ik_diagnose(left, seed_leg, pt, eul_fail)
            for line in lines:
                print(line)
            msg = (f'左臂对{SIZE_NAMES_CN[label]}螺母 {tag} 逆解失败（试遍种子 '
                   f'{"、".join(seed_names)}）：xyz(mm)={np.round(pt*1000,1)} '
                   f'euler(deg)={np.degrees(eul_fail).round(1)}。对照探针见上。')
            if getattr(cfg, 'allow_ik_fail', False):
                print('  ⚠ --allow-ik-fail：跳过预检继续。真实 MoveJP/MoveL 仍由驱动把关，'
                      '驱动解不了会在该步安全中止（不会盲动）。')
            else:
                raise TaskError(
                    msg + '若探针证明点本身可达、只是裸 IK 服务误判，确认现场安全后可加 '
                          '--allow-ik-fail 让真实 MoveJP 尝试（驱动仍会拒绝不可达点）；'
                          '否则排查视觉位置/深度/外参，或换 '
                          f'left.grasp_orientation_by_size.{label} / left.grasp_orientation 姿态。')
        if first_fail is None:
            print('视觉点全部可达，开始抓放。')
        print_plan(cfg, store, left_legs, appr, places, release_leg,
                   detections, R_BTC, t_BTC, grasp_poses=grasp_poses,
                   ready_left=ready_left, ready_right=ready_right)

        remaining = [k for k in cfg.order if k in by_label]
        for idx, label in enumerate(remaining):
            det = by_label[label]
            run_one_nut(cfg, runner, left, right, label, det, grasp_poses[label][0],
                        R_BTC, t_BTC, left_legs, appr, places)
            if idx < len(remaining) - 1:
                print(f'  螺母之间停顿 {cfg.between_leg_seconds:.1f}s，'
                      f'下一颗 {SIZE_NAMES_CN[remaining[idx + 1]]}...')
                runner._dwell(cfg.between_leg_seconds, '螺母之间')
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
    p.add_argument('--detector', choices=('manual', 'json', 'external', 'input', 'yolo'),
                   default=None,
                   help="input=临时联调：终端手动输入三颗螺母 base_link 位置，不走相机；"
                        "yolo=实时彩色/对齐深度快照 + YOLO 识别")
    p.add_argument('--speed', type=float, default=1.0, metavar='N',
                   help='整体速度倍率（相对 yaml，如 1.5=提速 50%%，0.5=减半），'
                        '同时缩放视觉 MoveJP/MoveL、段间接入、序列逐点的速度与加速度')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--go-ready', action='store_true',
                   help='真机只做开机动作（使能/张开手/回 ready）后退出，便于单独验证')
    p.add_argument('--allow-ik-fail', action='store_true',
                   help='裸 IK 预检全失败也继续：真实 MoveJP/MoveL 仍由驱动把关，'
                        '不可达会在该步安全中止。仅在对照探针证明是预检误判时使用')
    p.add_argument('--show', action='store_true',
                   help='YOLO 识别时弹窗实时显示画面/检测框（即使 yaml detector.show_window=false）')
    args = p.parse_args()

    try:
        cfg = TaskConfig(
            args.config, arm_override=args.arm, order_override=args.order,
            detector_override=args.detector, speed_scale=args.speed)
        cfg.allow_ik_fail = bool(args.allow_ik_fail)
        if args.show:
            cfg.detector_raw['show_window'] = True
        if args.speed != 1.0:
            print(f'速度倍率 --speed {args.speed:g}：视觉段 {cfg.speed:g}/{cfg.linear_speed:g}、'
                  f'段间接入 {cfg.join_speed:g}、序列逐点 {cfg.sequence_speed:g}')
            if cfg.join_speed > 0.3 + 1e-9 or cfg.sequence_speed > 0.5 + 1e-9:
                print('  ⚠ 已超过 yaml 安全建议上限（join 0.3 / 逐点 0.5 rad/s），'
                      '首调请谨慎，手持急停')
        store = PoseStore(cfg.poses_path)
        R, t, K = load_transforms(cfg)
        left_legs, appr, places, release_leg, ready_left, ready_right = load_all_legs(cfg)

        # 三个尺寸的姿态源各自解析（不可用的为 (None, 原因)：dry-run 只警告，--execute 硬中止）
        grasp_poses = resolve_grasp_eulers(cfg, store)

        if args.go_ready:
            if not args.execute:
                raise TaskError('--go-ready 是真机动作，必须与 --execute 一起用')
            go_ready(cfg, left_legs, appr, ready_left, ready_right)
        elif args.execute:
            missing = [(k, grasp_poses[k][1]) for k in cfg.order
                       if grasp_poses.get(k, (None, ''))[0] is None]
            if missing:
                lines = '；'.join(f'{SIZE_NAMES_CN[k]}({k}): {why}' for k, why in missing)
                raise TaskError(
                    f'左臂视觉抓取姿态源不可用——{lines}。位姿名请先运行 '
                    'capture_task_pose.py --arm left --name <名>（如 left_grasp_l/m/s）；'
                    '或把 left.grasp_orientation(_by_size.<尺寸>) 改成 '
                    '{file?, sequence, point?} 指向记录段')
            # 执行前仍打印骨架与交接点核对（检测点到位后再补打印）
            execute(cfg, store, R, t, K, left_legs, appr, places, release_leg,
                    grasp_poses, ready_left, ready_right)
        else:
            detections = []
            if cfg.detector_type == 'yolo':
                from nut_yolo import detect_once
                detections, _ = detect_until_complete(
                    cfg, lambda: detect_once(cfg)[0])
            elif cfg.detector_type in ('json', 'external', 'input'):
                from nut_detectors import build_detector
                detector = build_detector(cfg, None, K)
                detections, _ = detect_until_complete(
                    cfg, lambda: detector.detect(cfg.order))
            print_plan(cfg, store, left_legs, appr, places, release_leg,
                       detections, R, t, grasp_poses=grasp_poses,
                       ready_left=ready_left, ready_right=ready_right)
            print('\n[dry-run] 未驱动机器人。确认段表/交接点/抓取点后加 --execute 真机执行。')
    except TaskError as exc:
        print(f'\n[中止] {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
