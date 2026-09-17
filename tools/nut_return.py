#!/usr/bin/env python3
"""螺母任务安全回退：把双臂安全送回开机安全起始位（ready 段起点 pt0）。

两条路线
--------
1. **在 ready 轨迹上**（开机回放完 ready、抓取中途失败停在 ready 附近、抓完一颗回 ready）：
   把 left/right.ready 这条示教轨迹【逆向】逐点 MoveJ 回放，终点就是 pt0。
   正向 ready 是示教过的关节路径，逆向回放的是同一串关节角，安全且可复现。

2. **不在 ready 轨迹上**（抓取中间态、停在任务段回位点、刚手动摆过等）：
   先走一段【安全再接近】把臂送回 ready 末点，再逆向 ready 回 pt0。再接近有两条路线，
   都坚持"全部算完、校验通过才开始运动"：
     路线A（首选，抓取前段的镜像）：竖直上中转平面 -> 同高横移到 ready 末点正上方
       -> 竖直进 ready 末点。中转平面 = 桌面 + 250mm，与主业务抓取同一个高度；
       全程「外部 IK 分段 + MoveJ」，每点多种子逆解，近奇异窄点还会二分加密。
     路线B（A 不可行时）：竖直上中转平面后，关节空间整段插到 ready 末点【记录臂型】，
       并用驱动正解逐点校验中间位形（工具离桌面高度、水平半径、单步位移）。
       为什么需要 B：实测右臂从中央抓取位直线横移回来时，中段会撞上逆解奇异区
       （某点不收敛，过了它又跳到相差 234° 的另一臂型）；关节插值逐关节单调、
       必然落在记录臂型上，再用正解确认整段都在桌面安全面之上。

终点 pt0 是框架 join_to_start 认定的「已在起点」，所以回退后重跑 nut_pick_place 会安全地
重放 ready，而不会从桌面上的姿态做无避障盲动。

用法（zsh，工作区根目录；真机前先 source ROS 与 install/setup.zsh）
-----------------------------------------------------------------
    /usr/bin/python3 tools/nut_return.py                # dry-run：打印路线/关节差/IK 预检
    /usr/bin/python3 tools/nut_return.py --execute      # 真机：自动上使能 -> 张手 -> 回退
    /usr/bin/python3 tools/nut_return.py --execute --arm left
    /usr/bin/python3 tools/nut_return.py --execute --blind-join   # 不走再接近，直接慢速 MoveJ 接 ready 末点

安全约定
--------
- 默认 dry-run；`--execute`（或 `--apply`）才动真机。
- 已在安全起始位（差 <= return.tolerance）则该臂一步都不动，脚本可反复执行。
- 不在 ready 轨迹上时默认走安全再接近（竖直上中转平面 -> 同高横移 -> 竖直进 ready 末点，
  全程 IK 预检 + 驱动正解复核；直线撞逆解奇异区时改走关节空间整段插值，正解逐点校验）。
  任何一点不过就在运动前中止，臂一行都不动。
  `--blind-join` 改为直接慢速 MoveJ 接 ready 末点：快但没有避障，只在确认现场空旷时用。
- 一臂运动时另一臂必须停住，漂移超 motion.other_tolerance 立即中止防干涉。
- 自动上使能，跑完【保持使能】；不自动掉使能，避免臂失去支撑。
- 再接近/逆向复现的都是【录制时】的几何，没有碰撞规划；工作区布置变了要重新核对。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nut_robot import DEFAULT_CONFIG, TaskConfig, TaskError  # noqa: E402
from nut_sequences import SequenceRunner, load_leg  # noqa: E402

ARM_CN = {'left': '左', 'right': '右'}
MAX_BRANCH_DEG = 60.0        # 再接近逐段最大单关节步进（超过视为换臂型，中止）


# ---------------- 纯函数（离线可测） -----------------------------------------

def reverse_path(joints):
    """记录段关节序列 (N,7) -> 逆向回放序列：末点在前、起点（安全位）在最后。"""
    q = np.asarray(joints, float)
    if q.ndim != 2 or q.shape[1] != 7:
        raise TaskError(f'逆向回放需要 (N,7) 关节序列，收到形状 {q.shape}')
    if len(q) < 2:
        raise TaskError(f'逆向回放至少要 2 个点，当前 {len(q)} 个')
    if not np.all(np.isfinite(q)):
        raise TaskError('关节序列含非有限值，记录文件可能损坏')
    return q[::-1].copy()


def max_joint_diff(a, b):
    """两组关节角的最大逐关节绝对差（rad）。"""
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def nearest_index(path, current):
    """current 在逆向路径上的最近点下标，及到该点的最大关节差。"""
    diffs = [max_joint_diff(current, q) for q in path]
    idx = int(np.argmin(diffs))
    return idx, float(diffs[idx])


def plan_arm_return(joints, current, tolerance, allow_blind_join=False):
    """纯函数：给定 ready 段关节序列与当前关节角，决定这只臂走哪条路线。

    返回 dict，action 取值：
      already_home - 已在安全起始位（pt0），本臂不运动；
      reverse      - 从 start_index 起沿逆向路径逐点回放到 pt0；
      blind_join   - 先在轨迹外，直接慢速 MoveJ 接 ready 末点再整段逆向（start_index=0）；
      refuse       - 不在轨迹上、且未允许盲接入（上层会改成「安全再接近」或中止）。
    """
    path = reverse_path(joints)
    target = path[-1]
    cur = None if current is None else np.asarray(current, float)
    plan = {'path': path, 'target': target, 'current': cur,
            'home_diff': None, 'nearest_index': None, 'nearest_diff': None,
            'steps': len(path) - 1}
    if cur is None:
        plan['action'] = 'refuse'
        plan['reason'] = '没有收到该臂关节反馈'
        return plan
    if cur.shape != (7,):
        plan['action'] = 'refuse'
        plan['reason'] = f'关节反馈长度 {cur.shape}，不是 7'
        return plan
    home_diff = max_joint_diff(cur, target)
    idx, ndiff = nearest_index(path, cur)
    plan['home_diff'], plan['nearest_index'], plan['nearest_diff'] = home_diff, idx, ndiff
    if home_diff <= tolerance:
        plan['action'] = 'already_home'
        plan['start_index'] = len(path) - 1
        return plan
    if ndiff <= tolerance:
        plan['action'] = 'reverse'
        plan['start_index'] = idx
        return plan
    if allow_blind_join:
        plan['action'] = 'blind_join'
        plan['start_index'] = 0
        return plan
    plan['action'] = 'refuse'
    plan['reason'] = (f'当前姿态不在 ready 轨迹容差 {tolerance:.2f} rad 内'
                      f'（离逆向链最近点 Δ={ndiff:.3f} rad，离安全位 Δ={home_diff:.3f} rad）')
    return plan


def _slerp(q0, q1, t):
    """四元数球面线性插值（xyz w 顺序，自动处理符号与近平行）。"""
    a = np.asarray(q0, float)
    b = np.asarray(q1, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = float(np.arccos(max(-1.0, min(1.0, dot))))
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / s) * a + (np.sin(t * theta) / s) * b


def build_reapproach_path(cur_xyz, goal_xyz, transit_z, eps=1e-3):
    """再接近折线：当前 -> (竖直上)中转平面 -> (同高横移)goal 正上方 -> (竖直进)goal。

    当前已经高于中转平面时不做「上」，直接在同高横移；完全重合的点会被去掉。
    """
    cur = np.asarray(cur_xyz, float)
    goal = np.asarray(goal_xyz, float)
    pts = [cur]
    if cur[2] < transit_z - 0.01:
        pts.append(np.array([cur[0], cur[1], transit_z]))
    pts.append(np.array([goal[0], goal[1], transit_z]))
    pts.append(goal)
    out = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) > eps:
            out.append(p)
    return out


def polyline_samples(points, q_start, q_goal, step=0.025, step_lateral=None):
    """折线路径采样：位置按段等分，姿态沿【整条弧长】做 slerp（转腕摊到全程）。

    竖直为主的段用 step（默认 25mm），横向段用 step_lateral（默认 2*step）。
    返回 [(xyz, quat), ...]，不含起点、含终点。姿态在整个过程中平滑过渡，
    避免在某一段里一次性拧腕。
    """
    step = float(step)
    step_lateral = float(step_lateral) if step_lateral else step * 2.0
    pts = [np.asarray(p, float) for p in points]
    segs, total = [], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.linalg.norm(b - a))
        if d > 1e-9:
            segs.append((a, b, d))
            total += d
    if total <= 1e-9:
        return []
    out, done = [], 0.0
    for a, b, d in segs:
        vertical = abs(b[2] - a[2]) >= 0.5 * d
        step_use = step if vertical else step_lateral
        n = max(1, int(np.ceil(d / step_use)))
        for i in range(1, n + 1):
            t = (done + d * (i / n)) / total
            out.append((a + (b - a) * (i / n), _slerp(q_start, q_goal, t)))
        done += d
    return out


def pick_transit_z(ik_check_fn, x, y, ideal_z, floor_z, step=0.025):
    """从理想中转高度逐级下降，取第一个逆解可达的高度。

    ik_check_fn(point) -> (ok, 种子说明)。返回 (z|None, 试了几档, 命中说明)。
    """
    z = float(ideal_z)
    tried = 0
    while z >= float(floor_z) - 1e-9:
        ok, who = ik_check_fn(np.array([x, y, z], float))
        tried += 1
        if ok:
            return z, tried, who
        z -= float(step)
    return None, tried, None


# ---------------- 安全再接近（不在 ready 轨迹上时） ---------------------------

def joint_lerp_samples(q_from, q_to, max_step_rad):
    """关节空间等分插值：每步每个关节都不超过 max_step_rad。返回 [q1..qn]（不含起点）。

    关节插值的好处是逐关节单调、必然落在记录臂型上（不会有换臂型），代价是中间位形
    没有解析式 —— 所以调用方必须再用驱动正解校验整段离桌面的高度。
    """
    a = np.asarray(q_from, float)
    b = np.asarray(q_to, float)
    span = float(np.max(np.abs(b - a)))
    n = max(1, int(np.ceil(span / float(max_step_rad))))
    return [a + (b - a) * (i / n) for i in range(1, n + 1)]


def check_joint_path_clearance(fk_fn, joints_list, table_z, min_clear,
                               max_xy_radius=None, max_step=None):
    """用正解校验一串关节位形：每点工具高度都要在桌面安全面之上。

    fk_fn(joints) -> (xyz|None, euler|None)。返回 (是否通过, 报告 dict)。
    这是关节空间整段回退的安全闸门：中间位形算不出来（正解失败）或低于安全面，
    一律判不通过，调用方在【运动前】中止。
    """
    zs, pts, wild = [], [], []
    prev = None
    for i, q in enumerate(joints_list, 1):
        xyz, _e = fk_fn(q)
        if xyz is None:
            return False, {'fail': f'第 {i}/{len(joints_list)} 个关节位形正解失败',
                           'z_min': None}
        z = float(xyz[2])
        zs.append(z)
        pts.append(xyz)
        if z < table_z + min_clear:
            return False, {'fail': f'第 {i}/{len(joints_list)} 个关节位形工具 z={z * 1000:+.0f}mm '
                                    f'低于安全面 {table_z + min_clear:+.0f}mm',
                           'z_min': z, 'bad_xyz': xyz}
        if max_xy_radius is not None and float(np.hypot(xyz[0], xyz[1])) > max_xy_radius:
            return False, {'fail': f'第 {i}/{len(joints_list)} 个关节位形工具水平半径 '
                                   f'{float(np.hypot(xyz[0], xyz[1])) * 1000:.0f}mm 超出 '
                                   f'{max_xy_radius * 1000:.0f}mm',
                           'z_min': z, 'bad_xyz': xyz}
        if prev is not None and max_step is not None:
            d = float(np.linalg.norm(xyz - prev))
            if d > max_step:
                wild.append((i, d))
        prev = xyz
    return True, {'z_min': min(zs) if zs else None, 'z_max': max(zs) if zs else None,
                  'steps': len(zs), 'wild': wild}


class _RouteFailed(Exception):
    """某条再接近路线在预检阶段判定不可行（此时一行都没动，可以换路线）。"""


def _set_rise(ctx, z):
    """把"竖直升到高度 z"的折线与采样写进 ctx（已在 z 之上就不升）。"""
    pts = [np.asarray(ctx['cur'], float)]
    if ctx['cur'][2] < z - 0.01:
        pts.append(np.array([ctx['cur'][0], ctx['cur'][1], z], float))
    ctx['rise_points'] = pts
    ctx['rise_samples'] = polyline_samples(pts, ctx['cur_q'], ctx['cur_q'], ctx['step'])


def _reapproach_ctx(robot, ready_leg, cfg, seeds, seed_names):
    """两条再接近路线的公共前置：读姿态/目标、定安全高度、规划竖直上升段。

    中转平面取三者的最高处：理想高度(桌面+clearance)、当前高度、ready 末点高度，
    但不低于"桌面 + joint_min_clear"的硬下限。抬得越高越远离桌面，但也越容易撞上
    逆解奇异/够不着（实测），所以不主动抬高。
    """
    if robot.pose is None:
        raise _RouteFailed('没有末端位姿反馈')
    step = float(cfg.return_reapproach_step)
    floor_z = float(cfg.return_table_z) + float(cfg.return_joint_min_clear)
    cur_xyz = np.asarray(robot.pose[0], float)
    goal_xyz = np.asarray(ready_leg.poses[-1]['xyz'], float)
    ideal_z = max(float(cfg.return_table_z) + float(cfg.return_transit_clearance),
                  float(cur_xyz[2]), float(goal_xyz[2]), floor_z)
    ctx = {'cur': cur_xyz, 'cur_q': np.asarray(robot.pose[1], float),
           'goal': goal_xyz,
           'goal_q': Rot.from_euler('xyz',
                                    np.radians(ready_leg.poses[-1]['eul_deg'])).as_quat(),
           'ready_end': np.asarray(ready_leg.joints[-1], float),
           'seeds': [np.asarray(x, float) for x in (seeds or []) if x is not None],
           'seed_names': list(seed_names or []),
           'step': step, 'floor_z': floor_z, 'ideal_z': ideal_z,
           'transit_z': ideal_z, 'tried': None, 'seed_note': None}
    _set_rise(ctx, ideal_z)
    return ctx


def _solve_polyline(ctx, robot, cfg, base, seed_goal_last=False):
    """按折线逐点多种子逆解，返回 (关节序列, 单步最大角° , 二分加密次数)。

    某点不收敛就在"上一个已解点 -> 该点"之间二分加密（最细 1/32 步长）：能穿过的窄
    死区就穿过去，加密到底仍不收敛、或解跳到别的臂型分支，都判这条路线不可行。
    seed_goal_last：最后一个点是 ready 末点本身时（路线A 的下降终点），用记录臂型当
    首种子，保证落在录制的那一支上；路线B 的上升段终点不是 ready 末点，不能这么种。
    """
    solved, max_deg = [], 0.0
    prev = None if robot.joints is None else np.asarray(robot.joints, float)
    last_pt = [np.asarray(ctx['rise_points'][0], float), np.asarray(ctx['cur_q'], float)]
    seeds, ready_end = ctx['seeds'], ctx['ready_end']

    def _solve_one(xyz, q, is_last):
        eul = Rot.from_quat(q).as_euler('xyz')
        extra = ([ready_end] if (is_last and seed_goal_last) else [])
        extra += ([prev] if prev is not None else []) + seeds
        j, _u = robot.ik_solve(xyz, eul, extra_seeds=extra,
                               prefer_current=prev is None)
        return j if (j is not None and len(j) == 7) else None

    def _push(joints):
        nonlocal prev, max_deg
        if prev is not None:
            deg = float(np.degrees(max_joint_diff(joints, prev)))
            if deg > MAX_BRANCH_DEG:
                raise _RouteFailed(
                    f'第 {len(solved) + 1} 个插值点相对上一点单关节要跳 {deg:.0f}° > '
                    f'{MAX_BRANCH_DEG:.0f}°，判定换臂型')
            max_deg = max(max_deg, deg)
        solved.append([float(v) for v in joints])
        prev = np.asarray(joints, float)

    refined = 0
    idx = 0
    while idx < len(base):
        xyz, q = base[idx]
        is_last = idx == len(base) - 1
        joints = _solve_one(xyz, q, is_last)
        if joints is not None:
            _push(joints)
            last_pt = [np.asarray(xyz, float), np.asarray(q, float)]
            idx += 1
            continue
        d = float(np.linalg.norm(xyz - last_pt[0]))
        trial, ok = None, False
        for k in (2, 4, 8, 16, 32):
            sub = polyline_samples([last_pt[0], np.asarray(xyz, float)],
                                   last_pt[1], np.asarray(q, float),
                                   max(d / k, 1e-4), max(d / k, 1e-4))
            trial, ok = [], True
            for sx, sq in sub:
                sj = _solve_one(sx, sq, is_last)
                if sj is None or (prev is not None and
                                  float(np.degrees(max_joint_diff(sj, prev)))
                                  > MAX_BRANCH_DEG):
                    ok = False
                    break
                trial.append(sj)
                prev = np.asarray(sj, float)     # 段内先按新解递推，失败则整段丢弃
            if ok:
                break
            prev = np.asarray(solved[-1], float) if solved else prev
        if not ok:
            raise _RouteFailed(
                f'第 {idx + 1}/{len(base)} 个插值点 {np.round(xyz * 1000, 1)}mm 逆解不收敛，'
                f'二分加密到 {d / 32 * 1000:.1f}mm 仍失败，判定不可达')
        prev = np.asarray(solved[-1], float)
        for sj in trial:
            _push(sj)
        refined += 1
        last_pt = [np.asarray(xyz, float), np.asarray(q, float)]
        idx += 1
    return solved, max_deg, refined


def _fk_gate(robot, cfg, joints_list, where):
    """对一段关节目标做驱动正解校验；不通过抛 _RouteFailed（此时一行都没动）。"""
    if not joints_list:
        return {'steps': 0, 'z_min': None, 'wild': []}
    ok, rep = check_joint_path_clearance(
        robot.fk_solve, joints_list, float(cfg.return_table_z),
        float(cfg.return_joint_min_clear), max_xy_radius=0.8,
        max_step=float(cfg.return_joint_max_tool_step))
    if not ok:
        raise _RouteFailed(f'{where}驱动正解校验不通过：{rep["fail"]}')
    if rep['wild']:
        i, d = rep['wild'][0]
        raise _RouteFailed(
            f'{where}驱动正解校验发现近奇异位形：第 {i} 个中间位形工具一步窜 '
            f'{d * 1000:.0f}mm > {float(cfg.return_joint_max_tool_step) * 1000:.0f}mm')
    return rep


def _route_cartesian(ctx, robot, cfg):
    """路线 A：竖直上中转平面 -> 同高横移 -> 竖直进 ready 末点（抓取前段的镜像）。

    中转平面要在 ready 末点正上方可用，故先做逐级下降的可达性探测（够不着就降）。
    """
    goal_eul = Rot.from_quat(ctx['goal_q']).as_euler('xyz')

    def _probe(pt):
        ok, used = robot.ik_check(pt, goal_eul, extra_seeds=ctx['seeds'])
        who = (ctx['seed_names'][used] if (ok and used is not None
                                           and used < len(ctx['seed_names'])) else None)
        return ok, who or '默认种子'

    transit_z, tried, who = pick_transit_z(_probe, ctx['goal'][0], ctx['goal'][1],
                                          ctx['ideal_z'], ctx['floor_z'], ctx['step'])
    if transit_z is None:
        raise _RouteFailed(
            f'ready 末点正上方自 {ctx["ideal_z"] * 1000:+.0f}mm 逐级下降 {tried} 档到下限 '
            f'{ctx["floor_z"] * 1000:+.0f}mm（桌面 {cfg.return_table_z * 1000:+.0f}mm 上方 '
            f'{float(cfg.return_joint_min_clear) * 1000:.0f}mm）逆解全部失败')
    ctx['transit_z'], ctx['tried'], ctx['seed_note'] = transit_z, tried, who
    _set_rise(ctx, transit_z)

    points = build_reapproach_path(ctx['cur'], ctx['goal'], transit_z)
    base = polyline_samples(points, ctx['cur_q'], ctx['goal_q'], ctx['step'])
    if not base:
        raise _RouteFailed('当前已在 ready 末点，没有可走的插值点')
    solved, max_deg, refined = _solve_polyline(ctx, robot, cfg, base,
                                               seed_goal_last=True)
    gap = max_joint_diff(solved[-1], ctx['ready_end'])
    if gap > 0.20:
        raise _RouteFailed(f'末点逆解与 ready 段记录臂型差 {gap:.3f} rad(>0.20)，继续回放会抡臂')
    # 竖直上升段是逐个 25mm 的直线，扫掠天然有界；横移/下降段再用正解复核一遍工具轨迹
    rep = _fk_gate(robot, cfg, solved[len(ctx['rise_samples']):], '路线A(同高横移)')
    return {'route': 'cartesian', 'points': points, 'samples': base,
            'joints': solved, 'max_step_deg': max_deg, 'branch_gap': gap,
            'fk': rep, 'base_steps': len(base), 'refined': refined}


def _route_joint(ctx, robot, cfg):
    """路线 B：竖直上中转平面 -> 关节空间整段插到 ready 末点【记录臂型】，全程正解校验。

    路线 A 的直线横移有撞上逆解奇异区的风险（实测右臂中央抓取位回退时，中段某点不收敛、
    过了它又跳到相差 234° 的另一臂型）。关节插值逐关节单调、必然落在记录臂型上、
    不会有换臂型，代价是中间位形没有解析式 —— 所以每个中间位形都用驱动正解算出工具
    位置，确认整段都在桌面安全面之上、且不出现"一步窜很远"的近奇异位形。
    """
    rise_solved, max_deg, _ = _solve_polyline(ctx, robot, cfg,
                                              list(ctx['rise_samples']))
    q_from = (np.asarray(rise_solved[-1], float) if rise_solved
              else (np.asarray(robot.joints, float) if robot.joints is not None
                    else ctx['ready_end']))
    step_rad = np.radians(float(cfg.return_joint_step_deg))
    lerp = joint_lerp_samples(q_from, ctx['ready_end'], step_rad)
    rep = _fk_gate(robot, cfg, lerp, '路线B(关节整段) ')
    return {'route': 'joint', 'points': ctx['rise_points'],
            'samples': [(np.asarray(p, float), None) for p in ctx['rise_points']],
            'joints': rise_solved + [np.asarray(q, float).tolist() for q in lerp],
            'max_step_deg': max(max_deg, float(np.degrees(step_rad))),
            'branch_gap': max_joint_diff(lerp[-1] if lerp else q_from, ctx['ready_end']),
            'fk': rep, 'lerp_steps': len(lerp),
            'lerp_deg': float(cfg.return_joint_step_deg),
            'base_steps': len(ctx['rise_samples']), 'refined': 0}


def plan_reapproach(robot, ready_leg, cfg, seeds=None, seed_names=None):
    """规划「安全再接近 ready 末点」的整条关节路径（只算不动）。

    先试路线 A（镜像抓取前段：竖直上中转平面 -> 同高横移 -> 竖直进 ready 末点，
    全程外部 IK 预检）；A 不可行（逆解奇异区/换臂型）就退到路线 B（关节空间整段插值
    到记录臂型，全程驱动正解校验离桌高度）。两条都不行就抛 TaskError —— 此时一行都没动。
    """
    ctx = None
    try:
        ctx = _reapproach_ctx(robot, ready_leg, cfg, seeds, seed_names)
    except _RouteFailed as exc:
        raise TaskError(f'{robot.arm} 臂安全再接近不可行（臂未移动）：{exc}')
    try:
        rp = _route_cartesian(ctx, robot, cfg)
        rp['route_note'] = None
    except _RouteFailed as exc_a:
        try:
            rp = _route_joint(ctx, robot, cfg)
            rp['route_note'] = str(exc_a)
        except _RouteFailed as exc_b:
            raise TaskError(
                f'{robot.arm} 臂安全再接近：两条路线都不可行，已在运动前中止（臂未移动）。'
                f'路线A(同高横移) {exc_a}；路线B(关节整段) {exc_b}。'
                f'请核对工具坐标系/现场布置，或人工把该臂摆到 ready 轨迹附近再跑本脚本')
    rp.update({'cur': ctx['cur'], 'cur_q': ctx['cur_q'], 'goal': ctx['goal'],
               'goal_q': ctx['goal_q'], 'transit_z': ctx['transit_z'],
               'rise_points': ctx['rise_points'], 'tried': ctx['tried'],
               'seed_note': ctx['seed_note']})
    return rp


# ---------------- 渲染 -------------------------------------------------------

def _fmt_xyz(xyz_m):
    p = np.asarray(xyz_m, float) * 1000.0
    return f'({p[0]:+.0f},{p[1]:+.0f},{p[2]:+.0f})mm'


def _joint_gap_text(current, target, joint_names=None, prefix=None):
    """列出 current->target 差 >0.05rad 的关节（多的在前）；没有则说已在容差内。"""
    names = list(joint_names) if joint_names else [f'j{i}' for i in range(7)]
    diff = np.abs(np.asarray(target, float) - np.asarray(current, float))
    hot = sorted(((names[i], float(diff[i])) for i in range(len(diff)) if diff[i] > 0.05),
                 key=lambda x: -x[1])
    body = ('，'.join(f'{nm} {v:.2f}rad/{np.degrees(v):.1f}°' for nm, v in hot)
            if hot else f'全部关节差 <=0.05rad（最大 {float(diff.max()):.3f}rad）')
    return f'{prefix}：{body}' if prefix else body


def render_arm_plan(arm, leg, plan, tolerance, speed, acce, joint_names=None):
    """把一只臂的回退计划渲染成 dry-run 文本行（纯函数，离线可测）。"""
    cn = ARM_CN.get(arm, arm)
    n = len(leg.joints)
    path = plan['path']
    poses = list(leg.poses)[::-1]
    out = [f'[{cn}臂] {leg.target}（{n} 点，{leg.file}）']
    out.append('  逆向回放链：' + ' -> '.join(f'pt{n - 1 - i}' for i in range(len(path))))
    for i in range(len(path)):
        out.append(f'     第 {i} 步 pt{n - 1 - i} {_fmt_xyz(poses[i]["xyz"])}')
    if plan['current'] is not None and plan['home_diff'] is not None:
        out.append(f'  当前姿态：离安全位 pt0 Δ={plan["home_diff"]:.3f} rad；'
                   f'离逆向链最近点第 {plan["nearest_index"]} 步 Δ={plan["nearest_diff"]:.3f} rad')
    act = plan['action']
    if act == 'already_home':
        out.append(f'  → 已在安全起始位（Δ <= {tolerance:.2f} rad）：本臂不运动')
    elif act == 'reverse':
        out.append(f'  → 从逆向链第 {plan["start_index"]} 步继续，逐点 MoveJ'
                   f'（{speed:g}/{acce:g}）到 pt0')
    elif act == 'blind_join':
        out.append('  ⚠ 当前姿态不在这条轨迹上：按 --blind-join 先低速 MoveJ 到 ready 末点，'
                   '再整段逆向；这段是无避障盲动，可能扫到桌面')
    else:
        out.append(f'  → 不在 ready 轨迹上：{plan["reason"]}')
        if plan['current'] is not None and plan['nearest_index'] is not None:
            out.append('     ' + _joint_gap_text(
                plan['current'], path[plan['nearest_index']], joint_names,
                f'相对逆向链最近点（第 {plan["nearest_index"]} 步）'))
    if act in ('reverse', 'blind_join') and plan['current'] is not None:
        out.append('  到 pt0 需要明显转动的关节（>0.05rad）：'
                   + _joint_gap_text(plan['current'], path[-1], joint_names, None))
    return out


def render_reapproach(arm, rp, cfg):
    """渲染「安全再接近」方案（纯函数，离线可测）。"""
    cn = ARM_CN.get(arm, arm)
    out = [f'  ⚠ {cn}臂不在 ready 轨迹上 → 先做【安全再接近】(抓取前段的镜像)：']
    out.append(f'     中转平面 z={rp["transit_z"] * 1000:+.0f}mm'
               f'（桌面 {cfg.return_table_z * 1000:+.0f}mm + '
               f'{cfg.return_transit_clearance * 1000:.0f}mm'
               + (f'；自 {cfg.return_reapproach_step * 1000:.0f}mm 步长降 '
                  f'{rp["tried"]} 档命中，种子：{rp["seed_note"]}）'
                  if rp.get('tried') else '；路线B 不需要在末点正上方探测可达性）'))
    if rp['route'] == 'cartesian':
        pts = rp['points']
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:]), 1):
            d = float(np.linalg.norm(b - a))
            kind = '竖直' if abs(b[2] - a[2]) >= 0.5 * d else '横移'
            out.append(f'     第 {i} 段 {kind} {_fmt_xyz(a)} -> {_fmt_xyz(b)}  '
                       f'距离 {d * 1000:.0f}mm')
        out.append(f'     路线A(同高横移)：共 {len(rp["joints"])} 个关节目标点，逆解全部通过'
                   f'（单步最大 {rp["max_step_deg"]:.1f}°），末点与记录臂型差 '
                   f'{rp["branch_gap"]:.3f} rad')
        if rp.get('refined'):
            out.append(f'     其中 {rp["refined"]} 处近奇异点已二分加密（基准 '
                       f'{rp["base_steps"]} 点），确认能穿过去才继续')
        fk = rp['fk']
        if fk.get('z_min') is not None:
            out.append(f'     驱动正解复核横移/下降段：{fk["steps"]} 个位形工具最低 '
                       f'z={fk["z_min"] * 1000:+.0f}mm（桌面以上 '
                       f'{(fk["z_min"] - cfg.return_table_z) * 1000:.0f}mm >= 要求 '
                       f'{cfg.return_joint_min_clear * 1000:.0f}mm），无一步窜动'
                       f' —— 全部算完才开始运动')
    else:
        rise = rp['rise_points']
        if len(rise) > 1:
            out.append(f'     第 1 段 竖直上到中转平面 {_fmt_xyz(rise[1])}')
        out.append(f'     第 {len(rise)} 段 关节空间整段插到 ready 末点【记录臂型】：'
                   f'{rp["lerp_steps"]} 步，每步单关节 <= {rp["lerp_deg"]:.1f}°')
        fk = rp['fk']
        clear = (fk['z_min'] - cfg.return_table_z) * 1000
        out.append(f'     路线B(关节整段) 驱动正解校验：{fk["steps"]} 个中间位形工具最低 '
                   f'z={fk["z_min"] * 1000:+.0f}mm（桌面以上 {clear:.0f}mm >= 要求 '
                   f'{cfg.return_joint_min_clear * 1000:.0f}mm），半径 <=800mm，'
                   f'无一步窜动 —— 全部算完才开始运动')
        out.append(f'     （路线A 不可行，故改走 B：{rp["route_note"]}）')
    out.append('     然后从 ready 末点逆向回放整段 ready 到 pt0')
    return out


# ---------------- 真机执行 ---------------------------------------------------

def _run_joint_sequence(runner, rc, other, joints_list, tag, cfg, speed, acce):
    """逐点 MoveJ 一段关节目标；每点前校验反馈新鲜与另一臂静止。"""
    other_ref = other.joints
    if other_ref is None:
        raise TaskError(f'{rc.arm} 臂{tag}时另一只臂无关节反馈，无法做干涉保护')
    other_ref = list(other_ref)
    n = len(joints_list)
    for i, q in enumerate(joints_list, 1):
        if not rc.feedback_fresh(cfg.state_timeout):
            raise TaskError(f'{rc.arm} 臂{tag}第 {i}/{n} 步前关节反馈过期/无效')
        now = other.joints
        if now is None:
            raise TaskError(f'{rc.arm} 臂{tag}第 {i}/{n} 步前另一只臂反馈丢失')
        drift = max(abs(a - b) for a, b in zip(other_ref, now))
        if drift > cfg.other_tolerance:
            raise TaskError(
                f'{rc.arm} 臂{tag}第 {i}/{n} 步时另一只臂漂移 {drift:.3f} rad > '
                f'{cfg.other_tolerance}，疑似被带动/拖动，中止防干涉')
        diff = rc.joint_diff(q)
        if diff is not None and diff <= cfg.start_tolerance:
            print(f'    [{rc.arm}] 已在{tag}第 {i}/{n} 步容差内（Δ={diff:.3f}），跳过')
            continue
        runner._goto_point(rc, q, f'{tag}第 {i}/{n} 步', speed=speed, acce=acce)
        runner._dwell(cfg.point_dwell_seconds, f'{tag}第 {i} 步到位')


def _run_arm_return(runner, rc, other, plan, cfg, speed, acce):
    """沿逆向路径逐点 MoveJ 回退一只臂（已在容差内的点跳过）。"""
    path = plan['path']
    start = int(plan['start_index'])
    joints_list = [path[i] for i in range(start, len(path))]
    _run_joint_sequence(runner, rc, other, joints_list, '逆向回退', cfg, speed, acce)


def _run(args):
    import rclpy
    from nut_robot import RobotClient

    cfg = TaskConfig(args.config)
    if args.speed <= 0:
        raise TaskError(f'--speed 倍率必须为正，当前 {args.speed}')
    speed = cfg.return_speed * args.speed
    acce = cfg.return_acce * args.speed
    blind = bool(args.blind_join or cfg.return_allow_blind_join)
    open_hand = bool(cfg.return_open_hand and not args.keep_hand)
    arms = list(cfg.return_order) if args.arm == 'both' else [args.arm]

    legs = {}
    for arm in arms:
        spec = cfg.left_ready if arm == 'left' else cfg.right_ready
        if spec is None:
            raise TaskError(
                f'{arm}.ready 未配置：nut_return.py 靠【逆向 ready 轨迹】回退，'
                f'请先在 {cfg.path.name} 里配好 {arm}.ready（纯运动段 file/sequence）。'
                f'没有 ready 段就没有可复现的安全回退路径，本脚本不会盲动')
        legs[arm] = load_leg(arm, spec, cfg.check_other)
    if speed > 0.3 + 1e-9 or acce > 0.3 + 1e-9:
        print(f'  ⚠ 回退速度 {speed:g}/{acce:g} 超过建议上限 0.3 rad/s，首调请降速')

    rclpy.init()
    node = rclpy.create_node('nut_return')
    clients = {a: RobotClient(node, cfg.namespace, a) for a in ('left', 'right')}
    try:
        # dry-run 也要真跑 IK/FK 预检（只算不动），所以两种模式都先确认服务在位
        for arm in arms:
            print(f'等待 {arm} 臂运动/IK/FK 服务...')
            clients[arm].wait_services(5.0)
        # 与 nut_pick_place 一致：先拿反馈（未使能也能读位置）再决定动作/使能
        for arm in sorted(set(arms) | ({'left', 'right'} if args.execute else set())):
            rc = clients[arm]
            if not rc.wait_state(3.0):
                raise TaskError(f'{arm} 臂未收到 joint_states/pose_states 反馈，'
                                f'驱动是否已启动？')

        runner = SequenceRunner(clients['left'], clients['right'], cfg)

        plans, reapproaches = {}, {}
        print()
        print('=' * 72)
        print('安全回退计划（终点 = ready 段起点 pt0 = 开机安全起始位）')
        print('=' * 72)
        for arm in arms:
            rc = clients[arm]
            leg = legs[arm]
            if rc.joint_names is not None and list(rc.joint_names) != list(leg.names):
                raise TaskError(
                    f'{arm} 臂实时关节名顺序与记录 {leg.target} 不一致：'
                    f'{rc.joint_names} vs {leg.names}')
            plan = plan_arm_return(leg.joints, rc.joints, cfg.return_tolerance, blind)
            if plan['action'] == 'refuse' and not blind:
                # 不在 ready 轨迹上：规划「安全再接近」（只算不动，失败即在原地中止）
                rp = plan_reapproach(rc, leg, cfg,
                                     seeds=[leg.joints[0], leg.joints[-1]],
                                     seed_names=[f'{leg.target} pt0', f'{leg.target} 末点'])
                reapproaches[arm] = rp
                plan['action'] = 'reapproach'
                plan['start_index'] = 0
            plans[arm] = plan
            for line in render_arm_plan(arm, leg, plan, cfg.return_tolerance,
                                        speed, acce, rc.joint_names):
                print(line)
            if arm in reapproaches:
                for line in render_reapproach(arm, reapproaches[arm], cfg):
                    print(line)
            print()

        todo = [a for a in arms if plans[a]['action'] in ('reverse', 'blind_join',
                                                          'reapproach')]
        selected = '、'.join(f'{ARM_CN[a]}臂' for a in arms)
        if not todo:
            print(f'[nut_return] {selected}都已在安全起始位，无需运动（可反复执行）。')
            return 0
        if not args.execute:
            extra = ('（上面的 IK/FK 预检是真跑过的）' if reapproaches
                     else '（本路线没有可预检的插值点，只是纯逆向/盲接入）')
            print(f'[dry-run] 未驱动机器人{extra}。确认路线后加 --execute 真机执行。')
            return 0

        print('自动上使能（回退结束保持使能，不自动掉使能）...')
        for arm in arms:
            clients[arm].enable()

        if open_hand:
            for arm in todo:
                rc = clients[arm]
                rc.hand_setup(cfg.hand_speed, cfg.hand_force)
                print(f'  [{ARM_CN[arm]}手] 张开 {cfg.hand_open_vals}（松开可能夹着的螺母）')
                rc.hand_open(cfg.hand_open_vals)
            runner._dwell(0.5, '等手指张开')

        for arm in todo:
            rc = clients[arm]
            other = clients['right' if arm == 'left' else 'left']
            if arm in reapproaches:
                rp = reapproaches[arm]
                print(f'[{ARM_CN[arm]}臂] 安全再接近 ready 末点'
                      f'（{len(rp["joints"])} 点，中转平面 {rp["transit_z"] * 1000:+.0f}mm）...')
                _run_joint_sequence(runner, rc, other, rp['joints'], '安全再接近',
                                    cfg, speed, acce)
            mode = '盲接入 + 逆向' if plans[arm]['action'] == 'blind_join' else '逆向'
            print(f'[{ARM_CN[arm]}臂] {mode}回放 {legs[arm].target} -> pt0 ...')
            _run_arm_return(runner, rc, other, plans[arm], cfg, speed, acce)
            rc.wait_state(0.5)
            tgt = np.asarray(legs[arm].poses[0]['xyz'], float) * 1000.0
            now = None if rc.pose is None else np.asarray(rc.pose[0], float) * 1000.0
            if now is not None:
                print(f'  [{ARM_CN[arm]}] 已回安全起始位，实际 {np.round(now, 1)}mm，'
                      f'位置误差 {float(np.linalg.norm(now - tgt)):.0f}mm')

        print()
        print(f'回退完成：{selected}已回到 ready 段起点（开机安全起始位），使能保持。')
        if args.arm != 'both':
            print(f'注意：本次只回退了 {selected}，另一只臂仍停在原处。')
        print('下次跑 nut_pick_place 会从这一点安全重放 ready，不会从桌面姿态盲动。')
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--arm', choices=('left', 'right', 'both'), default='both',
                   help='只回退一只臂（另一臂须已支撑/使能；仍会做漂移保护）')
    p.add_argument('--execute', '--apply', dest='execute', action='store_true',
                   help='真机上使能并回退；默认只 dry-run 打印（dry-run 也会真跑 IK 预检）')
    p.add_argument('--blind-join', action='store_true',
                   help='不在 ready 轨迹上时不做安全再接近，直接低速 MoveJ 接 ready 末点'
                        '（无避障，可能扫到桌面，只在确认现场空旷时用）')
    p.add_argument('--keep-hand', action='store_true', help='回退前不张开手（默认张开）')
    p.add_argument('--speed', type=float, default=1.0, metavar='N',
                   help='return.speed/acce 的倍率（如 0.5=减半）')
    args = p.parse_args()
    try:
        return _run(args)
    except TaskError as exc:
        print(f'\n[中止] {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\n[中断] 已停止后续下发；在途运动不会自动取消，臂保持当前姿态，'
              '使能未掉。必要时用实体急停。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
