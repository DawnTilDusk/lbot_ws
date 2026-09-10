#!/usr/bin/env python3
"""从一次或多次（手动）采样会话中，提取末端位姿，生成自动标定目标点。

这些位姿都是机械臂【物理上到达过】的；其中“板被高质量看到”的位姿同时保证相机可见。
输出 base_link 下的位置(m)+欧拉角(rad, intrinsic XYZ)。

用法：
    # 单会话（只取板完整可见的高质量位姿）
    python3 tools/calib_gen_waypoints.py recordings/calib_xxx --num 16

    # 多会话合并：位置参数=取高质量可见点；--all-sessions=该会话全部可达位姿都纳入
    #   （角点不足的点会标注 expected: maybe，自动采样时若识别不到会超时跳过）
    python3 tools/calib_gen_waypoints.py \
        recordings/calib_20260908_201333_734438 \
        --all-sessions recordings/calib_20260908_214053_781626 \
        --num 80
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from calib_common import CALIB_DIR, T_from_pose, rotation_angle_between


def _session_tag(session):
    return Path(session).name.replace('calib_', '')


def load_poses(session, min_corners, dmin, dmax, keep_all):
    """返回位姿列表。keep_all=True 时纳入全部可达样本（忽略角点/距离闸），否则只取高质量可见。"""
    events = [json.loads(l) for l in open(Path(session) / 'samples.jsonl', encoding='utf-8') if l.strip()]
    meta = next((e for e in events if e.get('type') == 'metadata'), {})
    tag = _session_tag(session)
    out = []
    for e in events:
        if e.get('type') != 'sample':
            continue
        corners = int(e['board']['corners'])
        dist = float(np.linalg.norm(e['board']['tvec']))
        visible = corners >= min_corners and dmin <= dist <= dmax
        if not keep_all and not visible:
            continue
        T = T_from_pose(e['ee']['position'], e['ee']['orientation_xyzw'])
        out.append({
            'point_id': e['point_id'],
            'session': tag,
            'position': np.array(e['ee']['position'], dtype=float),
            'T': T,
            'dist': dist,
            'corners': corners,
            'visible': visible,
        })
    return meta, out


def dedupe(poses, pos_tol=0.02, rot_tol_deg=5.0):
    """合并近似重复位姿（不同会话采到的同一点），保留角点更多/可见的那个。"""
    kept = []
    for p in sorted(poses, key=lambda x: (not x['visible'], -x['corners'])):
        dup = None
        for k in kept:
            if np.linalg.norm(p['position'] - k['position']) < pos_tol and \
               rotation_angle_between(p['T'][:3, :3], k['T'][:3, :3]) < rot_tol_deg:
                dup = k
                break
        if dup is None:
            kept.append(p)
        else:
            # 合并来源信息，保留质量更好的
            dup['sources'] = dup.get('sources', [dup['session'] + '/' + dup['point_id']])
            dup['sources'].append(p['session'] + '/' + p['point_id'])
            if p['visible'] and not dup['visible'] or p['corners'] > dup['corners']:
                dup['position'], dup['T'], dup['corners'], dup['visible'] = \
                    p['position'], p['T'], p['corners'], (dup['visible'] or p['visible'])
    return kept


def farthest_subsample(poses, num):
    """在“位置+朝向”特征空间里做最远点采样，保证位姿多样。"""
    if len(poses) <= num:
        return poses
    feats = np.array([np.concatenate([p['position'], p['T'][:3, :3].reshape(-1) * 0.35]) for p in poses])
    selected = [0]
    d = np.linalg.norm(feats - feats[0], axis=1)
    for _ in range(num - 1):
        nxt = int(np.argmax(d))
        selected.append(nxt)
        d = np.minimum(d, np.linalg.norm(feats - feats[nxt], axis=1))
    return [poses[i] for i in selected]


def order_nearest(waypoints, start_pos=None):
    """贪心最近邻排序，减少相邻目标点之间的大幅运动。"""
    remaining = list(waypoints)
    if start_pos is None:
        ordered = [remaining.pop(0)]
    else:
        i = int(np.argmin([np.linalg.norm(w['position'] - start_pos) for w in remaining]))
        ordered = [remaining.pop(i)]
    while remaining:
        cur = ordered[-1]['position']
        i = int(np.argmin([np.linalg.norm(w['position'] - cur) for w in remaining]))
        ordered.append(remaining.pop(i))
    return ordered


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sessions', nargs='*', help='采样目录；这些会话取【板完整可见】的高质量位姿')
    p.add_argument('--all-sessions', nargs='*', default=[],
                   help='这些会话的【全部可达位姿】都纳入（角点不足的标注 maybe，自动采样超时会跳过）')
    p.add_argument('--out', type=Path, default=CALIB_DIR / 'auto_waypoints.yaml')
    p.add_argument('--num', type=int, default=80, help='最多目标点数量（去重后超出则最远点采样）')
    p.add_argument('--min-corners', type=int, default=30)
    p.add_argument('--dmin', type=float, default=0.30, help='板距下限 m')
    p.add_argument('--dmax', type=float, default=0.85, help='板距上限 m')
    p.add_argument('--start', type=float, nargs=3, default=None,
                   help='起始末端位置 x y z（用于排序），默认用第一个点')
    args = p.parse_args()

    meta = {}
    poses = []
    for s in args.sessions:
        m, g = load_poses(s, args.min_corners, args.dmin, args.dmax, keep_all=False)
        meta = meta or m
        print(f'[可见] {_session_tag(s)}: {len(g)} 个高质量位姿')
        poses += g
    for s in args.all_sessions:
        m, a = load_poses(s, args.min_corners, args.dmin, args.dmax, keep_all=True)
        meta = meta or m
        nvis = sum(1 for x in a if x['visible'])
        print(f'[全部] {_session_tag(s)}: {len(a)} 个可达位姿（其中 {nvis} 个板完整可见）')
        poses += a

    if len(poses) < 6:
        raise SystemExit(f'合并后仅 {len(poses)} 个位姿，太少。')
    poses = dedupe(poses)
    print(f'去重后 {len(poses)} 个唯一位姿')
    picks = farthest_subsample(poses, min(args.num, len(poses)))
    start = np.array(args.start) if args.start else None
    picks = order_nearest(picks, start)

    wps = []
    for i, g in enumerate(picks, 1):
        eul = Rot.from_matrix(g['T'][:3, :3]).as_euler('xyz')
        wps.append({
            'name': f'wp_{i:02d}',
            'position': [float(round(v, 5)) for v in g['position']],
            'euler': [float(round(v, 5)) for v in eul],
            'source': g.get('sources', [g['session'] + '/' + g['point_id']]),
            'corners_when_sampled': g['corners'],
            'expected': 'visible' if g['visible'] else 'maybe',
            'board_dist_m': round(g['dist'], 3),
        })
    data = {
        'arm': meta.get('arm', 'right'),
        'frame': meta.get('base_frame', 'base_link'),
        'euler_convention': 'intrinsic XYZ (rad)',
        'units': {'position': 'm', 'euler': 'rad'},
        'speed': 0.5,
        'acce': 0.5,
        'note': '自动标定目标点，合并自多次手动会话的真实到达位姿；expected=maybe 的点可能识别不到，超时可跳过。',
        'waypoints': wps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    n_maybe = sum(1 for w in wps if w['expected'] == 'maybe')
    print(f'\n共 {len(wps)} 个目标点（{len(wps)-n_maybe} 个预期可见，{n_maybe} 个可能识别不到→超时跳过），已写入 {args.out}')
    print(f'{"name":<7}{"expected":<9}{"corner":>7}  {"position (m)":<26}{"euler xyz (deg)"}')
    for w in wps:
        ed = np.degrees(w['euler'])
        print(f'{w["name"]:<7}{w["expected"]:<9}{w["corners_when_sampled"]:>7}  '
              f'{str(np.round(w["position"],3)):<26}{np.round(ed,1)}')


if __name__ == '__main__':
    main()
