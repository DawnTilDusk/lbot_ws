#!/usr/bin/env python3
"""微调预录序列里【某个记录点】的末端位置（默认最后一个点）。

回放以关节角为准，所以不是只改 pose 显示字段：用驱动 IK 对新 xyz + 原姿态反解关节，
多种子（本点/上一点关节、逐关节扰动、多级高斯）里挑关节变化最小的同臂型解，再用 FK
复核位置(<=3mm)/姿态(<=0.02rad)，关节角和 pose 快照两份一起回写。只调用 IK/FK
计算服务，不产生任何机器人运动。改前自动备份。

例：左手抓起螺母挪到中间的段，末点下降 0.6cm（dz 负=向下，单位米）：
  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/retune_waypoint.py \
      --file recordings/left_grasp_middle1/events.jsonl \
      --sequence left_middle_grasp_001 --arm left --dz -0.006

  右臂盒位释放段末点下降 1cm：
  /usr/bin/python3 tools/retune_waypoint.py \
      --file recordings/right_middle_back1/events.jsonl \
      --sequence right_grasp_move1_001 --arm right --dz -0.01

  右臂中央重抓末点绕 X(俯仰)多压 5 度（位置不变）：
  /usr/bin/python3 tools/retune_waypoint.py \
      --file recordings/right_middle_grasp1/events.jsonl \
      --sequence right_grasp_middle1_001 --arm right --drx -5

回退：cp <file>.bak_<时间戳> <file>
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def spin_call(node, cli, req, timeout=10.0):
    import rclpy
    fut = cli.call_async(req)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        if fut.done():
            return fut.result()
    return None


def build_seeds(q_rec, q_prev):
    seeds = [q_rec]
    if q_prev is not None:
        seeds.append(q_prev)
    for j in range(7):                      # 逐关节小/中/大扰动
        for s in (0.03, 0.08, 0.18, 0.35):
            d = np.zeros(7); d[j] = s
            seeds += [q_rec + d, q_rec - d]
    rng = np.random.default_rng(1)          # 多级高斯
    for sigma, n in ((0.03, 12), (0.1, 20), (0.25, 25), (0.6, 25)):
        for _ in range(n):
            seeds.append(q_rec + rng.normal(0, sigma, 7))
    return seeds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--file', required=True, type=Path, help='events.jsonl 路径')
    p.add_argument('--sequence', required=True, help='序列 label（如 left_middle_grasp_001）')
    p.add_argument('--arm', required=True, choices=('left', 'right'))
    p.add_argument('--point', type=int, default=None,
                   help='点序号（1 起，默认最后一个点）')
    p.add_argument('--dx', type=float, default=0.0, help='base_link X 平移 m（+前）')
    p.add_argument('--dy', type=float, default=0.0, help='base_link Y 平移 m')
    p.add_argument('--dz', type=float, default=0.0, help='base_link Z 平移 m（+上，下降给负）')
    p.add_argument('--drx', type=float, default=0.0, help='绕 X 欧拉角增量（度）')
    p.add_argument('--dry', type=float, default=0.0, help='绕 Y 欧拉角增量（度）')
    p.add_argument('--drz', type=float, default=0.0, help='绕 Z 欧拉角增量（度，=拧腕朝向）')
    p.add_argument('--max-joint-diff-deg', type=float, default=25.0,
                   help='最近解的最大单关节变化阈值，超过判为翻臂型不写')
    p.add_argument('--suffix', default=None, help='备份后缀（默认 .bak_年月日_时分秒）')
    args = p.parse_args()

    path = args.file
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        sys.exit(f'文件不存在：{path}')
    delta = np.array([args.dx, args.dy, args.dz], float)
    drot_deg = np.array([args.drx, args.dry, args.drz], float)
    if np.linalg.norm(delta) < 1e-9 and np.linalg.norm(drot_deg) < 1e-9:
        sys.exit('平移/旋转增量全为 0，无需修改（用 --dx/--dy/--dz/--drx/--dry/--drz）')

    import rclpy
    from lbot_arm_interfaces.srv import InverseKinematics, ForwardKinematics

    evs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    try:
        seq = next(e for e in evs
                   if e.get('type') == 'sequence' and e.get('label') == args.sequence)
    except StopIteration:
        sys.exit(f'{path.name} 里没有序列 {args.sequence}')
    wp_ids_all = [e['point_id'] for e in evs if e.get('type') == 'waypoint']
    ids = seq['point_ids']
    target_id = ids[args.point - 1] if args.point else ids[-1]
    ti = wp_ids_all.index(target_id)
    wp = next(e for e in evs if e.get('type') == 'waypoint' and e['point_id'] == target_id)
    prev_wp = None
    if ti > 0:
        prev_id = wp_ids_all[ti - 1]
        prev_wp = next(e for e in evs
                       if e.get('type') == 'waypoint' and e['point_id'] == prev_id)

    arm = args.arm
    js = wp['states'][f'{arm}_arm/joint_states']['message']
    q_rec = np.array(js['position'], float)
    q_prev = (np.array(prev_wp['states'][f'{arm}_arm/joint_states']
                       ['message']['position'], float) if prev_wp else None)
    pose = wp['states'][f'{arm}_arm/pose_states']['message']['pose']
    p_rec = np.array([pose['position']['x'], pose['position']['y'],
                      pose['position']['z']])
    q_rec_quat = [pose['orientation']['x'], pose['orientation']['y'],
                  pose['orientation']['z'], pose['orientation']['w']]
    eul0 = Rot.from_quat(q_rec_quat).as_euler('xyz')
    eul_tgt = eul0 + np.radians(drot_deg)
    p_tgt = p_rec + delta

    rclpy.init()
    node = rclpy.create_node('retune_waypoint')
    ik = node.create_client(InverseKinematics, f'/robot1/{arm}_arm/inverse_kinematics')
    fk = node.create_client(ForwardKinematics, f'/robot1/{arm}_arm/forward_kinematics')
    try:
        for cli in (ik, fk):
            if not cli.wait_for_service(timeout_sec=5.0):
                sys.exit(f'服务不可用：{cli.srv_name}（驱动是否启动？）')

        cands, tried = [], 0
        for seed in build_seeds(q_rec, q_prev):
            req = InverseKinematics.Request()
            req.position.x, req.position.y, req.position.z = map(float, p_tgt)
            req.euler.x, req.euler.y, req.euler.z = map(float, eul_tgt)
            req.joints = [float(x) for x in seed]
            res = spin_call(node, ik, req)
            if res is None or not res.success or len(res.joints) != 7:
                continue
            tried += 1
            q_new = np.array(res.joints, float)
            fres = spin_call(node, fk,
                             ForwardKinematics.Request(joints=[float(x) for x in q_new]))
            if fres is None or not fres.success:
                continue
            p_fk = np.array([fres.position.x, fres.position.y, fres.position.z])
            e_fk = np.array([fres.euler.x, fres.euler.y, fres.euler.z])
            perr = float(np.linalg.norm(p_fk - p_tgt))
            q_fk = Rot.from_euler('xyz', e_fk).as_quat()
            oerr = float(2 * np.arccos(min(1.0, abs(float(np.dot(q_fk,
                            Rot.from_euler('xyz', eul_tgt).as_quat()))))))
            dq = np.abs(wrap(q_new - q_rec))
            if perr <= 0.003 and oerr <= 0.02:
                cands.append((dq.max(), q_new, p_fk, q_fk, perr, oerr, dq))

        print(f'== {path.name} {args.sequence} {target_id}（{arm}）')
        print(f'   原 xyz(mm)={np.round(p_rec * 1000, 1)}  '
              f'eul(deg)={np.degrees(eul0).round(2)}')
        print(f'   平移(mm)={np.round(delta * 1000, 1)}  旋转增量(deg)={drot_deg.round(2)}')
        print(f'   目标 xyz(mm)={np.round(p_tgt * 1000, 1)}  目标 eul(deg)='
              f'{np.degrees(eul_tgt).round(2)}')
        print(f'   IK 成功 {tried}，FK 复核通过 {len(cands)}')
        if not cands:
            sys.exit('   !! 无通过 FK 复核的解，文件未改动（加大种子/检查目标是否可达）')
        cands.sort(key=lambda c: c[0])
        score, q_new, p_fk, q_fk, perr, oerr, dq = cands[0]
        print(f'   最近解：新 xyz(mm)={np.round(p_fk * 1000, 1)}  位置误差 {perr*1000:.1f}mm '
              f'姿态误差 {np.degrees(oerr):.2f}°')
        print(f'   关节变化(deg)={np.degrees(dq).round(2)} 最大 {np.degrees(score):.2f}°')
        if np.degrees(score) > args.max_joint_diff_deg:
            sys.exit(f'   !! 最大关节变化 {np.degrees(score):.1f}° > '
                     f'{args.max_joint_diff_deg:.1f}°，疑似翻臂型，不写。'
                     f'确认安全后可用 --max-joint-diff-deg 放宽')

        suffix = args.suffix or time.strftime('.bak_%Y%m%d_%H%M%S')
        bak = path.with_name(path.name + suffix)
        if not bak.exists():
            shutil.copy2(path, bak)
        js['position'] = [float(x) for x in q_new]
        om = wp['states'][f'{arm}_arm/pose_states']['message']['pose']
        om['position'].update(x=float(p_fk[0]), y=float(p_fk[1]), z=float(p_fk[2]))
        om['orientation'].update(x=float(q_fk[0]), y=float(q_fk[1]),
                                 z=float(q_fk[2]), w=float(q_fk[3]))
        path.write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in evs) + '\n')
        print(f'   已写入；备份 {bak.name}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
