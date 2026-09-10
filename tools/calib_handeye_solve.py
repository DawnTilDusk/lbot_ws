#!/usr/bin/env python3
"""Eye-to-Hand 手眼标定求解器。

读取 calib_handeye_sample.py 采集的样本（机械臂末端位姿 B_T_E + 标定板在相机中位姿 C_T_M），
求解 base_link -> camera_color_optical_frame 的外参 B_T_C，写入
开发资源/calibration/gemini2_extrinsics.yaml，并报告残差。

数学：Eye-to-Hand 下，向 cv2.calibrateHandEye 传入 inv(B_T_E)（即 E_T_B）作为
gripper2base、C_T_M 作为 target2cam，返回的 cam2gripper 即 B_T_C。
多种方法各解一次，按闭环残差择优。

用法：
    python3 tools/calib_handeye_solve.py recordings/calib_YYYYmmdd_HHMMSS_ffffff
    python3 tools/calib_handeye_solve.py --selftest          # 无硬件合成数据自检
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from calib_common import (CALIB_DIR, COLOR_CAMERA_INFO_PATH, EXTRINSICS_PATH,
                          T_from_pose, T_from_rvec_tvec, T_to_pose, T_to_rvec_tvec,
                          invert_T, make_T, mat_to_quat_xyzw, rotation_angle_of,
                          read_jsonl)

GRIPPER_TO_BOARD_PATH = CALIB_DIR / 'gemini2_gripper_to_board.yaml'

METHODS = [
    ('TSAI', cv2.CALIB_HAND_EYE_TSAI),
    ('PARK', cv2.CALIB_HAND_EYE_PARK),
    ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD),
    ('ANDREFF', cv2.CALIB_HAND_EYE_ANDREFF),
    ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS),
]


def mean_transform(Ts):
    """平移取均值；旋转用矩阵均值后 SVD 正交化（Procrustes）。"""
    t_mean = np.mean([T[:3, 3] for T in Ts], axis=0)
    R_sum = np.sum([T[:3, :3] for T in Ts], axis=0)
    if not np.isfinite(R_sum).all():
        raise RuntimeError('旋转均值含 NaN/Inf：样本位姿异常或退化（常见于机械臂未运动）。')
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return make_T(R_mean, t_mean)


def compute_residuals(BTE_list, CTM_list, Z):
    """给定 B_T_C=Z，计算末端->板 X 的一致性与板位置闭环重投影误差。"""
    X_list = [invert_T(BTE) @ Z @ CTM for BTE, CTM in zip(BTE_list, CTM_list)]
    X_mean = mean_transform(X_list)

    trans_dev = []   # 各样本 X 平移到均值的距离 mm
    rot_dev = []     # 各样本 X 旋转到均值的角度 deg
    reproj = []      # 用 Z+X_mean 预测板在相机中的位置 vs 实测 mm
    for BTE, CTM, Xi in zip(BTE_list, CTM_list, X_list):
        trans_dev.append(np.linalg.norm(Xi[:3, 3] - X_mean[:3, 3]) * 1000.0)
        rot_dev.append(rotation_angle_of(Xi[:3, :3] @ X_mean[:3, :3].T))
        CTM_pred = invert_T(Z) @ BTE @ X_mean
        reproj.append(np.linalg.norm(CTM_pred[:3, 3] - CTM[:3, 3]) * 1000.0)
    return {
        'X_mean': X_mean,
        'trans_std_mm': float(np.std(trans_dev)),
        'trans_mean_mm': float(np.mean(trans_dev)),
        'trans_max_mm': float(np.max(trans_dev)),
        'rot_mean_deg': float(np.mean(rot_dev)),
        'rot_max_deg': float(np.max(rot_dev)),
        'reproj_mean_mm': float(np.mean(reproj)),
        'reproj_max_mm': float(np.max(reproj)),
    }


def solve_handeye(BTE_list, CTM_list):
    """对每种方法求解并评估，返回 (结果表, 最佳方法名, 最佳 Z, 最佳残差)。"""
    g2b_R, g2b_t, t2c_R, t2c_t = [], [], [], []
    for BTE, CTM in zip(BTE_list, CTM_list):
        ETB = invert_T(BTE)          # eye-to-hand：传入末端->基座
        g2b_R.append(ETB[:3, :3])
        g2b_t.append(ETB[:3, 3].reshape(3, 1))
        t2c_R.append(CTM[:3, :3])
        t2c_t.append(CTM[:3, 3].reshape(3, 1))

    results = {}
    for name, method in METHODS:
        try:
            Rc, tc = cv2.calibrateHandEye(g2b_R, g2b_t, t2c_R, t2c_t, method=method)
        except cv2.error:
            continue
        Z = make_T(Rc, tc.reshape(3))
        res = compute_residuals(BTE_list, CTM_list, Z)
        results[name] = {'Z': Z, 'res': res}

    if not results:
        raise RuntimeError('所有手眼标定方法均失败，检查样本数量/质量。')
    best = min(results, key=lambda k: results[k]['res']['reproj_mean_mm'])
    return results, best, results[best]['Z'], results[best]['res']


def load_session(session_dir):
    """从采样目录读取 metadata 与样本，返回 (meta, BTE_list, CTM_list)。"""
    session_dir = Path(session_dir)
    jsonl = session_dir / 'samples.jsonl'
    if not jsonl.exists():
        raise FileNotFoundError(f'未找到样本文件：{jsonl}')
    events = read_jsonl(jsonl)
    meta = next((e for e in events if e.get('type') == 'metadata'), {})
    BTE_list, CTM_list, id_list = [], [], []
    for e in events:
        if e.get('type') != 'sample':
            continue
        ee = e['ee']
        bd = e['board']
        BTE_list.append(T_from_pose(ee['position'], ee['orientation_xyzw']))
        CTM_list.append(T_from_rvec_tvec(bd['rvec'], bd['tvec']))
        id_list.append(e.get('point_id', f'point_{len(id_list)+1:03d}'))
    if len(BTE_list) < 3:
        raise RuntimeError(f'有效样本仅 {len(BTE_list)} 个，手眼标定至少需要 3~4 个，建议 15 个以上。')

    # 退化检测：机械臂末端位姿必须随样本变化（Eye-to-Hand 靠“臂带动板运动”求解）
    poss = np.array([T[:3, 3] for T in BTE_list])
    pos_spread = float(np.linalg.norm(poss.max(axis=0) - poss.min(axis=0)))
    max_rot = 0.0
    for T in BTE_list[1:]:
        max_rot = max(max_rot, rotation_angle_of(T[:3, :3] @ BTE_list[0][:3, :3].T))
    if pos_spread < 0.01 and max_rot < 1.0:
        raise RuntimeError(
            f'所有样本的机械臂末端位姿几乎不变（位置变化 {pos_spread*1000:.1f} mm，姿态变化 {max_rot:.2f}°）。\n'
            '手眼标定要求机械臂【带动刚性夹持在末端的标定板】运动到不同位姿，机械臂不动则方程退化、无解。\n'
            '常见原因：标定板没有固定到右臂末端而是手持移动，或采样时机械臂没有运动。\n'
            '请把标定板牢固装到右臂末端，用示教/遥操作移动右臂到 15~25 个【位置和姿态都不同】的姿态，'
            '板始终在相机视野内、停稳后再记录，然后重新采样。')
    return meta, BTE_list, CTM_list, id_list


def per_sample_reproj_mm(BTE_list, CTM_list, Z):
    """用当前 Z 与平均 E_T_M 预测板在相机中的位置，返回每样本位置误差(mm)。"""
    res = compute_residuals(BTE_list, CTM_list, Z)
    X_mean = res['X_mean']
    errs = []
    for BTE, CTM in zip(BTE_list, CTM_list):
        pred = invert_T(Z) @ BTE @ X_mean
        errs.append(float(np.linalg.norm(pred[:3, 3] - CTM[:3, 3]) * 1000.0))
    return errs


def robust_solve(BTE_list, CTM_list, id_list, trim_mm=8.0, min_keep=12, max_iter=10):
    """迭代剔除闭环重投影误差大的离群姿态（如板在夹持中打滑），再求解。

    返回 (results, best, Z, res, kept_ids, dropped_ids)。若剔除会导致样本过少，
    则保留当前最优子集。
    """
    idx = list(range(len(BTE_list)))
    dropped = []
    best_state = None
    for _ in range(max_iter):
        BTE = [BTE_list[i] for i in idx]
        CTM = [CTM_list[i] for i in idx]
        results, best, Z, res = solve_handeye(BTE, CTM)
        errs = per_sample_reproj_mm(BTE, CTM, Z)
        med = float(np.median(errs))
        # 阈值：不小于 trim_mm，且随中位数放宽（避免全是好数据时误删）
        thr = max(trim_mm, med * 3.0)
        worst_i = int(np.argmax(errs))
        if errs[worst_i] <= thr or len(idx) <= min_keep:
            best_state = (results, best, Z, res, idx, dropped)
            break
        dropped.append(id_list[idx[worst_i]])
        idx.pop(worst_i)
        best_state = (results, best, Z, res, list(idx), list(dropped))
    results, best, Z, res, idx, dropped = best_state
    kept_ids = [id_list[i] for i in idx]
    return results, best, Z, res, kept_ids, dropped


def write_extrinsics(Z, meta, best, res, out_path, parent_frame, child_frame):
    pos, quat = T_to_pose(Z)
    data = {
        'parent_frame': parent_frame,
        'child_frame': child_frame,
        'translation': [float(v) for v in pos],
        'rotation_xyzw': [float(v) for v in quat],
        'method': best,
        'samples': int(meta.get('samples', 0)),
        'residual_reproj_mean_mm': round(res['reproj_mean_mm'], 4),
        'residual_reproj_max_mm': round(res['reproj_max_mm'], 4),
        'residual_trans_std_mm': round(res['trans_std_mm'], 4),
        'residual_rot_mean_deg': round(res['rot_mean_deg'], 4),
        'units': {'translation': 'm', 'rotation': 'quaternion_xyzw'},
        'note': '现场手眼标定结果；相机/支架移动或分辨率/对齐改变后需重新标定。',
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    return data


def write_camera_info(meta, out_path):
    cam = meta.get('camera_info')
    if not cam:
        return False
    data = {
        'frame_id': cam.get('frame_id', 'camera_color_optical_frame'),
        'width': cam.get('width'),
        'height': cam.get('height'),
        'distortion_model': cam.get('distortion_model', 'plumb_bob'),
        'K': cam.get('K'),
        'D': cam.get('D'),
        'P': cam.get('P'),
        'note': '由采样时 /camera/color/camera_info 保存；改分辨率/对齐后需重新保存。',
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    return True


def print_result_table(results):
    print(f'{"方法":<10}{"闭环重投影均值(mm)":>20}{"最大(mm)":>12}'
          f'{"平移std(mm)":>14}{"旋转均值(°)":>14}')
    for name, r in results.items():
        s = r['res']
        print(f'{name:<10}{s["reproj_mean_mm"]:>20.3f}{s["reproj_max_mm"]:>12.3f}'
              f'{s["trans_std_mm"]:>14.3f}{s["rot_mean_deg"]:>14.4f}')


def write_gripper_to_board(X, arm, out_path):
    """保存末端->标定板常量变换 E_T_M，供 calib_handeye_verify.py 在线交叉核验。"""
    pos, quat = T_to_pose(X)
    data = {
        'parent_frame': f'{arm}_arm end-effector (pose_states frame)',
        'child_frame': 'charuco_board',
        'translation': [float(v) for v in pos],
        'rotation_xyzw': [float(v) for v in quat],
        'units': {'translation': 'm', 'rotation': 'quaternion_xyzw'},
        'note': '末端到标定板的夹持常量；重新夹持/换板后需重新标定。',
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    return out_path


# ---- 合成数据自检 -----------------------------------------------------------

def selftest():
    from scipy.spatial.transform import Rotation as Rot

    def T_from_euler(rpy, t):
        M = np.eye(4)
        M[:3, :3] = Rot.from_euler('xyz', rpy).as_matrix()
        M[:3, 3] = t
        return M

    rng = np.random.default_rng(42)
    Z_true = T_from_euler([0.15, -0.05, 0.2], [0.45, -0.35, 0.85])   # B_T_C
    X_true = T_from_euler([0.1, 0.2, -0.3], [0.0, 0.0, 0.12])        # E_T_M

    BTE_list, CTM_list = [], []
    for _ in range(20):
        rpy = rng.uniform(-0.6, 0.6, 3)
        t = rng.uniform([0.3, -0.4, 0.0], [0.7, 0.4, 0.6])
        BTE = T_from_euler(rpy, t)
        CTM = invert_T(Z_true) @ BTE @ X_true
        # 加测量噪声：板位姿 t 0.5mm，rvec 0.05°
        CTMn = CTM.copy()
        CTMn[:3, 3] += rng.normal(0, 0.0005, 3)
        dR = Rot.from_euler('xyz', rng.normal(0, np.deg2rad(0.05), 3)).as_matrix()
        CTMn[:3, :3] = dR @ CTMn[:3, :3]
        BTE_list.append(BTE)
        CTM_list.append(CTMn)

    results, best, Z, res = solve_handeye(BTE_list, CTM_list)
    print_result_table(results)
    t_err = np.linalg.norm(Z[:3, 3] - Z_true[:3, 3]) * 1000
    r_err = rotation_angle_of(Z[:3, :3] @ Z_true[:3, :3].T)
    print(f'\n择优方法：{best}')
    print(f'外参平移误差：{t_err:.3f} mm（真值 {Z_true[:3,3].round(3)}，估计 {Z[:3,3].round(3)}）')
    print(f'外参旋转误差：{r_err:.4f}°')
    print(f'闭环重投影：均值 {res["reproj_mean_mm"]:.3f} mm，最大 {res["reproj_max_mm"]:.3f} mm')
    ok = t_err < 1.0 and r_err < 0.5
    print('\n自检结果：' + ('通过 ✅（噪声下平移<1mm、旋转<0.5°）' if ok else '未通过 ❌'))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('session', nargs='?', help='采样目录（含 samples.jsonl）')
    p.add_argument('--selftest', action='store_true', help='用合成数据验证数学，无需硬件')
    p.add_argument('--parent-frame', default=None, help='外参父帧，默认取采样 metadata（通常 base_link）')
    p.add_argument('--child-frame', default='camera_color_optical_frame')
    p.add_argument('--out', type=Path, default=EXTRINSICS_PATH, help='外参 YAML 输出路径')
    p.add_argument('--no-trim', action='store_true', help='关闭离群姿态自动剔除')
    p.add_argument('--trim-mm', type=float, default=8.0, help='离群剔除阈值 mm（默认 8）')
    args = p.parse_args()

    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)

    if not args.session:
        p.error('请提供采样目录，或使用 --selftest')

    meta, BTE_list, CTM_list, id_list = load_session(args.session)
    print(f'载入样本 {len(BTE_list)} 个，来自 {args.session}')
    if args.no_trim:
        results, best, Z, res = solve_handeye(BTE_list, CTM_list)
        kept_ids, dropped = id_list, []
    else:
        results, best, Z, res, kept_ids, dropped = robust_solve(
            BTE_list, CTM_list, id_list, trim_mm=args.trim_mm)
    print_result_table(results)
    if dropped:
        print(f'自动剔除离群姿态 {len(dropped)} 个：{", ".join(dropped)}')
        print(f'（保留 {len(kept_ids)}/{len(id_list)} 个；如需关闭剔除用 --no-trim）')
    meta['samples'] = len(kept_ids)

    parent = args.parent_frame or meta.get('base_frame') or 'base_link'
    data = write_extrinsics(Z, meta, best, res, args.out, parent, args.child_frame)
    if write_camera_info(meta, COLOR_CAMERA_INFO_PATH):
        print(f'已保存彩色相机内参：{COLOR_CAMERA_INFO_PATH}')
    gb_path = write_gripper_to_board(res['X_mean'], meta.get('arm', 'right'), GRIPPER_TO_BOARD_PATH)
    print(f'已保存末端->标定板变换：{gb_path}（供在线验证）')

    pos, quat = T_to_pose(Z)
    print(f'\n择优方法：{best}')
    print(f'外参 {parent} -> {args.child_frame}')
    print(f'  translation (m)   = [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]')
    print(f'  rotation_xyzw     = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]')
    print(f'  闭环重投影误差：均值 {res["reproj_mean_mm"]:.3f} mm，最大 {res["reproj_max_mm"]:.3f} mm')
    print(f'  末端->板一致性：平移 std {res["trans_std_mm"]:.3f} mm，旋转均值 {res["rot_mean_deg"]:.4f}°')
    print(f'\n已写入外参文件：{args.out}')
    print('发布静态 TF：python3 tools/calib_handeye_publish_tf.py')
    if res['reproj_mean_mm'] > 5 or res['rot_mean_deg'] > 1.0:
        print('\n⚠️  残差偏大（重投影 >5mm 或旋转 >1°），外参不建议直接用于抓取。常见原因：')
        print('   1) 标定板未刚性固定在臂末端（胶带/手指柔性夹持会在运动中晃动）——最常见；')
        print('   2) 采样时未完全停稳、运动模糊；')
        print('   3) 板距过远/接近正对相机导致板姿态估计弱（应 0.3~0.6m、倾斜 20~50°、铺满视野）；')
        print('   4) 板方格尺寸未按实测回填。建议刚性固定后重新采样。')
    else:
        print('\n✅ 残差良好，外参可用。建议再用 calib_handeye_verify.py 在线核验。')


if __name__ == '__main__':
    main()
