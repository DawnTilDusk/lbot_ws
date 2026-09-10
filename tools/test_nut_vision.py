#!/usr/bin/env python3
"""nut_vision 合成数据离线单测（不需要 ROS/相机/机械臂）。

  cd tools
  /usr/bin/python3 -m unittest -v test_nut_vision.py

思路：640x480、fx=fy=600、工作距离约 0.6m（≈1px/mm），反渲染一张「倾斜纸面 +
3 种尺寸六边凸起（螺纹孔回退纸面深度）+ 黑框」的对齐深度/彩图；另外构造
噪声、相触、固定阈值分类等用例。
"""
import unittest

import cv2
import numpy as np

from nut_vision import (DEFAULT_PARAMS, LABELS, _plane_basis,
                        depth_to_meters, run_pipeline)

W, H = 640, 480
FX = FY = 600.0
CX, CY = W / 2, H / 2
Z0 = 0.6

# (ac, bc, flats_m, thick_m, phi)  平面坐标 ac/bc 米，对边距/厚度米，朝向角
NUTS_DEF = [
    (-0.070, -0.040, 0.010, 0.0052, 0.3),   # 小
    (0.050, -0.055, 0.010, 0.0052, 0.8),
    (-0.050, 0.045, 0.013, 0.0068, 0.1),    # 中
    (0.075, 0.020, 0.013, 0.0068, 0.6),
    (-0.005, -0.005, 0.017, 0.0084, 0.0),   # 大
    (0.055, 0.060, 0.017, 0.0084, 0.5),
]
FRAME_HALF = (0.130, 0.100)       # 外框半宽 (a,b) 米
FRAME_INNER = (0.125, 0.095)      # 内沿（中间是 5mm 黑线）
FRAME_ROT = np.deg2rad(10)


def _hex_vertices(radius, phi):
    ang = phi + np.arange(6) * np.pi / 3
    v = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
    if np.cross(v[1] - v[0], v[2] - v[1]) < 0:  # 保证逆时针
        v = v[::-1]
    return v


def _points_in_convex(points, verts):
    """逐点凸多边形内判定 (M,2) + (6,2) -> (M,) bool。"""
    outside = np.zeros(len(points), bool)
    for k in range(len(verts)):
        edge = verts[(k + 1) % len(verts)] - verts[k]
        cross = (edge[0] * (points[:, 1] - verts[k, 1])
                 - edge[1] * (points[:, 0] - verts[k, 0]))
        outside |= cross < 0
    return ~outside


def make_scene(nuts_def=None, noise_mm=0.0, holes=0.0, frame_rot=FRAME_ROT):
    """反渲染 -> (color, depth16, truth, n, d, K)。"""
    rng = np.random.default_rng(7)
    nuts_def = NUTS_DEF if nuts_def is None else nuts_def
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)

    # 纸面法向：绕 x 8°、绕 y 5°，过 (0,0,Z0)
    ca, sa = np.cos(np.deg2rad(8)), np.sin(np.deg2rad(8))
    cb, sb = np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5))
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    n = Rx @ Ry @ np.array([0, 0, 1.0])
    n = n / np.linalg.norm(n)
    d = float(-n @ np.array([0, 0, Z0]))
    e1, e2 = _plane_basis(n)

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    D = np.stack([(uu - CX) / FX, (vv - CY) / FY,
                  np.ones_like(uu, float)], axis=-1)
    denom = D @ n
    t_plane = -d / denom                    # 纸面射线参数
    P = D * t_plane[..., None]
    a, b = P @ e1, P @ e2

    # 黑框（旋转矩形环带）
    fr_a = np.cos(frame_rot) * a + np.sin(frame_rot) * b
    fr_b = -np.sin(frame_rot) * a + np.cos(frame_rot) * b
    inside_outer = (np.abs(fr_a) <= FRAME_HALF[0]) & \
                   (np.abs(fr_b) <= FRAME_HALF[1])
    inside_inner = (np.abs(fr_a) <= FRAME_INNER[0]) & \
                   (np.abs(fr_b) <= FRAME_INNER[1])
    band, interior = inside_outer & ~inside_inner, inside_inner

    # 螺母：六边形顶面取「上移厚度 h 的平行平面」深度，内孔回退纸面
    pix = np.column_stack([a.ravel(), b.ravel()])
    denom_flat = denom.ravel()
    nut_px = np.zeros(H * W, bool)
    hole_px = np.zeros(H * W, bool)
    z_surf = t_plane.copy()
    for ac, bc, w, hgt, phi in nuts_def:
        verts = _hex_vertices(w / np.sqrt(3), phi) + np.array([ac, bc])
        m = _points_in_convex(pix, verts)
        rad = np.hypot(pix[:, 0] - ac, pix[:, 1] - bc)
        in_hole = m & (rad <= 0.30 * w)
        top = m & ~in_hole
        nut_px |= m
        hole_px |= in_hole
        # 顶面靠相机 h：满足 n·x+d=-h（d0<0，朝相机为 -n 方向）
        t_top = (-hgt - d) / denom_flat
        z_surf.reshape(-1)[top] = t_top[top]
    nut_px = nut_px.reshape(H, W)
    hole_px = hole_px.reshape(H, W)
    nut_top = nut_px & ~hole_px

    if noise_mm:
        z_surf = np.where(interior | nut_top,
                          z_surf + rng.normal(0, noise_mm / 1000.0,
                                              z_surf.shape),
                          z_surf)
    if holes:
        inv = (rng.random(z_surf.shape) < holes) & interior & ~nut_top
        z_surf[inv] = 0.0

    depth16 = np.clip(np.rint(z_surf * 1000.0), 0, 65535).astype(np.uint16)

    # 彩图：框外桌面 225 / 纸白 245 / 黑线 25 / 螺母灰 170 / 孔 50
    color = np.full((H, W, 3), 225, np.uint8)
    color[interior] = (245, 245, 245)
    color[band] = (25, 25, 25)
    color[nut_px] = (170, 170, 170)
    color[hole_px] = (50, 50, 50)

    truth = [{'a': ac, 'b': bc, 'flats': w * 1000, 'thick': hgt * 1000,
              'p_cam': ac * e1 + bc * e2 + (-d) * n,
              'e1': e1, 'e2': e2, 'n': n, 'd': d}
             for ac, bc, w, hgt, phi in nuts_def]
    return color, depth16, truth, n, d, K


def _params(**over):
    p = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    for sec, vals in over.items():
        p[sec].update(vals)
    return p


def _closest(nuts, tr):
    """按平面坐标距离找最接近真值的检出。"""
    best, bd = None, 1e9
    for nt in nuts:
        dd = np.hypot(nt.p_cam @ tr['e1'] - tr['a'],
                      nt.p_cam @ tr['e2'] - tr['b']) * 1000
        if dd < bd:
            best, bd = nt, dd
    return best, bd


class NutVisionTests(unittest.TestCase):
    def test_plane_and_frame(self):
        color, depth, truth, n0, d0, K = make_scene()
        nuts, dbg = run_pipeline(color, depth, K, _params())
        self.assertTrue(dbg.notes[0].startswith('frame_found'),
                        dbg.notes[0])
        # 管线强制法向朝相机（d>0），与真值可能整体差一个负号
        ang = np.degrees(np.arccos(np.clip(abs(dbg.plane_n @ n0), 0, 1)))
        self.assertLess(ang, 0.5)
        self.assertLess(abs(abs(dbg.plane_d) - abs(d0)) * 1000, 2.0)
        self.assertGreater(dbg.plane_d, 0)
        self.assertGreater(dbg.inlier_ratio, 0.8)

    def test_six_nuts_sizes_centers(self):
        color, depth, truth, _, _, K = make_scene()
        nuts, _ = run_pipeline(color, depth, K, _params())
        self.assertEqual(len(nuts), 6)
        counts = {lab: sum(n.label == lab for n in nuts) for lab in LABELS}
        self.assertEqual(counts, {'s': 2, 'm': 2, 'l': 2})
        for tr in truth:
            nt, err_mm = _closest(nuts, tr)
            self.assertIsNotNone(nt)
            self.assertLess(err_mm, 4.0,
                            f'中心误差 {err_mm:.1f}mm @ flats {tr["flats"]}')
            self.assertLess(abs(nt.flats_mm - tr['flats']), 1.3,
                            f'对边距 {nt.flats_mm:.1f} vs {tr["flats"]}')
            self.assertLess(abs(nt.height_mm - tr['thick']), 2.0)

    def test_noise_and_depth_holes(self):
        color, depth, truth, _, _, K = make_scene(noise_mm=0.8, holes=0.01)
        nuts, _ = run_pipeline(color, depth, K, _params())
        self.assertEqual(len(nuts), 6)
        counts = {lab: sum(n.label == lab for n in nuts) for lab in LABELS}
        self.assertEqual(counts, {'s': 2, 'm': 2, 'l': 2})
        for tr in truth:
            nt, err_mm = _closest(nuts, tr)
            self.assertLess(err_mm, 5.0)
            self.assertLess(abs(nt.flats_mm - tr['flats']), 2.0)

    def test_thresholds_mode(self):
        one_each = [t for t in NUTS_DEF if t[0] < 0]   # 每尺寸一颗
        color, depth, _, _, _, K = make_scene(nuts_def=one_each)
        params = _params(classify={'mode': 'thresholds',
                                   's_max_mm': 11.5, 'm_max_mm': 15.0})
        nuts, _ = run_pipeline(color, depth, K, params)
        self.assertEqual([n.label for n in sorted(nuts, key=lambda x: x.flats_mm)],
                         ['s', 'm', 'l'])

    def test_touching_pair_rejected(self):
        # 两颗中号中心距 10.5mm（<对边距 13，物理相触），并域拉长应被门限拦下
        touching = [
            (-0.020, 0.0, 0.013, 0.0068, 0.0),
            (-0.0095, 0.0, 0.013, 0.0068, 0.0),
        ]
        color, depth, _, _, _, K = make_scene(nuts_def=touching)
        nuts, _ = run_pipeline(color, depth, K, _params())
        self.assertEqual(len(nuts), 0,
                         f'相触并域不应通过形状门限，却检出 {len(nuts)} 颗')

    def test_depth_conversion(self):
        d16 = np.array([[0, 1, 1234]], np.uint16)
        d32 = np.array([[0.0, 0.001, 1.234]], np.float32)
        np.testing.assert_allclose(depth_to_meters(d16),
                                   [[0, 0.001, 1.234]], atol=1e-6)
        np.testing.assert_allclose(depth_to_meters(d32),
                                   [[0, 0.001, 1.234]], atol=1e-6)

    def test_resolution_mismatch_raises(self):
        color, depth, _, _, _, K = make_scene()
        small = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_NEAREST)
        with self.assertRaises(RuntimeError):
            run_pipeline(color, small, K, _params())


if __name__ == '__main__':
    unittest.main(verbosity=2)
