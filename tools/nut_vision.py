#!/usr/bin/env python3
"""螺母视觉识别：黑框 ROI + 桌面平面高度分割 + 米制尺寸分类。

输入：Orbbec Gemini2 的彩色图 +【对齐彩色】深度图（depth_registration:=true）+ 内参 K。
输出：框内每颗平放螺母的中心相机坐标（camera_color_optical_frame，X右/Y下/Z前，米）、
      l/m/s 尺寸分类、对边距/对角距/厚度（mm）。

管线（纯函数，不依赖 rclpy，可离线单测/回放快照）：
  1. find_frame_roi   彩图暗阈值找打印黑框（最大四边形）-> ROI mask，内缩避开黑线
  2. fit_table_plane  ROI 内深度反投影 3D 点，RANSAC 拟合纸面平面 n·x+d=0
  3. segment_nuts     逐像素点到平面的有符号高度 -> 阈值/形态学/连通域；
                      连通域轮廓点投到平面二维基，minAreaRect 得米制对边距；
                      质心射线与平面求交 -> 桌面高度的中心 3D（避开螺纹孔深洞）
  4. classify_sizes   配置阈值或按直径相对间隙聚类 -> s/m/l（支持同尺寸多颗）

实机查看 / 快照 / 离线回放见文件末尾 main()：
  python3 tools/nut_vision.py                      # 实时查看器（叠加检测结果）
  python3 tools/nut_vision.py --once               # 拍一帧，打印 JSON
  python3 tools/nut_vision.py --offline 快照目录    # 离线回放（不需要 ROS/相机）
  窗口按键：q=退出  s=保存快照  d=切换高度图调试面板  p=暂停
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from camera_pick_move import load_camera_K, load_extrinsics

# ---------------- 默认参数（可被 nut_vision.yaml 覆盖） ----------------------

DEFAULT_PARAMS = {
    'frame': {
        'enable': True,           # False=不用黑框，全图当 ROI
        'method': 'edge',         # edge=Canny 四边形（细/手绘框）/ dark=暗色填充块
        'canny_lo': 60.0,         # Canny 低阈值（框线淡/纸皱误检多时整体调）
        'canny_hi': 160.0,        # Canny 高阈值（通常 = 2~3 倍低阈值）
        'edge_dilate_px': 3,      # 边缘膨胀：把断开/毛糙的手绘框线接上
        'quad_rect_min': 0.55,    # 四边形面积/最小外接矩形 >= 该值（手绘框放宽到 .55）
        'edge_contrast': 25.0,    # 沿边灰度要比框内部暗多少（纸线对比，灰度差）
        'ring_dark_values': (120, 135, 150, 165),  # ring 法扫描的黑像素阈值
        'ring_close_px': 3,       # ring 法闭运算核（补角部小缺口）
        'ring_min_area_ratio': 0.006,   # ring 法外轮廓最小面积比
        'ring_hole_ratio': 0.15,  # 亮内腔/外轮廓面积比下限（螺母实心孔很小）
        'ring_inner_min': 140.0,  # 内腔灰度中位下限（白纸才算数）
        'dark_value': 70,         # dark 法：灰度 <= 该值视为黑
        'blur_px': 5,             # dark 法高斯模糊核（奇数）
        'close_px': 15,           # dark 法闭运算核：角上连线
        'min_area_ratio': 0.012,  # 框四边形 >= 画面面积比例（真机细框≈0.025）
        'aspect_min': 0.35,       # 四边形长宽比范围
        'aspect_max': 2.8,
        'erode_px': 12,           # ROI 向内收缩像素，避开黑线本身（@0.6m≈10mm）
        'on_missing': 'image',    # 找不到框：image=全图回退 / fail=报错
        'manual_roi': None,       # [x, y, w, h] 像素，给定后不再自动找框
        # 粗框四边邻域 RANSAC 找真实框线 -> 透视梯形 ROI（失败自动回退粗框）
        'refine_lines': True,
        'refine_band_px': 45,     # 距粗边多远的暗像素参与找线（粗框可偏 20-40px）
        'refine_inlier_px': 3.5,  # 直线内点阈值（手绘线有波浪，2.5 太严）
        'refine_dark_value': 130, # 找线暗像素阈值（框线 p5≈80）
        'refine_min_span': 0.50,  # 内点沿线覆盖至少该比例的边长才算数
        'refine_iters': 200,      # RANSAC 迭代次数
        'refine_max_drift_px': 28,  # 精修角点相对粗框角点允许漂移
        # ring 外环凸包 4 点近似 -> 透视梯形（首选精修，见 _refine_quad_hull）
        'hull_eps_list': (0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04),
        'refine_edge_support': 0.50,  # 凸包每边至少该比例采样点旁有暗像素
        'refine_corner_support': 0.60,  # 角点两侧邻边近端都要暗（防伪角切纸）
        'refine_edge_band_px': 5,     # 沿线采样时左右看多宽
        'refine_edge_value': 150,     # 框线暗判据（亮面螺母 180+，框线 <120）
    },
    'plane': {
        'ransac_iters': 300,      # RANSAC 迭代次数（拟合用原始深度，不做中值）
        'thresh_mm': 2.5,         # 内点：到平面距离阈值
        'sample_stride': 4,       # 拟合用像素步长（降采样）
        'max_points': 20000,      # 参与拟合的最大点数（超过随机抽）
        'min_points': 80,         # 最少有效点，否则视为无平面
    },
    'segment': {
        'height_min_mm': 3.5,     # 高出纸面 >= 视为螺母（M6 厚 5.2mm，留噪点裕度）
        'height_max_mm': 25.0,    # 高于此的是手/臂/异物，排除
        'median_ksize': 3,        # 高度图专用深度中值滤波（只支持 3/5，0=关）
        'open_px': 2,             # 形态学开运算（去小噪点；最小螺母仅~11px，核勿大）
        'close_px': 5,            # 形态学闭运算（补倒角断裂；螺纹孔保留为洞）
        'flats_min_mm': 8.0,      # 对边距合理范围（本场最小件实测 40mm）
        'flats_max_mm': 88.0,     # 大号实测 70mm；斜俯视近侧壁+暗阈值在 136~145
                                  # 摆动会让剪影放到 78.6~83.1，留余量到 88
        'area_min_mm2': 45.0,     # 凸包面积范围（40mm 六角≈1400mm²，留余量）
        'area_max_mm2': 9000.0,   # 70mm 六角≈4250mm²；斜俯视+阈值摆动实测至 7494
        'elong_max': 1.35,        # 对角距/对边距上限（正六角 ≈1.155）
        'solidity_min': 0.80,     # 外轮廓面积/凸包面积：单颗凸形状≈1，相触<0.85
        'top_band_mm': 1.5,       # 度量时取域内高度中位以下该值内的顶面点
        'center_warn_mm': 3.0,    # 矩心与点云中位偏差超此值只告警（可能相触/缺角）
        'check_hole': False,      # 是否要求中心有暗孔（金属反光时建议关，仅打分）
        'hole_contrast': 25,      # 中心盘比外圈暗多少才算有孔（灰度）
        # 彩色剪影路径：黑色螺母红外深度会整片丢失，直接在彩图按暗阈值找剪影，
        # 轮廓射线投到桌面平面量尺寸（不依赖螺母顶面深度）
        'color_enable': True,
        'color_elong_max': 1.6,   # 剪影路径对角/对边上限（俯视斜面投影长轴偏长，放宽）
        # auto：阈值=ROI 灰度中位（纸面）-45，clamp 到 [85,150]，光照变化自跟随；
        # fixed：才用 color_dark_value（黑螺母 25~60、框线 p5≈80、纸面 170+）
        'color_thresh_mode': 'auto',
        'color_auto_margin': 45,
        'color_thresh_min': 85,
        'color_thresh_max': 150,
        'color_dark_value': 75,   # fixed 模式灰度阈值
        'color_open_px': 3,
        'color_close_px': 3,      # 核别大：相邻螺母间隙可能只有几像素
        'plane_exclude_dilate': 11,  # 拟合平面时暗色螺母块外扩排除的核（像素）
        # 深度高度法候选大部分落在被拒暗块（相触哑铃/贴边块）内、且只是
        # 其碎片（面积<0.75倍暗块）时拒绝——黑螺母侧壁红外碎片高度会造假检出
        'dark_reject_overlap': 0.35,
        # 剪影进入 ROI 内缘这条带子（像素，@0.6m≈6mm）按贴框线拒绝；
        # 叠加 frame.erode_px 后，螺母中心离框线要留约 (erode+band)/0.86 mm
        'touch_band_px': 7,
        # 窄颈相触（8 字形）拆分：腐蚀找两瓣种子 + 距离变换分水岭；
        # 深重叠（腐蚀 >split_max_erode_frac 倍等效半径）不拆，整组拒
        'split_enable': True,
        'split_max_erode_frac': 0.40,
        'split_seed_min_frac': 0.08,   # 种子至少占并域面积比例
        'split_piece_min_frac': 0.22,  # 每瓣至少占并域面积比例（50/70mm 实测小瓣 0.29）
        'split_min_solidity': 0.70,    # 并域凸度低于此值不拆（形状太怪）
    },
    'classify': {
        'mode': 'gaps',           # gaps=按直径间隙聚类 / thresholds=按固定阈值
        's_max_mm': 11.5,         # thresholds 模式：<= 小号（10/13 间隙中点）
        'm_max_mm': 15.0,         # <= 中号，再大是大号（13/17 间隙中点）
        'gap_ratio': 0.12,        # gaps 模式：相对间隙 > 此值切开成新组
        'gap_abs_mm': 1.5,        # 且绝对间隙至少这么多 mm，两个条件取松
        'max_kinds': 3,           # 最多分几档（对应 s/m/l）
    },
}

LABELS = ('s', 'm', 'l')
LABEL_COLOR_BGR = {'s': (255, 200, 0), 'm': (0, 215, 255),
                   'l': (40, 40, 255), '?': (160, 160, 160)}


# ---------------- 参数加载 ---------------------------------------------------

def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_params(path=None):
    """读 nut_vision.yaml 并与 DEFAULT_PARAMS 深合并；无文件用默认。"""
    params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    if path:
        p = Path(path)
        if p.exists():
            user = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
            params = _deep_merge(params, user)
    return params


# ---------------- 数据结构 ---------------------------------------------------

@dataclass
class Nut:
    label: str                # 's'/'m'/'l'/'?'（分类不确定时 ?）
    u: float                  # 质心像素
    v: float
    p_cam: np.ndarray         # 质心射线与桌面平面交点，camera optical，米
    flats_mm: float           # 对边距（minAreaRect 短边）
    corners_mm: float         # 对角距（长边）
    height_mm: float          # 顶面高出平面的中位数
    area_mm2: float           # 外轮廓在平面上的面积
    score: float = 0.0        # 形状打分 0~1（调试用）
    instance: int = 0         # 画面内序号（按 v 再 u 排序，稳定编号）
    contour_uv: np.ndarray = None          # 像素外轮廓（绘制用）
    extra: dict = field(default_factory=dict)

    def to_json(self, p_base=None):
        d = {
            'label': self.label,
            'instance': self.instance,
            'u': round(float(self.u), 1),
            'v': round(float(self.v), 1),
            'p_cam': [round(float(x), 4) for x in self.p_cam],
            'flats_mm': round(self.flats_mm, 2),
            'corners_mm': round(self.corners_mm, 2),
            'height_mm': round(self.height_mm, 2),
            'area_mm2': round(self.area_mm2, 1),
            'score': round(float(self.score), 2),
        }
        if p_base is not None:
            d['p_base'] = [round(float(x), 4) for x in p_base]
        return d


@dataclass
class DebugInfo:
    roi_mask: np.ndarray = None       # uint8 0/255
    quad: np.ndarray = None           # 黑框四边形 4x2（自动找到时）
    plane_n: np.ndarray = None
    plane_d: float = None
    inlier_ratio: float = None
    height_m: np.ndarray = None       # 逐像素到平面高度（米，无效 0）
    obj_mask: np.ndarray = None       # 高度阈值后的螺母候选 mask
    notes: list = field(default_factory=list)


# ---------------- 深度 / 投影基础 --------------------------------------------

def depth_to_meters(depth):
    """16UC1(mm) / 32FC(m) -> float64 米，无效为 0。"""
    scale = 1000.0 if depth.dtype == np.uint16 else 1.0
    z = depth.astype(np.float64) / scale
    z[~np.isfinite(z)] = 0.0
    return z


def deproject_image(z_m, K):
    """整幅深度反投影：返回 (H,W,3) 相机系点，无效深度处为 0。"""
    h, w = z_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        X = (u - cx) * z_m / fx
        Y = (v - cy) * z_m / fy
    return np.stack([X, Y, z_m], axis=-1)


# ---------------- 1. 黑框 ROI ------------------------------------------------

def _quad_contrast(gray, quad, thick=5):
    """四边形边沿线灰度中位数 vs 内部灰度中位数，返回 (边, 内)；无样本返回 None。"""
    import cv2
    h, w = gray.shape
    edge_m = np.zeros((h, w), np.uint8)
    pts = np.round(quad).astype(np.int32)
    cv2.polylines(edge_m, [pts], True, 255, thick)
    inner_m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inner_m, [pts], 255)
    inner_m = cv2.erode(inner_m, cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * thick + 1, 2 * thick + 1)))
    edge_m &= ~inner_m
    if edge_m.sum() == 0 or inner_m.sum() == 0:
        return None
    return float(np.median(gray[edge_m > 0])), float(np.median(gray[inner_m > 0]))


def _find_quad_by_edges(gray, fcfg):
    """Canny 边缘 + 4 顶点凸多边形近似找黑框，适合细/手绘框线。"""
    import cv2
    h, w = gray.shape
    area_img = h * w
    edges = cv2.Canny(gray, float(fcfg['canny_lo']), float(fcfg['canny_hi']),
                      apertureSize=3)
    dk = max(1, int(fcfg['edge_dilate_px']))
    edges = cv2.dilate(edges, cv2.getStructuringElement(
        cv2.MORPH_RECT, (dk, dk)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, None
    for c in contours:
        a = cv2.contourArea(c)
        if a < fcfg['min_area_ratio'] * area_img or a > 0.6 * area_img:
            continue
        peri = cv2.arcLength(c, True)
        ap = None
        for eps in (0.02, 0.035, 0.05, 0.075, 0.10):
            cand = cv2.approxPolyDP(c, eps * peri, True)
            if len(cand) == 4:
                ap = cand
                break
        if ap is None or not cv2.isContourConvex(ap):
            continue
        quad = ap.reshape(4, 2).astype(np.float32)
        x0, y0, qw, qh = cv2.boundingRect(ap)
        # 贴着画面边的轮廓是「框出画」的假四边形，拒绝
        if x0 <= 1 or y0 <= 1 or x0 + qw >= w - 1 or y0 + qh >= h - 1:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(ap)
        if rw <= 1 or rh <= 1:
            continue
        asp = max(rw, rh) / min(rw, rh)
        if not (fcfg['aspect_min'] <= asp <= fcfg['aspect_max']):
            continue
        if a / float(rw * rh) < float(fcfg['quad_rect_min']):
            continue
        cc = _quad_contrast(gray, quad)
        if cc is None:
            continue
        edge_v, inner_v = cc
        contrast = inner_v - edge_v
        if contrast < float(fcfg['edge_contrast']):
            continue
        # 面积越大、沿边越暗越优先
        score = a * (contrast + 1.0)
        if best_score is None or score > best_score:
            best, best_score = quad, score
    return best


def _find_quad_by_dark(gray, fcfg):
    """暗色阈值 + 形态学闭运算找填充黑块（粗框线/旧逻辑兜底）。"""
    import cv2
    h, w = gray.shape
    area_img = h * w
    k = max(1, int(fcfg['blur_px']))
    if k % 2 == 0:
        k += 1
    bl = cv2.GaussianBlur(gray, (k, k), 0)
    dark = cv2.threshold(bl, int(fcfg['dark_value']), 255,
                         cv2.THRESH_BINARY_INV)[1]
    ck = max(1, int(fcfg['close_px']))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck)))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, None
    for c in contours:
        if cv2.contourArea(c) < fcfg['min_area_ratio'] * area_img:
            continue
        x0, y0, cw0, ch0 = cv2.boundingRect(c)
        if x0 <= 1 or y0 <= 1 or x0 + cw0 >= w - 1 or y0 + ch0 >= h - 1:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue
        asp = max(rw, rh) / min(rw, rh)
        if not (fcfg['aspect_min'] <= asp <= fcfg['aspect_max']):
            continue
        box = cv2.boxPoints(rect)
        if cv2.contourArea(c) / float(rw * rh) < 0.25:
            continue
        score = cv2.contourArea(c)
        if best_score is None or score > best_score:
            best, best_score = box.astype(np.float32), score
    return best


def _find_quad_by_ring(gray, fcfg):
    """暗色阈值 + RETR_CCOMP 找「围着亮内腔的矩形环」（手绘细框实战法）。

    框线在角上闭合不严也没关系：只要外轮廓包住一块够大够亮的内腔即可；
    螺母是实心暗块、没有大而亮的孔，自然被排除。"""
    import cv2
    h, w = gray.shape
    area_img = h * w
    min_a = float(fcfg.get('ring_min_area_ratio', 0.006)) * area_img
    min_ratio = float(fcfg.get('ring_hole_ratio', 0.15))
    inner_min = float(fcfg.get('ring_inner_min', 140.0))
    ck = max(1, int(fcfg.get('ring_close_px', 3)))
    tvs = fcfg.get('ring_dark_values', (120, 135, 150, 165))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    best, best_score = None, None
    cands = []  # (score, box, contour)：跨阈值收集，给凸包精修多试几条轮廓
    for tv in tvs:
        m = cv2.threshold(gray, int(tv), 255, cv2.THRESH_BINARY_INV)[1]
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        hier = hier[0]
        for ci, c in enumerate(cnts):
            if hier[ci][3] != -1:
                continue  # 只看外层
            a0 = cv2.contourArea(c)
            if a0 < min_a:
                continue
            x0, y0, cw, ch = cv2.boundingRect(c)
            if x0 <= 2 or y0 <= 2 or x0 + cw >= w - 2 or y0 + ch >= h - 2:
                continue
            # 收集所有子孔，取最大
            kids, ki = [], hier[ci][2]
            while ki >= 0:
                kids.append(cnts[ki])
                ki = hier[ki][0]
            if not kids:
                continue
            kbig = max(kids, key=cv2.contourArea)
            ah = cv2.contourArea(kbig)
            if ah / max(a0, 1.0) < min_ratio:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            if rw <= 1 or rh <= 1:
                continue
            asp = max(rw, rh) / min(rw, rh)
            if not (fcfg['aspect_min'] <= asp <= fcfg['aspect_max']):
                continue
            # 孔内必须亮（白纸/浅色底面）
            km = np.zeros((h, w), np.uint8)
            cv2.drawContours(km, [kbig], -1, 255, cv2.FILLED)
            km = cv2.erode(km, erode_k)
            if km.sum() == 0:
                continue
            inner_v = float(np.median(gray[km > 0]))
            if inner_v < inner_min:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
            score = ah * (inner_v / 128.0)
            cands.append((score, box, c.reshape(-1, 2)))
            if best_score is None or score > best_score:
                best, best_score = box, score
    # 同一块框在多个阈值下都会入选：按 bbox 中心距去重，保留高分轮廓
    cands.sort(key=lambda t: -t[0])
    uniq = []
    for sc, bx, cc in cands:
        cx0, cy0 = float(bx[:, 0].mean()), float(bx[:, 1].mean())
        if any(abs(float(q[1][:, 0].mean()) - cx0) < 25
               and abs(float(q[1][:, 1].mean()) - cy0) < 25
               for q in uniq):
            continue
        uniq.append((sc, bx, cc))
    ring_cnts = [cc for _, _, cc in uniq]
    return best, ring_cnts


def _line_intersect(p1, t1, p2, t2):
    """两直线 p1+s*t1、p2+u*t2 的交点（方向不必归一化）。"""
    A = np.column_stack([t1, -t2]).astype(np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s = np.linalg.solve(A, np.asarray(p2, float) - np.asarray(p1, float))
    return np.asarray(p1, float) + s[0] * np.asarray(t1, float)


def _refine_quad_lines(gray, quad, fcfg):
    """粗四边形各边邻域内 RANSAC 找真实框线，交回透视四角多边形。

    手绘黑框斜俯拍时是梯形：minAreaRect 的轴对齐矩形在角上会把框线甚至
    框外区域圈进 ROI（轴对齐回退方案因此要内缩 12px，挤掉螺母余量）。
    任一边支撑不足、角点漂移过大或面积异常则返回 None，回退粗四边形。
    """
    import cv2
    band = float(fcfg.get('refine_band_px', 45))
    inl_thr = float(fcfg.get('refine_inlier_px', 3.5))
    dark_v = float(fcfg.get('refine_dark_value', 130))
    min_span = float(fcfg.get('refine_min_span', 0.50))
    iters = int(fcfg.get('refine_iters', 200))
    max_drift = float(fcfg.get('refine_max_drift_px', 28))
    rng = np.random.default_rng(7)

    pts0 = np.asarray(quad, np.float64).reshape(-1, 2)
    ctr = pts0.mean(axis=0)
    pts0 = pts0[np.argsort(np.arctan2(pts0[:, 1] - ctr[1],
                                     pts0[:, 0] - ctr[0]))]  # 逆时针
    dark = np.argwhere(gray < dark_v)[:, ::-1].astype(np.float64)

    sides = []
    for i in range(4):
        a0, b0 = pts0[i], pts0[(i + 1) % 4]
        t0 = b0 - a0
        L = np.linalg.norm(t0)
        if L < 10:
            return None
        t0 /= L
        n0 = np.array([-t0[1], t0[0]])
        rel = dark - a0
        q = dark[(np.abs(rel @ n0) < band) & (rel @ t0 > -6) &
                 (rel @ t0 < L + 6)]
        if len(q) < 30:
            return None
        best = None  # (内点数, 内点集)
        for _ in range(iters):
            ia, ib = rng.choice(len(q), 2, replace=False)
            dd = q[ib] - q[ia]
            ll = np.linalg.norm(dd)
            if ll < 0.4 * L:
                continue
            tt = dd / ll
            res = np.abs((q - q[ia]) @ np.array([-tt[1], tt[0]]))
            inl = res < inl_thr
            if inl.sum() < 20:
                continue
            sp = (q[inl] - q[ia]) @ tt
            if (sp.max() - sp.min()) < min_span * L:
                continue
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), tt, q[inl])
        if best is None:
            return None
        vx, vy, x0, y0 = cv2.fitLine(best[2].astype(np.float32),
                                     cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        tt = np.array([vx, vy], float)
        if tt @ t0 < 0:
            tt = -tt
        sides.append((np.array([x0, y0], float), tt))

    # corner[i] = side(i-1) ∩ side(i)（逆时针）
    corners = []
    for i in range(4):
        cp = _line_intersect(*sides[(i - 1) % 4], *sides[i])
        if cp is None:
            return None
        corners.append(cp)
    cq = np.array(corners, np.float64)
    if np.any(np.linalg.norm(cq - pts0, axis=1) > max_drift):
        return None
    a0 = cv2.contourArea(pts0.astype(np.float32))
    a1 = cv2.contourArea(cq.astype(np.float32))
    if not (0.7 * a0 < a1 < 1.3 * a0):
        return None
    # 凸性自检（四边形应近似凸）
    if cv2.contourArea(cq.astype(np.float32)) < 0.9 * cv2.contourArea(
            cv2.convexHull(cq.astype(np.float32))):
        return None
    return cq.astype(np.float32)


def _refine_quad_hull(gray, cnt, coarse, fcfg):
    """ring 外环轮廓凸包 + approxPolyDP 取 4 凸点 = 黑框外缘透视梯形。

    斜俯拍下框是梯形；minAreaRect 粗框反而会被轮廓杂点（塑封反光褶皱）
    拉偏，2026-09-10 快照粗框左边比真线外偏 20~70px，而凸包 TL=(409,542)
    与灰度剖面量出的真左边线（过 (344,700)/(386,600)，延长到 y=542 得
    x≈410）完全吻合。螺母与框线暗桥接时，凸包沿外缘切线绕过内凹仍落在
    真实框角上。
    校验：恰好 4 点凸多边形、面积与粗框相当、不出画面、每条弦沿线暗像素
    支撑比例够（防止弦从纸面/亮褶皱上切过）、且每个顶点两侧的相邻边都
    压在暗线上。失败 None。
    """
    import cv2
    h, w = gray.shape
    hull = cv2.convexHull(np.asarray(cnt, np.float32).reshape(-1, 1, 2))
    arc = cv2.arcLength(hull, True)
    eps_list = fcfg.get('hull_eps_list',
                        (0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04))
    support = float(fcfg.get('refine_edge_support', 0.50))
    corner_sup = float(fcfg.get('refine_corner_support', 0.60))
    band = max(1, int(fcfg.get('refine_edge_band_px', 5)))
    dark_v = int(fcfg.get('refine_edge_value', 150))
    c0 = np.asarray(coarse, np.float64).reshape(-1, 2)
    a0 = cv2.contourArea(c0.astype(np.float32))
    if a0 < 1:
        return None

    def seg_support(p1, p2, t_lo=0.0, t_hi=1.0):
        """边沿 t 区间采样，±band 邻域内能找到暗像素的采样点比例。

        曾试过用邻域暗填充率区分「细框线」与「螺母块内切弦」，但受光螺母
        顶面灰度 180+、截面填充率同样很低（2026-09-10 实测 0.2 左右），
        分不开，反而会误杀好边（live 快照好边 0.56），已撤。
        """
        L = np.linalg.norm(p2 - p1)
        n_s = max(6, int((t_hi - t_lo) * L / 4.0))
        ts = np.linspace(t_lo, t_hi, n_s)
        smp = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        hits = 0
        for sx, sy in smp:
            x0, y0 = int(round(sx)), int(round(sy))
            patch = gray[max(0, y0 - band):y0 + band + 1,
                         max(0, x0 - band):x0 + band + 1]
            if patch.size and patch.min() < dark_v:
                hits += 1
        return hits / n_s

    for eps in eps_list:
        ap = cv2.approxPolyDP(hull, float(eps) * arc, True)
        if len(ap) != 4 or not cv2.isContourConvex(ap):
            continue
        q = ap.reshape(4, 2).astype(np.float64)
        ctr = q.mean(axis=0)
        q = q[np.argsort(np.arctan2(q[:, 1] - ctr[1], q[:, 0] - ctr[0]))]
        a1 = cv2.contourArea(q.astype(np.float32))
        if not (0.65 * a0 < a1 < 1.35 * a0):
            continue
        if np.any(q < 2) or np.any(q[:, 0] > w - 3) or np.any(q[:, 1] > h - 3):
            continue
        ok = True
        for i in range(4):
            p1, p2 = q[i], q[(i + 1) % 4]
            if seg_support(p1, p2) < support:
                ok = False
                break
        if not ok:
            continue
        # 角点校验：顶点前后各取邻边 4%~22% 段，真框角两边都暗
        for i in range(4):
            pa, pv, pb = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
            if (seg_support(pv, pa, 0.04, 0.22) < corner_sup
                    or seg_support(pv, pb, 0.04, 0.22) < corner_sup):
                ok = False
                break
        if ok:
            return q.astype(np.float32)
    return None


def _ring_union_contour(gray, coarse, fcfg):
    """粗框邻域内把各阈值下的暗像素 OR 起来，取最大外轮廓。

    某条框边在单一阈值下因反光闭合不全时（2026-09-10 快照左边上段被塑
    封反光洗白），其它阈值仍可能有该边像素；合并后凸包能找回真角。螺母
    都在框内，凸包不受内部点影响；斜跨框内的褶皱线同样是内点。"""
    import cv2
    h, w = gray.shape
    near = np.zeros((h, w), np.uint8)
    cv2.fillPoly(near, [np.round(np.asarray(coarse, np.float32)).astype(np.int32)],
                 255)
    band = max(3, int(fcfg.get('refine_band_px', 45)))
    near = cv2.dilate(near, cv2.getStructuringElement(
        cv2.MORPH_RECT, (band, band)))
    merged = np.zeros((h, w), np.uint8)
    for tv in fcfg.get('ring_dark_values', (120, 135, 150, 165)):
        merged |= cv2.threshold(gray, int(tv), 255,
                                cv2.THRESH_BINARY_INV)[1]
    merged = cv2.morphologyEx(
        merged & near, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea).reshape(-1, 2) if cnts else None


def find_frame_roi(color, fcfg):
    """在彩图中找打印的黑色矩形框，返回 (mask uint8 0/255, quad|None, note)。

    三级：edge=Canny 4 顶点四边形（清晰打印框）→ ring=亮内腔矩形环
   （手绘细框/框内有黑螺母时最稳）→ dark=暗色填充块（粗框兜底）。"""
    import cv2
    h, w = color.shape[:2]
    manual = fcfg.get('manual_roi')
    if manual:
        x, y, rw, rh = [int(v) for v in manual]
        mask = np.zeros((h, w), np.uint8)
        mask[max(0, y):min(h, y + rh), max(0, x):min(w, x + rw)] = 255
        return mask, None, 'manual_roi'
    if not fcfg.get('enable', True):
        return np.full((h, w), 255, np.uint8), None, 'frame_disabled'

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # ring 优先：它要求外环围一块够亮够大的内腔，比通用 Canny 四边形特异
    # 得多——edge 法在实验室背景里会把远处桌沿/桌布折线误当框（2026-09-10
    # 新快照即如此），ring 命中打印黑框而 edge 锁到背景
    best, ring_cnts = _find_quad_by_ring(gray, fcfg)
    note = 'frame_found_ring' if best is not None else None
    if best is None:
        best = _find_quad_by_edges(gray, fcfg)
        if best is not None:
            note = 'frame_found'
    if best is None:
        best = _find_quad_by_dark(gray, fcfg)
        if best is not None:
            note = 'frame_found_dark'

    if best is None:
        if fcfg.get('on_missing', 'image') == 'fail':
            raise RuntimeError('未找到黑框（frame.on_missing=fail）')
        return np.full((h, w), 255, np.uint8), None, 'frame_not_found->full_image'

    # ring 命中：先用粗框邻域内多阈值暗像素的合并轮廓试凸包（单边被反光
    # 洗白时，其它阈值仍留该边像素；内部螺母/褶皱对凸包无影响），再退回
    # 跨阈值外环逐一试凸包；全部校验失败再回退四边邻域 RANSAC 精修
    refined = None
    union_cnt = _ring_union_contour(gray, best, fcfg)
    if union_cnt is not None:
        refined = _refine_quad_hull(gray, union_cnt, best, fcfg)
        if refined is not None:
            note += '+hullU'
    if refined is None:
        for rc in ring_cnts:
            refined = _refine_quad_hull(gray, rc, best, fcfg)
            if refined is not None:
                note += '+hull'
                break
    if refined is None and fcfg.get('refine_lines', True):
        refined = _refine_quad_lines(gray, best, fcfg)
        if refined is not None:
            note += '+lines'
    if refined is not None:
        best = refined

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(best).astype(np.int32)], 255)
    e = max(0, int(fcfg['erode_px']))
    if e:
        mask = cv2.erode(mask, cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * e + 1, 2 * e + 1)))
    return mask, best, note


# ---------------- 2. 桌面平面 RANSAC -----------------------------------------

def fit_table_plane(z_m, K, roi_mask, pcfg):
    """RANSAC 拟合纸面平面 n·x+d=0，法向朝向相机（原点处 n·0+d>0）。

    返回 (n(3,), d, inlier_ratio)。点太少抛 RuntimeError。
    """
    st = max(1, int(pcfg.get('sample_stride', 4)))
    valid = (z_m > 0) & (roi_mask > 0)
    ys, xs = np.where(valid)
    if len(xs) > 2000:
        xs, ys = xs[::st], ys[::st]
    if len(xs) > int(pcfg.get('max_points', 20000)):
        sel = np.random.default_rng(0).choice(
            len(xs), int(pcfg['max_points']), replace=False)
        xs, ys = xs[sel], ys[sel]
    if len(xs) < int(pcfg.get('min_points', 80)):
        raise RuntimeError(f'ROI 内有效深度点不足（{len(xs)}），相机是否出图/对齐？')

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.column_stack([
        (xs - cx) * z_m[ys, xs] / fx,
        (ys - cy) * z_m[ys, xs] / fy,
        z_m[ys, xs],
    ])

    rng = np.random.default_rng(42)
    thresh = float(pcfg['thresh_mm']) / 1000.0
    iters = int(pcfg['ransac_iters'])
    best_inl = np.ones(n_pts := len(P), bool)  # 兜底：极端情况下全部点共线才会不更新
    for _ in range(iters):
        i, j, kk = rng.choice(n_pts, 3, replace=False)
        n = np.cross(P[j] - P[i], P[kk] - P[i])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ P[i]
        if d < 0:              # 让相机原点在平面正侧，螺母（更近）高度为正
            n, d = -n, -d
        res = np.abs(P @ n + d)
        inl = res < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl

    # 最小二乘精修（最小特征向量 = 法向）
    Q = P[best_inl]
    ctr = Q.mean(axis=0)
    cov = (Q - ctr).T @ (Q - ctr) / len(Q)
    n = np.linalg.eigh(cov)[1][:, 0]
    d = float(-n @ ctr)
    if d < 0:
        n, d = -n, -d
    inlier_ratio = float(best_inl.sum()) / n_pts
    return n, d, inlier_ratio


# ---------------- 3. 螺母分割/度量 -------------------------------------------

def _plane_basis(n):
    """平面二维正交基：e1 取相机 X 轴在平面上的投影，e2=n×e1。"""
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.array([0.0, 1.0, 0.0]) - n[1] * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _hole_score(gray, u, v, r_out, r_in):
    """中心盘平均灰度 vs 外圈环带：越暗分越高（0~1）。"""
    import cv2
    h, w = gray.shape
    if r_out < 2 or not (r_out < u < w - r_out and r_out < v < h - r_out):
        return 0.0
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    disk = xx ** 2 + yy ** 2 <= r_in ** 2
    ring = (xx ** 2 + yy ** 2 <= r_out ** 2) & (xx ** 2 + yy ** 2 > r_in ** 2)
    g = gray[int(v) - r_out:int(v) + r_out + 1, int(u) - r_out:int(u) + r_out + 1]
    if g.shape != disk.shape:
        return 0.0
    contrast = float(g[ring].mean()) - float(g[disk].mean())
    return float(np.clip(contrast / 40.0, 0, 1))


def build_dark_mask(color, roi_mask, scfg):
    """框内暗色剪影 mask（黑螺母专用，黑螺母顶面红外深度会整片丢失/偏置）。

    返回 (mask, tv)；关闭时返回 (None, 0)。
    阈值模式 color_thresh_mode：
      auto  = ROI 灰度中位（纸面亮度）- color_auto_margin，并 clamp 到
              [color_thresh_min, color_thresh_max]——现场光照变化时跟着纸面
              亮度走；螺母受光面可能与纸面同亮，留在剪影里作"内洞"即可，
              外轮廓仍闭合，不影响米制度量。实测：纸面 179 时阈值约 134。
      fixed = 固定 color_dark_value（黑螺母 25~60、框线 p5≈80、纸面 170+）。
    """
    import cv2
    if not scfg.get('color_enable', True):
        return None, 0
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    roi = roi_mask > 0
    if not np.any(roi):
        return None, 0
    if str(scfg.get('color_thresh_mode', 'auto')).lower() == 'fixed':
        tv = int(scfg.get('color_dark_value', 75))
        if tv <= 0:
            return None, 0
    else:
        paper = float(np.median(gray[roi]))
        tv = int(np.clip(paper - float(scfg.get('color_auto_margin', 45)),
                         float(scfg.get('color_thresh_min', 85)),
                         float(scfg.get('color_thresh_max', 150))))
    m = ((gray < tv) & roi).astype(np.uint8) * 255
    ok = max(1, int(scfg.get('color_open_px', 3)))
    ck = max(1, int(scfg.get('color_close_px', 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck)))
    return m, tv


def _metric_contour_plane(uv, K, n, d, e1, e2):
    """像素轮廓点 -> 射线交桌面平面 -> 平面凸包 -> (flats,corners,area_mm2,hull2d)。"""
    import cv2
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dd = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy,
                   np.ones(len(uv))], axis=1)
    denom = dd @ n
    ok = np.abs(denom) > 1e-9
    dd, denom = dd[ok], denom[ok]
    if len(dd) < 4:
        return None
    X = dd * (-d / denom)[:, None]
    ab = np.column_stack([X @ e1, X @ e2]).astype(np.float32)
    hull = cv2.convexHull(ab.reshape(-1, 1, 2))
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    flats = min(rw, rh) * 1000.0
    corners = max(rw, rh) * 1000.0
    area_mm2 = cv2.contourArea(hull) * 1e6
    return flats, corners, area_mm2


def _split_figure8(comp_u8, scfg):
    """两颗凸螺母以窄颈相触（8 字形）时，用距离变换分水岭拆成两块。

    做法：填掉螺纹内孔 -> 逐步腐蚀，第一次出现恰好 2 个够大连通域时把
    它们当种子 -> 在距离变换图上 watershed 还原完整两瓣。深重叠（需要
    腐蚀掉 >split_max_erode_frac 倍等效半径才分开，或种子 >2 个）返回
    []，交给安全策略整组拒绝。
    """
    import cv2
    cs, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    filled = np.zeros_like(comp_u8)
    cv2.drawContours(filled, [max(cs, key=cv2.contourArea)], -1, 255,
                     cv2.FILLED)
    area_tot = float(cv2.countNonZero(filled))
    r_eq = np.sqrt(area_tot / np.pi)
    k_max = max(2, int(r_eq * float(scfg.get('split_max_erode_frac', 0.40))))
    seed_min = area_tot * float(scfg.get('split_seed_min_frac', 0.08))
    seeds = None
    for k in range(1, k_max + 1):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2 * k + 1, 2 * k + 1))
        er = cv2.erode(filled, ker)
        nl, ll, sl, _ = cv2.connectedComponentsWithStats(er, 8)
        ids = [i for i in range(1, nl)
               if sl[i, cv2.CC_STAT_AREA] >= seed_min]
        if len(ids) == 2:
            seeds = (ll, ids)
            break
        if len(ids) > 2:
            return []
    if seeds is None:
        return []
    ll, ids = seeds
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    markers = np.zeros(comp_u8.shape, np.int32)
    markers[filled == 0] = 1
    markers[ll == ids[0]] = 2
    markers[ll == ids[1]] = 3
    dn = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(255 - dn, cv2.COLOR_GRAY2BGR), markers)
    min_frac = float(scfg.get('split_piece_min_frac', 0.22))
    pieces = []
    for val in (2, 3):
        pm = ((markers == val) & (comp_u8 > 0)).astype(np.uint8) * 255
        if cv2.countNonZero(pm) < min_frac * area_tot:
            return []  # 一瓣过小：不是两颗等量级螺母
        pieces.append(pm)
    return pieces


def _make_silhouette_nut(comp_u8, z_m, h_raw, K, n, d, e1, e2, gray,
                         roi_mask, inner, scfg, px_per_mm, warn_tag=''):
    """单个暗剪影块：轮廓/凸包度量 + 贴边与形状门限 -> Nut；不通过返回 None。"""
    import cv2
    fx = K[0, 0]
    cs, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    cnt = max(cs, key=cv2.contourArea)
    hull_px = cv2.convexHull(cnt)
    solidity = cv2.contourArea(cnt) / max(cv2.contourArea(hull_px), 1e-6)
    metric = _metric_contour_plane(
        hull_px[:, 0, :].astype(np.float64), K, n, d, e1, e2)
    touch = bool(np.any((comp_u8 > 0) & (inner == 0) & (roi_mask > 0)))
    if metric is None or touch:
        return None
    flats, corners, area_mm2 = metric
    elong = corners / max(flats, 1e-6)
    sol_min = float(scfg.get('solidity_min', 0.80))
    if not (scfg['flats_min_mm'] <= flats <= scfg['flats_max_mm']
            and elong <= float(scfg.get('color_elong_max', 1.6))
            and scfg['area_min_mm2'] <= area_mm2 <= scfg['area_max_mm2']
            and solidity >= sol_min):
        return None

    M = cv2.moments(comp_u8, binaryImage=True)
    if abs(M['m00']) < 1e-9:
        return None
    uc, vc = M['m10'] / M['m00'], M['m01'] / M['m00']
    D = np.array([(uc - K[0, 2]) / fx, (vc - K[1, 2]) / K[1, 1], 1.0])
    if abs(n @ D) < 1e-9:
        return None
    p_cam = D * (-d / (n @ D))

    sel = (comp_u8 > 0) & (z_m > 0)
    frac = float(sel.sum()) / max(float((comp_u8 > 0).sum()), 1)
    if frac > 0.35:
        med_h = float(np.median(h_raw[sel]))
        source = 'color+depth'
    else:
        med_h, source = 0.0, 'color'
    r_in_px = max(2.0, 0.22 * flats * px_per_mm)
    r_out_px = max(r_in_px + 2.0, 0.42 * flats * px_per_mm)
    hole = _hole_score(gray, int(uc), int(vc),
                       int(r_out_px), int(r_in_px))
    shape_score = float(np.clip(
        0.5 * solidity + 0.5 * min(1.0, 1.155 / max(elong, 1e-6) / 0.92),
        0, 1))
    return Nut(
        label='?', u=uc, v=vc, p_cam=p_cam,
        flats_mm=flats, corners_mm=corners,
        height_mm=med_h * 1000.0, area_mm2=area_mm2,
        score=shape_score,
        contour_uv=cnt.reshape(-1, 2),
        extra={'elong': round(elong, 3),
               'solidity': round(float(solidity), 3),
               'hole': round(hole, 2),
               'source': source,
               'warning': warn_tag},
    )


def _dark_silhouette_nuts(z_m, h_raw, K, n, d, e1, e2, gray, roi_mask,
                          dark_mask, scfg):
    """彩色剪影路径：暗块连通域 -> 贴边/形状门限 -> 米制度量 -> Nut。

    返回 (nuts, bad_labels, bad_n, bad_areas)。bad_labels 给每个「像螺母
    但没通过门限」的暗块（相触哑铃、贴框线块）标上 1..n-1 的组件号；
    深度高度法候选若主要落在这种块内、且本身只是该块的碎片，必须一起
    拒绝——黑螺母近侧竖直面红外仍有回波，会在相触组上读出一圈碎片
    高度，不拦就会把相触的两颗误拆成多颗假检出。
    """
    import cv2
    h, w = z_m.shape
    fx = K[0, 0]
    tb = max(0, int(scfg.get('touch_band_px', 7)))
    inner = (cv2.erode(roi_mask,
                       cv2.getStructuringElement(
                           cv2.MORPH_RECT, (2 * tb + 1, 2 * tb + 1)))
             if tb else roi_mask)
    z_roi = (float(np.median(z_m[(z_m > 0) & (roi_mask > 0)]))
             if np.any((z_m > 0) & (roi_mask > 0)) else 0.6)
    px_per_mm = fx / (z_roi * 1000.0)
    n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(dark_mask, 8)
    bad_labels = np.zeros((h, w), np.int32)
    bad_areas = [0]
    nuts = []
    next_bad = 1
    for li in range(1, n_c):
        area_px = int(st_c[li, cv2.CC_STAT_AREA])
        if area_px < 35:
            continue
        comp_u8 = (lab_c == li).astype(np.uint8) * 255
        nut = _make_silhouette_nut(comp_u8, z_m, h_raw, K, n, d, e1, e2,
                                   gray, roi_mask, inner, scfg, px_per_mm)
        if nut is not None:
            nuts.append(nut)
            continue

        # 整块没过门限：若是两颗凸螺母窄颈相触的 8 字形，试着拆开；几何上
        # 成功分成两瓣后，各自独立过门限（贴边/形状），过哪瓣收哪瓣，并带
        # touch_split 警告提示操作者这颗与邻件物理相触。深重叠拆不开。
        split_nuts = []
        if scfg.get('split_enable', True):
            cnts0, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if cnts0:
                hull0 = cv2.convexHull(max(cnts0, key=cv2.contourArea))
                sol0 = (cv2.contourArea(max(cnts0, key=cv2.contourArea))
                        / max(cv2.contourArea(hull0), 1e-6))
                if sol0 >= float(scfg.get('split_min_solidity', 0.70)):
                    pieces = _split_figure8(comp_u8, scfg)
                    if len(pieces) == 2:
                        for pm in pieces:
                            pn = _make_silhouette_nut(
                                pm, z_m, h_raw, K, n, d, e1, e2, gray,
                                roi_mask, inner, scfg, px_per_mm,
                                warn_tag='touch_split')
                            if pn is not None:
                                split_nuts.append(pn)
        if split_nuts:
            nuts.extend(split_nuts)
            # 原并域仍标 bad：拦住深度侧壁碎片在接触带上造假检出
            bad_labels[lab_c == li] = next_bad
            bad_areas.append(area_px)
            next_bad += 1
            continue

        bad_labels[lab_c == li] = next_bad
        bad_areas.append(area_px)
        next_bad += 1
    return nuts, bad_labels, next_bad, bad_areas


def segment_nuts(color, z_m, K, n, d, roi_mask, scfg, dark_mask=None):
    """高度法 + 彩色剪影法分割螺母并做米制度量，返回 Nut 列表（label 暂为 '?'）。

    中心坐标：质心射线与桌面平面的交点（桌面高度，正好是抓取参考面；
    即使中心螺纹孔处深度打到桌面上，交点也不受影响）。

    dark_mask：build_dark_mask 的结果（黑螺母剪影）。黑螺母顶面深度整片
    丢失时高度法看不见它们，改由剪影轮廓投影到桌面度量。
    """
    import cv2
    h, w = z_m.shape
    dbg_obj = np.zeros((h, w), np.uint8)

    z_work = z_m
    ksz = int(scfg.get('median_ksize', 5))
    if ksz >= 3:
        # 浮点深度中值滤波只支持 3/5 核；先滤后按原有效掩码裁回，不向外扩张有效区
        kk = 5 if ksz >= 5 else 3
        z_f = cv2.medianBlur(z_m.astype(np.float32), kk).astype(np.float64)
        z_work = np.where(z_m > 0, z_f, 0.0)

    P = deproject_image(z_work, K)
    valid = z_work > 0
    height = np.zeros((h, w), np.float64)
    height[valid] = P[valid] @ n + d          # 有符号距离，螺母为正

    # 米制度量另用原始深度的高度图：中值/形态学会让小螺母（~11px）
    # 边缘混入低高度点，凸包系统性测小 1~2mm
    P_raw = deproject_image(z_m, K)
    h_raw = np.zeros((h, w), np.float64)
    mv = z_m > 0
    h_raw[mv] = P_raw[mv] @ n + d

    h_lo = scfg['height_min_mm'] / 1000.0
    h_hi = scfg['height_max_mm'] / 1000.0
    cand = (height >= h_lo) & (height <= h_hi) & (roi_mask > 0) & valid
    cand = cand.astype(np.uint8) * 255
    ok = max(1, int(scfg.get('open_px', 3)))
    ck = max(1, int(scfg.get('close_px', 5)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck)))
    dbg_obj = cand

    n_lab, lab_img, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
    fx = K[0, 0]
    z_roi = float(np.median(z_work[(z_work > 0) & (roi_mask > 0)])) \
        if np.any((z_work > 0) & (roi_mask > 0)) else 0.6
    px_per_mm = fx / (z_roi * 1000.0)
    min_area_px = max(35.0, 0.5 * (scfg['flats_min_mm'] * px_per_mm) ** 2)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    e1, e2 = _plane_basis(n)

    # 先判彩色剪影组件：通过的是黑螺母检出，未通过的大暗块（相触/贴边）
    # 用来门控下面的深度高度法，防止其在同组暗块上读出碎片假检出
    color_nuts = []
    bad_labels, bad_n, bad_areas = None, 0, [0]
    if dark_mask is not None and (dark_mask > 0).any():
        color_nuts, bad_labels, bad_n, bad_areas = _dark_silhouette_nuts(
            z_m, h_raw, K, n, d, e1, e2, gray, roi_mask, dark_mask, scfg)

    nuts = []
    ov_thr = float(scfg.get('dark_reject_overlap', 0.35))
    for li in range(1, n_lab):
        area_px = int(stats[li, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        comp = (lab_img == li)
        if bad_labels is not None:
            # 大部分落在被拒暗块内、且明显只是该暗块的碎片 -> 假检出，拒绝
            vals, cnts = np.unique(bad_labels[comp], return_counts=True)
            if any(v != 0 and c / area_px > ov_thr
                   and area_px < 0.75 * bad_areas[v]
                   for v, c in zip(vals, cnts)):
                continue
        comp_u8 = comp.astype(np.uint8) * 255
        # 连通域邻域内用【原始深度】度量：中值滤波会把仅 2~3px 厚的小螺母
        # 环顶抹平，形态学闭运算又会把域撑到周围低高度点上
        near = cv2.dilate(
            comp_u8,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0
        foot_lo = min(0.002, 0.6 * h_lo)   # 域邻域内放宽下限，纳入倒角边缘
        foot = near & (h_raw >= foot_lo) & (h_raw <= h_hi)
        ys_f, xs_f = np.where(foot)
        pp = P_raw[ys_f, xs_f]
        good_f = np.linalg.norm(pp, axis=1) > 0
        ys_f, xs_f, pp = ys_f[good_f], xs_f[good_f], pp[good_f]
        if len(pp) < 8:
            continue
        med_h = float(np.median(h_raw[ys_f, xs_f]))  # 顶面厚度（原始高度中位）
        q1, q2 = pp @ e1, pp @ e2
        pts2d = np.column_stack([q1, q2]).astype(np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts2d)
        (rcx, rcy), (rw, rh), _ = cv2.minAreaRect(hull)
        flats = min(rw, rh) * 1000.0
        corners = max(rw, rh) * 1000.0
        area_mm2 = cv2.contourArea(hull) * 1e6

        # 像素 solidity：单颗凸六角≈1；两颗相触的哑铃域 <0.85
        cs, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        hull_px = cv2.convexHull(cnt)
        solidity = cv2.contourArea(cnt) / max(cv2.contourArea(hull_px), 1e-6)

        # 形状门限（全米制，不随距离变化）
        if not (scfg['flats_min_mm'] <= flats <= scfg['flats_max_mm']):
            continue
        elong = corners / max(flats, 1e-6)
        if elong > scfg['elong_max']:
            continue
        if not (scfg['area_min_mm2'] <= area_mm2 <= scfg['area_max_mm2']):
            continue
        if solidity < scfg.get('solidity_min', 0.80):
            continue

        M = cv2.moments(comp_u8, binaryImage=True)
        if abs(M['m00']) < 1e-9:
            continue
        uc, vc = M['m10'] / M['m00'], M['m01'] / M['m00']

        # 中心射线 ∩ 桌面平面（孔打到纸面也不影响）
        D = np.array([(uc - K[0, 2]) / fx, (vc - K[1, 2]) / K[1, 1], 1.0])
        denom = n @ D
        if abs(denom) < 1e-9:
            continue
        t = -d / denom
        p_cam = t * D

        # 交叉校验：顶面点云中位垂直投回平面，应与矩心射线交点重合
        pm = np.median(pp, axis=0)
        pm = pm - (pm @ n + d) * n
        center_err = float(np.linalg.norm(pm - p_cam)) * 1000.0
        warn = ('center_split' if center_err > scfg.get('center_warn_mm', 3.0)
                else '')

        r_in_px = max(2.0, 0.22 * flats * px_per_mm)
        r_out_px = max(r_in_px + 2.0, 0.42 * flats * px_per_mm)
        hole = _hole_score(gray, int(uc), int(vc), int(r_out_px), int(r_in_px))
        shape_score = float(np.clip(
            0.5 * solidity + 0.5 * min(1.0, 1.155 / max(elong, 1e-6) / 0.92),
            0, 1))
        if scfg.get('check_hole') and hole < 0.2:
            continue

        nuts.append(Nut(
            label='?', u=uc, v=vc, p_cam=p_cam,
            flats_mm=flats, corners_mm=corners,
            height_mm=med_h * 1000.0, area_mm2=area_mm2,
            score=(shape_score + hole) / 2.0 if scfg.get('check_hole') else shape_score,
            contour_uv=cnt.reshape(-1, 2),
            extra={'elong': round(elong, 3),
                   'solidity': round(float(solidity), 3),
                   'hole': round(hole, 2),
                   'center_err_mm': round(center_err, 2),
                   'warning': warn},
        ))

    # ---- 彩色剪影路径（组件预判在深度循环前完成，这里合入）----------------
    nuts.extend(color_nuts)
    if dark_mask is not None:
        dbg_obj = cv2.bitwise_or(dbg_obj, dark_mask)

    # 去重：同一颗螺母两条路径都报时保留深度高度法（有厚度），按质心米制距离判
    depth_nuts = [x for x in nuts if not str(x.extra.get('source', '')
                                           ).startswith('color')]
    color_kept = [x for x in nuts if str(x.extra.get('source', '')
                                        ).startswith('color')]
    kept = list(depth_nuts)
    for cn in color_kept:
        dup = False
        for dn in depth_nuts:
            gap = float(np.linalg.norm(dn.p_cam - cn.p_cam)) * 1000.0
            if gap < 0.5 * cn.flats_mm:
                dup = True
                break
        if not dup:
            kept.append(cn)
    nuts = kept

    # 稳定编号：画面从左到右
    nuts.sort(key=lambda x: x.u)
    for i, nt in enumerate(nuts):
        nt.instance = i
    return nuts, height, dbg_obj


# ---------------- 4. 尺寸分类 -------------------------------------------------

def classify_sizes(nuts, ccfg):
    """按直径把螺母分到 s/m/l（原地写 label），返回 nuts。

    thresholds：固定阈值 s_max_mm / m_max_mm；
    gaps：直径排序后按相对间隙聚类（支持同尺寸多颗），最多 max_kinds 组，
          组从小到大映射 s,m,l。组数不足时映射到最小的若干档。
    """
    if not nuts:
        return nuts
    mode = ccfg.get('mode', 'gaps')
    if mode == 'thresholds':
        for nt in nuts:
            nt.label = ('s' if nt.flats_mm <= ccfg['s_max_mm'] else
                        'm' if nt.flats_mm <= ccfg['m_max_mm'] else 'l')
        return nuts

    order = sorted(range(len(nuts)), key=lambda i: nuts[i].flats_mm)
    groups = [[order[0]]]
    gap_ratio = float(ccfg['gap_ratio'])
    gap_abs = float(ccfg.get('gap_abs_mm', 1.5))
    for a, b in zip(order, order[1:]):
        da, db = nuts[a].flats_mm, nuts[b].flats_mm
        # 相对间隙与绝对间隙取松（任一超过即切开）
        if (db - da) > max(gap_ratio * da, gap_abs):
            groups.append([b])
        else:
            groups[-1].append(b)

    # 组过多：反复合并「组均值最接近」的相邻组，直到 max_kinds
    while len(groups) > int(ccfg.get('max_kinds', 3)):
        means = [np.mean([nuts[i].flats_mm for i in g]) for g in groups]
        gaps = [means[i + 1] - means[i] for i in range(len(groups) - 1)]
        k = int(np.argmin(gaps))
        groups[k:k + 2] = [groups[k] + groups[k + 1]]

    # 无绝对尺度信息时，组数不足 3 档只能按从小到大占 s,m（现场若实际是 m,l，
    # 请改用 classify.mode=thresholds 按实测对边距精确分档）
    labels = LABELS[:len(groups)]
    for g, lab in zip(groups, labels):
        for i in g:
            nuts[i].label = lab
    return nuts


# ---------------- 管线入口 ---------------------------------------------------

def run_pipeline(color, depth, K, params):
    """整帧：color(BGR) + depth(原始消息格式，mm/m 均可) + K -> (nuts, DebugInfo)。

    深度必须是 depth_registration:=true 的对齐深度（与彩色同分辨率）；
    不一致直接报错——最近邻硬缩放会拿彩色 K 配不同视场，几何是错的。
    """
    import cv2
    dbg = DebugInfo()
    z_m = depth_to_meters(depth)
    if z_m.shape[:2] != color.shape[:2]:
        raise RuntimeError(
            f'深度 {z_m.shape[1]}x{z_m.shape[0]} 与彩色 '
            f'{color.shape[1]}x{color.shape[0]} 分辨率不一致；请用 '
            'depth_registration:=true 的对齐深度话题')
    roi_mask, quad, note = find_frame_roi(color, params['frame'])
    dbg.roi_mask, dbg.quad, dbg.notes = roi_mask, quad, [note]
    dark_mask, dark_tv = build_dark_mask(color, roi_mask, params['segment'])
    if dark_mask is not None:
        dbg.notes.append(f'dark_thresh={dark_tv}')
    # 平面只用「纸面」点：螺母块/黑块（即便深度无效）外扩后排除，避免高度法偏置
    plane_mask = roi_mask
    if dark_mask is not None:
        ex = max(1, int(params['segment'].get('plane_exclude_dilate', 11)))
        blk = cv2.dilate(dark_mask,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (ex, ex)))
        paper = ((roi_mask > 0) & (blk == 0)).astype(np.uint8) * 255
        if int((paper > 0).sum()) >= int(params['plane'].get('min_points', 80)):
            plane_mask = paper
        else:
            dbg.notes.append('plane_exclude_too_aggressive->full_roi')
    n, d, inl = fit_table_plane(z_m, K, plane_mask, params['plane'])
    dbg.plane_n, dbg.plane_d, dbg.inlier_ratio = n, d, inl
    nuts, height, obj = segment_nuts(color, z_m, K, n, d, roi_mask,
                                     params['segment'], dark_mask=dark_mask)
    dbg.height_m, dbg.obj_mask = height, obj
    nuts = classify_sizes(nuts, params['classify'])
    return nuts, dbg


# ---------------- 可视化 -----------------------------------------------------

def draw_overlay(color, nuts, dbg, params, base_of=None):
    """在彩图上叠加黑框/轮廓/中心/标签/坐标，返回 BGR 画布。"""
    import cv2
    frame = color.copy()
    if dbg.quad is not None:
        cv2.polylines(frame, [np.round(dbg.quad).astype(np.int32)], True,
                      (0, 200, 0), 2)
    for nt in nuts:
        col = LABEL_COLOR_BGR.get(nt.label, (160, 160, 160))
        if nt.contour_uv is not None:
            cv2.drawContours(frame, [nt.contour_uv], -1, col, 2)
        u, v = int(round(nt.u)), int(round(nt.v))
        cv2.drawMarker(frame, (u, v), col, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(frame, (u, v), 4, col, -1)
        tag = nt.label.upper()
        warn = '!' if nt.extra.get('warning') else ''
        txt = f'{tag} {nt.flats_mm:.1f}mm h{nt.height_mm:.1f}{warn}'
        cv2.putText(frame, txt, (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    lines = [f'nuts={len(nuts)}  plane_inlier={(dbg.inlier_ratio or 0) * 100:.0f}%'
             f'  [{dbg.notes[0] if dbg.notes else ""}]']
    for nt in sorted(nuts, key=lambda x: (x.label, x.instance)):
        pc = nt.p_cam * 1000
        line = (f'#{nt.instance} {nt.label.upper()} '
                f'cam=({pc[0]:6.0f},{pc[1]:6.0f},{pc[2]:6.0f})mm '
                f'flats={nt.flats_mm:.1f} h={nt.height_mm:.1f}')
        if base_of is not None:
            pb = base_of(nt.p_cam) * 1000
            line += f' base=({pb[0]:6.0f},{pb[1]:6.0f},{pb[2]:6.0f})'
        lines.append(line)
    lines.append('q=quit  s=save snapshot  d=debug height  p=pause')
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, ln, (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def height_panel(dbg, seg_cfg, size=(480, 540)):
    """高度图伪彩面板（蓝=纸面，红=高起物，无效/框外黑）。"""
    import cv2
    W, H = size
    canvas = np.zeros((H, W, 3), np.uint8)
    if dbg.height_m is None:
        return canvas
    h_hi = seg_cfg['height_max_mm'] / 1000.0
    vis = np.zeros_like(dbg.height_m)
    m = dbg.roi_mask > 0
    vis[m] = np.clip(dbg.height_m[m] / h_hi, 0, 1)
    vis = (vis * 255).astype(np.uint8)
    col = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    col[~m] = 0
    col[(dbg.height_m <= 0) & m] = (40, 40, 40)
    return cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)


# ---------------- ROS 取帧（实机路径用，离线不 import rclpy） ----------------

class RosFrameGrabber:
    """订阅彩色/对齐深度/camera_info，缓存最近若干帧并按【到达时间】配对。

    Gemini2 彩深两路头时间戳带约 3.9s 固定时钟差（传感器各自时钟），
    但 depth_registration 对齐后的配对帧到达本机只差几 ms，故按到达时间配。
    """

    COLOR_Q, DEPTH_Q = 10, 25

    def __init__(self, node, vision, K_file=None):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo
        from cv_bridge import CvBridge
        self.node = node
        self.bridge = CvBridge()
        self.colors = []    # [(arrival, header_stamp, arr)]，旧 -> 新
        self.depths = []
        self.K = K_file

        def push(buf, arr, m, qlen):
            hdr = float(m.header.stamp.sec) + m.header.stamp.nanosec * 1e-9
            buf.append((time.monotonic(), hdr, arr))
            del buf[:-qlen]

        def img(m):
            try:
                push(self.colors, self.bridge.imgmsg_to_cv2(m, 'bgr8'), m,
                     self.COLOR_Q)
            except Exception:
                pass

        def dep(m):
            try:
                push(self.depths, self.bridge.imgmsg_to_cv2(m, 'passthrough'), m,
                     self.DEPTH_Q)
            except Exception:
                pass

        def kci(m):
            if self.K is None:
                self.K = np.array(m.k, float).reshape(3, 3)

        node.create_subscription(Image, vision['color_topic'], img,
                                 qos_profile_sensor_data)
        node.create_subscription(Image, vision['depth_topic'], dep,
                                 qos_profile_sensor_data)
        node.create_subscription(CameraInfo, vision['color_info_topic'], kci,
                                 qos_profile_sensor_data)

    def _match(self, color_item, max_arr_dt):
        ac, _, color = color_item
        ad, _, depth = min(self.depths, key=lambda z: abs(z[0] - ac))
        if abs(ac - ad) > max_arr_dt:
            return None
        if depth.shape[:2] != color.shape[:2]:
            raise RuntimeError(
                f'深度 {depth.shape[1]}x{depth.shape[0]} 与彩色 '
                f'{color.shape[1]}x{color.shape[0]} 分辨率不一致，'
                '请确认 depth_registration:=true')
        return color, depth, self.K

    def get_pair(self, timeout=4.0, max_arr_dt=0.06, fresh=True):
        """等一帧【进入之后到达】的新彩色帧，并按到达时间配深度。

        fresh=True 先冲掉进入前的陈旧帧，避免用上一段运动的拖影画面。
        """
        import rclpy
        t0 = time.monotonic()
        seen = self.colors[-1][0] if (fresh and self.colors) else -1.0
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if self.K is None or not self.colors or not self.depths:
                continue
            newest = self.colors[-1]
            if fresh and newest[0] <= seen:
                continue
            pair = self._match(newest, max_arr_dt)
            if pair is not None:
                return pair
        missing = [name for name, ok in
                   (('color', bool(self.colors)),
                    ('depth', bool(self.depths)),
                    ('K', self.K is not None)) if not ok]
        raise RuntimeError(f'{timeout:.0f}s 内没拿到同步彩深帧（到达时差≤'
                           f'{max_arr_dt * 1000:.0f}ms），缺：{missing}'
                           f'；检查相机话题与 depth_registration:=true')

    def latest(self):
        """查看器高频刷新用：当前最新彩色的最近到达深度（不阻塞、不冲旧帧）。"""
        if not self.colors or not self.depths or self.K is None:
            return None
        pair = self._match(self.colors[-1], 0.1)
        if pair is None:
            return None
        return pair[0], pair[1], pair[2], self.colors[-1][0]


# ---------------- 快照存取 ----------------------------------------------------

def save_snapshot(out_dir, color, depth, K, nuts=None, dbg=None,
                  base_of=None, params=None):
    """保存 color.png + depth.png(uint16 mm) + K.yaml + 可选 detections.json/overlay。"""
    import cv2
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / 'color.png'), color)
    z_m = depth_to_meters(depth)
    dep16 = np.clip(z_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(d / 'depth.png'), dep16)
    (d / 'K.yaml').write_text(
        yaml.safe_dump({'K': np.asarray(K).tolist(),
                        'depth_scale': 0.001,
                        'saved': time.strftime('%Y-%m-%d %H:%M:%S')}),
        encoding='utf-8')
    if nuts is not None:
        data = {'nuts': [nt.to_json(base_of(nt.p_cam) if base_of else None)
                         for nt in nuts]}
        (d / 'detections.json').write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    if dbg is not None and nuts is not None:
        ov = draw_overlay(color, nuts, dbg, params or DEFAULT_PARAMS, base_of)
        cv2.imwrite(str(d / 'overlay.png'), ov)
    return d


def load_snapshot(d):
    """读快照目录 -> (color, depth(uint16 mm), K)。"""
    import cv2
    d = Path(d)
    color = cv2.imread(str(d / 'color.png'), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(d / 'depth.png'), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise RuntimeError(f'快照目录缺 color.png/depth.png：{d}')
    kf = d / 'K.yaml'
    if kf.exists():
        K = np.array(yaml.safe_load(kf.read_text(encoding='utf-8'))['K'], float)
    else:
        raise RuntimeError(f'快照缺 K.yaml：{d}')
    return color, depth, K


def detections_json(nuts, base_of=None):
    return {'nuts': [nt.to_json(base_of(nt.p_cam) if base_of else None)
                     for nt in nuts]}


# ---------------- 查看器 main ------------------------------------------------

def _extrinsics_loader(use_ext, extrinsics_path):
    if not use_ext:
        return None
    p = Path(extrinsics_path)
    if not p.exists():
        return None
    R, t, _ = load_extrinsics(p)
    return lambda pc: R @ np.asarray(pc) + t


def run_viewer(args, params, grabber=None, snapshot=None):
    """实时（grabber）或离线（snapshot）窗口循环。"""
    import cv2
    base_of = _extrinsics_loader(not args.no_base, args.extrinsics)
    show_debug = True
    paused = False
    last_color = last_depth = last_K = None
    last_sig = None
    win = 'nut vision'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    PW, PH = 960, 540
    DW, DH = 480, PH
    cv2.resizeWindow(win, PW + DW, PH)

    def process(color, depth, K):
        nuts, dbg = run_pipeline(color, depth, K, params)
        ov = draw_overlay(color, nuts, dbg, params, base_of)
        p_ov = cv2.resize(ov, (PW, PH))
        if show_debug:
            p_dbg = height_panel(dbg, params['segment'], (DW, DH))
            combo = np.hstack([p_ov, p_dbg])
        else:
            combo = p_ov
        cv2.imshow(win, combo)
        sig = tuple(sorted((round(nt.u / 20), round(nt.v / 20), nt.label)
                           for nt in nuts))
        nonlocal last_sig
        if sig != last_sig:
            last_sig = sig
            for nt in sorted(nuts, key=lambda x: (x.label, x.instance)):
                pc = nt.p_cam * 1000
                line = (f'#{nt.instance} {nt.label.upper()} '
                        f'cam=({pc[0]:.0f},{pc[1]:.0f},{pc[2]:.0f})mm '
                        f'flats={nt.flats_mm:.1f}mm h={nt.height_mm:.1f}mm')
                if base_of:
                    pb = base_of(nt.p_cam) * 1000
                    line += f' base=({pb[0]:.0f},{pb[1]:.0f},{pb[2]:.0f})mm'
                print(line, flush=True)
        return nuts, dbg

    if snapshot is not None:
        color, depth, K = load_snapshot(snapshot)
        nuts, dbg = process(color, depth, K)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('d'), ord('D')):
                show_debug = not show_debug
                nuts, dbg = process(color, depth, K)
            if key in (ord('s'), ord('S')):
                d = save_snapshot(args.save_dir or
                                  Path(__file__).resolve().parent.parent /
                                  '开发资源' / 'nut_sort' / 'snapshots' /
                                  time.strftime('%Y%m%d_%H%M%S'),
                                  color, depth, K, nuts, dbg, base_of, params)
                print(f'快照已保存：{d}')
        cv2.destroyWindow(win)
        return

    import rclpy
    last_stamp = -1.0
    while rclpy.ok():
        rclpy.spin_once(grabber.node, timeout_sec=0.02)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('d'), ord('D')):
            show_debug = not show_debug
        if key in (ord('p'), ord('P')):
            paused = not paused
        latest = grabber.latest()
        if latest is None or latest[3] == last_stamp:
            continue
        if paused:
            continue
        color, depth, K, stamp = latest
        last_stamp = stamp
        try:
            nuts, dbg = process(color, depth, K)
        except RuntimeError as exc:
            canvas = np.zeros((PH, PW, 3), np.uint8)
            cv2.putText(canvas, str(exc)[:80], (20, PH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, canvas)
            continue
        if key in (ord('s'), ord('S')):
            d = save_snapshot(args.save_dir or
                              Path(__file__).resolve().parent.parent /
                              '开发资源' / 'nut_sort' / 'snapshots' /
                              time.strftime('%Y%m%d_%H%M%S'),
                              color, depth, K, nuts, dbg, base_of, params)
            print(f'快照已保存：{d}')
        last_color, last_depth, last_K = color, depth, K
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--params', default=str(
        Path(__file__).resolve().parent.parent /
        '开发资源' / 'nut_sort' / 'nut_vision.yaml'),
        help='参数 yaml（缺省时用代码内默认值）')
    p.add_argument('--color-topic', default='/camera/color/image_raw')
    p.add_argument('--color-info-topic', default='/camera/color/camera_info')
    p.add_argument('--depth-topic', default='/camera/depth/image_raw')
    p.add_argument('--extrinsics', default=None,
                   help='外参 yaml，默认用 开发资源/calibration/gemini2_extrinsics.yaml')
    p.add_argument('--camera-info', default=None,
                   help='内参 yaml，默认同上目录的 gemini2_color_camera_info.yaml')
    p.add_argument('--no-base', action='store_true', help='不显示 base_link 坐标')
    p.add_argument('--once', action='store_true', help='只检测一帧并打印 JSON')
    p.add_argument('--offline', metavar='DIR', help='离线回放快照目录（不需要 ROS）')
    p.add_argument('--save-dir', default=None, help='s 键快照保存目录前缀')
    args = p.parse_args()

    if args.extrinsics is None:
        args.extrinsics = Path(__file__).resolve().parent.parent / \
            '开发资源' / 'calibration' / 'gemini2_extrinsics.yaml'
    if args.camera_info is None:
        args.camera_info = Path(__file__).resolve().parent.parent / \
            '开发资源' / 'calibration' / 'gemini2_color_camera_info.yaml'

    params = load_params(args.params)
    base_of = _extrinsics_loader(not args.no_base, args.extrinsics)

    # ---- 离线路径（不需要 rclpy） ----
    if args.offline:
        if args.once:
            color, depth, K = load_snapshot(args.offline)
            nuts, _ = run_pipeline(color, depth, K, params)
            print(json.dumps(detections_json(nuts, base_of),
                             ensure_ascii=False, indent=2))
            return
        run_viewer(args, params, snapshot=args.offline)
        return

    # ---- 实时路径 ----
    import rclpy
    from rclpy.node import Node
    rclpy.init()
    node = Node('nut_vision')
    K_file = load_camera_K(args.camera_info)
    if K_file is not None:
        print(f'内参来自文件 {Path(args.camera_info).name}')
    grabber = RosFrameGrabber(node, {
        'color_topic': args.color_topic,
        'depth_topic': args.depth_topic,
        'color_info_topic': args.color_info_topic,
    }, K_file=K_file)
    color, depth, K = grabber.get_pair()
    if args.once:
        nuts, dbg = run_pipeline(color, depth, K, params)
        out = detections_json(nuts, base_of)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        node.destroy_node()
        rclpy.shutdown()
        return
    print('实时查看器：q 退出，s 存快照，d 调试高度图，p 暂停。')
    try:
        run_viewer(args, params, grabber=grabber)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
