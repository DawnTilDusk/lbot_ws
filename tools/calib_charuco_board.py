#!/usr/bin/env python3
"""生成可打印的 ChArUco 标定板 PNG 与板配置 YAML。

用法：
    python3 tools/calib_charuco_board.py                 # 用默认参数生成
    python3 tools/calib_charuco_board.py --square-mm 30 --marker-mm 22 \
        --squares-x 8 --squares-y 6 --dpi 300

打印后务必：
  1) 等比例打印（不要“适应页面”缩放）；
  2) 用尺实测棋盘方格边长，把实测值回填到 charuco_board.yaml 的 square_length_m
     （并按比例核对 marker_length_m）；
  3) 板要平整贴在硬板上，不能卷曲反光。

OpenCV 4.6 使用旧版 aruco API。
"""
import argparse

import cv2
import numpy as np
from cv2 import aruco

from calib_common import (BOARD_CONFIG, CALIB_DIR, DEFAULT_DICT_NAME,
                          create_board, save_board_config)


def generate_board_image(board, squares_x, squares_y, square_mm, dpi, margin_mm=10):
    """按物理尺寸 + DPI 渲染高分辨率板图，返回 BGR 图像（单位像素精确对应毫米）。"""
    # 每米像素数
    ppm = dpi / 0.0254
    square_px = max(1, int(round(square_mm * 1e-3 * ppm)))
    board_w_px = square_px * squares_x
    board_h_px = square_px * squares_y
    margin_px = int(round(margin_mm * 1e-3 * ppm))
    img = board.draw((board_w_px, board_h_px), marginSize=margin_px, borderBits=1)
    return img


def self_check(board, cfg, img):
    """对生成的 PNG 重新检测，确认字典与几何自洽。"""
    dictionary = board.dictionary
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=parameters)
    if ids is None or len(ids) == 0:
        return False, '未检测到任何 ArUco 码'
    n_corners = (int(cfg['squares_x']) - 1) * (int(cfg['squares_y']) - 1)
    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners, ids, img, board)
    found = 0 if charuco_corners is None else len(charuco_corners)
    ok = found >= n_corners * 0.9
    msg = f'检测到 ArUco 码 {len(ids)} 个，ChArUco 角点 {found}/{n_corners}'
    return ok, msg


def positive_float(v):
    v = float(v)
    if v <= 0:
        raise argparse.ArgumentTypeError('必须为正数')
    return v


def positive_int(v):
    v = int(v)
    if v <= 0:
        raise argparse.ArgumentTypeError('必须为正整数')
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--squares-x', type=positive_int, default=8, help='X 方向方格数（列），默认 8')
    p.add_argument('--squares-y', type=positive_int, default=6, help='Y 方向方格数（行），默认 6')
    p.add_argument('--square-mm', type=positive_float, default=30.0, help='棋盘方格边长 mm，默认 30')
    p.add_argument('--marker-mm', type=positive_float, default=22.0, help='ArUco 码边长 mm，默认 22')
    p.add_argument('--dictionary', default=DEFAULT_DICT_NAME, help=f'ArUco 字典，默认 {DEFAULT_DICT_NAME}')
    p.add_argument('--dpi', type=positive_int, default=300, help='出图 DPI，默认 300')
    p.add_argument('--out-dir', type=lambda v: __import__('pathlib').Path(v), default=CALIB_DIR)
    args = p.parse_args()

    if args.marker_mm >= args.square_mm:
        p.error('--marker-mm 必须小于 --square-mm（标记内嵌于方格）')

    cfg = {
        'dictionary': args.dictionary,
        'squares_x': args.squares_x,
        'squares_y': args.squares_y,
        'square_length_m': round(args.square_mm / 1000.0, 6),
        'marker_length_m': round(args.marker_mm / 1000.0, 6),
        'note': '打印后用尺实测方格边长并回填 square_length_m；等比例打印，勿缩放。',
    }
    board = create_board(cfg)
    img = generate_board_image(board, args.squares_x, args.squares_y,
                               args.square_mm, args.dpi)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'charuco_board.png'
    yaml_path = out_dir / 'charuco_board.yaml'
    cv2.imwrite(str(png_path), img)
    save_board_config(cfg, yaml_path)

    ok, msg = self_check(board, cfg, img)
    print(f'已生成标定板图像：{png_path}（{img.shape[1]}x{img.shape[0]} px, {args.dpi} DPI）')
    print(f'已生成板配置：    {yaml_path}')
    print(f'自检：{msg} -> {"通过" if ok else "失败"}')
    print()
    print('下一步：')
    print(f'  1. 打印 {png_path.name}（等比例、不要缩放），贴在平整硬板上。')
    print('  2. 用尺实测棋盘方格边长，回填 charuco_board.yaml 的 square_length_m，')
    print('     并按 marker/square 比例核对 marker_length_m。')
    print('  3. 把板牢固夹持到右臂末端后，运行 calib_handeye_sample.py 采样。')
    if not ok:
        print('\n警告：自检未完全通过，请检查 OpenCV 版本或板参数。')


if __name__ == '__main__':
    main()
