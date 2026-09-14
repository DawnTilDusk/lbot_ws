#!/usr/bin/env python3
"""只读预览：实时相机或离线照片 -> 中心像素、相机 XYZ、base_link XYZ。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml
from nut_robot import TaskConfig, DEFAULT_CONFIG, TaskError
from nut_yolo import infer_pixels, locate, detect_once, annotate
from camera_pick_move import load_extrinsics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--image', type=Path, help='离线彩色照片；不传则读取实时相机')
    p.add_argument('--depth', type=Path, help='同帧对齐深度：16位毫米 PNG 或浮点米 NPY')
    p.add_argument('--camera-info', type=Path, help='离线图像对应的内参 YAML（K/width/height）')
    p.add_argument('--output', type=Path, default=Path('recordings/yolo_preview')/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args = p.parse_args()
    if args.depth and (not args.image or not args.camera_info):
        p.error('--depth 需要 --image 和 --camera-info')
    cfg = TaskConfig(args.config, detector_override='yolo')
    detections = None
    snapshot = None
    if args.image:
        color = cv2.imread(str(args.image))
        if color is None:
            raise TaskError(f'不能读取照片：{args.image}')
        records = infer_pixels(color, cfg.detector_raw)
        if args.depth:
            depth = (np.load(args.depth, allow_pickle=False) if args.depth.suffix == '.npy'
                     else cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED))
            if depth is None:
                raise TaskError('不能读取深度文件')
            info = yaml.safe_load(args.camera_info.read_text())
            if (info['height'],info['width']) != color.shape[:2]:
                raise TaskError('离线内参与彩色照片分辨率不匹配')
            K = np.array(info['K']).reshape(3,3)
            detections = locate(records, depth, K, color.shape, cfg.detector_raw)
            snapshot = dict(color=color, depth=depth, K=K)
    else:
        detections, snapshot = detect_once(cfg)
        color = snapshot['color']
    if detections is not None:
        R,t,ext = load_extrinsics(cfg.extrinsics_path)
        if ext.get('parent_frame') != 'base_link':
            raise TaskError('外参父坐标系不是 base_link')
        records = [dict(label=d.label, confidence=d.extra['confidence'], bbox=d.extra['bbox'],
                        u=d.u, v=d.v, z=d.z, p_cam=d.p_cam.tolist(),
                        p_base=(R@d.p_cam+t).tolist(), frame='base_link',
                        depth_method=d.extra['depth_method']) for d in detections]
    args.output.mkdir(parents=True, exist_ok=False)
    canvas = annotate(color, records, located=detections,
                      status='nut_yolo_preview: boxes + center-patch depth (no motion)')
    for r in records:
        print(json.dumps(r, ensure_ascii=False))
    for name,img in [('color.png',color),('centers.jpg',canvas)]:
        if not cv2.imwrite(str(args.output/name),img):
            raise TaskError(f'无法保存 {name}')
    report=dict(detections=records, has_depth=detections is not None,
                note='框中心+中心邻域深度；中心孔可能测到桌面。未执行运动。')
    if snapshot is not None:
        np.save(args.output/'depth.npy',snapshot['depth'])
        (args.output/'camera_info.yaml').write_text(yaml.safe_dump(dict(
            width=color.shape[1],height=color.shape[0],K=snapshot['K'].reshape(-1).tolist())))
        report['snapshot']={k:v for k,v in snapshot.items() if k not in ('color','depth','K')}
    (args.output/'detections.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(f'已保存：{args.output.resolve()}；仅预览，未调用机械臂。')
    if detections is None:
        print('仅有彩色照片：输出像素中心，不能计算真实三维坐标。')


if __name__ == '__main__':
    try:
        main()
    except (TaskError, OSError, ValueError, ImportError) as exc:
        sys.exit(f'预览失败：{exc}')
