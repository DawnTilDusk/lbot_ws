#!/usr/bin/env python3
"""Conda 推理子进程：原始彩色 PNG -> 原图坐标框和中心 JSON，不加载 ROS。"""
import argparse
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--image')
    p.add_argument('--output')
    p.add_argument('--serve', action='store_true', help='常驻模型，通过 stdin/stdout JSON 行通信')
    p.add_argument('--conf', type=float, default=.5)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    p.add_argument('--allow-pose', action='store_true', help='实时预览允许 needle 两关键点模型')
    args = p.parse_args()
    import torch
    if args.device == 'auto':
        args.device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'推理设备：{args.device}', file=sys.stderr, flush=True)
    from ultralytics import YOLO
    with redirect_stdout(sys.stderr):
        model = YOLO(args.model)
    mapping = {'large': 'l', 'medium': 'm', 'small': 's', 'white': 'white'}
    validate_model(model, args.allow_pose)
    if args.imgsz == 0:
        args.imgsz = 960 if model.task == 'pose' else 640
    if args.serve:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with redirect_stdout(sys.stderr):
                    records = predict(model, request['image'], args, mapping)
                print(json.dumps({'detections': records}, allow_nan=False), flush=True)
            except Exception as exc:
                print(json.dumps({'error': str(exc)}), flush=True)
        return
    if not args.image or not args.output:
        p.error('需要 --image 和 --output，或 --serve')
    records = predict(model, args.image, args, mapping)
    Path(args.output).write_text(json.dumps(records, allow_nan=False), encoding='utf-8')


def validate_model(model, allow_pose=False):
    if model.task == 'pose':
        shape = list(model.model.model[-1].kpt_shape)
        if not allow_pose or model.names != {0: 'needle'} or shape != [2, 3]:
            raise ValueError('只读预览仅支持 needle 的 tip/base 两关键点 Pose 模型')
    elif model.task != 'detect' or set(model.names.values()) not in (
            {'large', 'medium', 'small'}, {'large', 'medium', 'small', 'white'}):
        raise ValueError(f'模型类别或任务不匹配：{model.task} {model.names}')


def pose_keypoints(points, width, height, threshold):
    records = []
    for name, (u, v, confidence) in zip(('tip', 'base'), points):
        valid = (all(math.isfinite(float(x)) for x in (u, v, confidence))
                 and confidence >= threshold and 0 <= u < width and 0 <= v < height
                 and (u != 0 or v != 0))
        records.append(dict(name=name, u=float(u) if valid else None,
                            v=float(v) if valid else None, valid=bool(valid),
                            confidence=float(confidence) if math.isfinite(float(confidence)) else 0.))
    return records


def predict(model, image, args, mapping):
    if str(image).endswith('.npy'):
        import numpy as np
        image = np.load(image, allow_pickle=False)
    result = model.predict(image, conf=args.conf, imgsz=args.imgsz,
                           device=args.device, verbose=False)[0]
    records = []
    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if model.task == 'pose':
            h, w = result.orig_shape
            points = pose_keypoints(result.keypoints.data[index].cpu().tolist(), w, h, args.conf)
            records.append(dict(label='needle', task='pose', confidence=float(box.conf.item()),
                                bbox=[x1, y1, x2, y2], keypoints=points))
        else:
            records.append(dict(label=mapping[result.names[int(box.cls.item())]],
                                confidence=float(box.conf.item()), bbox=[x1, y1, x2, y2],
                                u=(x1+x2)/2, v=(y1+y2)/2))
    return records


if __name__ == '__main__':
    main()
