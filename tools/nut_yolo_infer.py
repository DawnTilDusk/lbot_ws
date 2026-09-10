#!/usr/bin/env python3
"""Conda 推理子进程：原始彩色 PNG -> 原图坐标框和中心 JSON，不加载 ROS。"""
import argparse
import json
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
    args = p.parse_args()
    from ultralytics import YOLO
    with redirect_stdout(sys.stderr):
        model = YOLO(args.model)
    mapping = {'large': 'l', 'medium': 'm', 'small': 's'}
    if set(model.names.values()) != set(mapping):
        raise ValueError(f'模型类别不匹配：{model.names}')
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


def predict(model, image, args, mapping):
    result = model.predict(image, conf=args.conf, imgsz=args.imgsz,
                           device=args.device, verbose=False)[0]
    records = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        records.append(dict(label=mapping[result.names[int(box.cls.item())]],
                            confidence=float(box.conf.item()), bbox=[x1, y1, x2, y2],
                            u=(x1+x2)/2, v=(y1+y2)/2))
    return records


if __name__ == '__main__':
    main()
