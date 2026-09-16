# 铁杆两关键点模型训练结果

训练实际完成 100 轮；从 yolo11n-pose.pt 开始，imgsz=960，GPU=RTX 5070 Laptop。数据：72 train / 20 val / 10 test；另 3 张遮挡图不参加训练。

最优模型：`weights/needle_pose_best.pt`，按训练框架验证集 fitness 选择。

SHA256：`5d23052ad37fe0e7e7bfd6d78db93663de0ef2a7d91b3737dd6a4f7ade78ffd4`

## 标准指标

```json
{
  "val": {
    "metrics/precision(B)": 0.9958184666122226,
    "metrics/recall(B)": 1.0,
    "metrics/mAP50(B)": 0.995,
    "metrics/mAP50-95(B)": 0.7751897759103642,
    "metrics/precision(P)": 0.9958184666122226,
    "metrics/recall(P)": 1.0,
    "metrics/mAP50(P)": 0.995,
    "metrics/mAP50-95(P)": 0.9949999999999999,
    "fitness": 1.7701897759103642
  },
  "test": {
    "metrics/precision(B)": 0.9940805565587063,
    "metrics/recall(B)": 1.0,
    "metrics/mAP50(B)": 0.995,
    "metrics/mAP50-95(B)": 0.7130761904761904,
    "metrics/precision(P)": 0.9940805565587063,
    "metrics/recall(P)": 1.0,
    "metrics/mAP50(P)": 0.995,
    "metrics/mAP50-95(P)": 0.9949999999999999,
    "fitness": 1.7080761904761903
  }
}
```

## 相对标注的原图像素误差

```json
{
  "val": {
    "images": 20,
    "matched": 20,
    "false_positives": 0,
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5,
    "mean_error_px": {
      "tip": 3.2235728410959714,
      "base": 3.0331069218004143
    },
    "median_error_px": {
      "tip": 2.869104381062701,
      "base": 3.1069857488092256
    },
    "p95_error_px": {
      "tip": 5.453743691260921,
      "base": 5.585709832891021
    },
    "both_within_5px": 15
  },
  "test": {
    "images": 10,
    "matched": 10,
    "false_positives": 0,
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5,
    "mean_error_px": {
      "tip": 4.434390538598083,
      "base": 3.2537723226648865
    },
    "median_error_px": {
      "tip": 4.139526763244106,
      "base": 2.624949087762052
    },
    "p95_error_px": {
      "tip": 7.242959447449179,
      "base": 6.998403490101619
    },
    "both_within_5px": 6
  }
}
```

预览：`/home/dawntildusk/nut_vision/runs/needle_pose_20260916/keypoint_preview/review.html`。原标注为圆圈、预测为十字。像素误差仅针对匹配检测，相对人工视觉标注，不是实际毫米误差。所有样本来自同次采集，方向比较单一，不能据此保证独立场景、遮挡场景或机器人插入精度。

模型没有替换螺母检测权重，也未接入机械臂动作。
