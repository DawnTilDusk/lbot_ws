# 四类别螺母训练结果

从 `../nuts_mixed_20260914/weights/best.pt` 迁移微调，类别扩展为 large、medium、small、white。训练 71 张，验证 31 张，完成 100 轮。

最优权重：`nut_white_best.pt`；最后一轮：`weights/last.pt`。

| 类别 | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| 全部 | 0.995 | 1.000 | 0.995 | 0.941 |
| large | 0.991 | 1.000 | 0.995 | 0.995 |
| medium | 0.993 | 1.000 | 0.995 | 0.965 |
| small | 0.997 | 1.000 | 0.995 | 0.965 |
| white | 0.998 | 1.000 | 0.995 | 0.840 |

以上为训练器对 best.pt 的最终验证输出，按终端精度记录。验证集与训练集场景相近，标注由模型辅助生成并复核，不能代表新环境性能。

预测预览：`val_batch0_pred.jpg`、`val_batch1_pred.jpg`。完整参数见 `args.yaml`，来源和哈希见 `nut_white_best.json`。

四类权重已同步为 nut_white_best.pt，nut_yolo_infer.py 与实时预览已支持 white。原三类 nut_best.pt 保留供抓取流程使用，白色螺母的机械臂业务尚未配置。

完整训练产物来源：`/home/dawntildusk/nut_vision/runs/nuts_white4_20260914/`；文中未随权重提交的训练图表、参数及 last.pt 位于该目录。
