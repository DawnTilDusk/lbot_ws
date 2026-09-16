# 新视角追加52轮训练

累计 100 轮；本段 52 轮。从上一轮 last.pt 新建优化器微调，lr0=0.0003，patience=0。

| 模型 | 验证 mAP50 | 验证 mAP50–95 |
|---|---:|---:|
| previous_best | 0.9950 | 0.8906 |
| continued_best | 0.9950 | 0.8959 |

按验证集推荐：continued_best

追加模型测试结果：{"metrics/precision(B)": 0.9871494662961894, "metrics/recall(B)": 0.99548732538643, "metrics/mAP50(B)": 0.995, "metrics/mAP50-95(B)": 0.8674446975850401, "fitness": 0.8674446975850401}

同场景测试集已在上轮评估中使用，不能视为新的独立测试。145张待复核图片仍未参与。已同步到仓库 weights/nut_white_best.pt；旧四类模型已备份。三类抓取权重 nut_best.pt 保持原样。


训练来源：/home/dawntildusk/nut_vision/runs/nuts_newview_continue52_20260916
