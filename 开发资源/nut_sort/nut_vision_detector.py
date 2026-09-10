#!/usr/bin/env python3
"""抓放框架（nut_pick_place）的螺母视觉 external 检测器适配器。

把 tools/nut_vision.py 的纯视觉管线接到既有检测器契约：
  __init__(self, node, sub_cfg, K)
  detect(expected) -> [Detection(label, p_cam, u, v, z, extra)]

- 真机（--execute，node 存在）：订阅彩色+对齐深度，detect 时等一帧【进入之后到达】
  的同步帧对跑管线，直接给相机光学系 p_cam（桌面平面交点），DepthPixelDetector
  包装层对 p_cam 原样透传。
- dry-run（node=None）：默认返回空列表并打印提示；若在 detector 段配置
  snapshot: <快照目录>，则离线跑该快照，便于不带相机核对检测结果。

同尺寸多颗（框架 validate_detections 每档只允许一颗）由 pick 决定：
  pick: leftmost  每档只回画面最左一颗（默认），其余打印日志
  pick: largest   每档只回对边距最大的一颗
  pick: all       全部返回（多颗同档时框架会按既有保护中止，留给后续扩展）

nut_task.yaml detector 段可加：
  params: 开发资源/nut_sort/nut_vision.yaml   # 视觉参数（默认就是它）
  pick: leftmost
  # snapshot: 开发资源/nut_sort/snapshots/20260910_150000  # dry-run 离线快照
"""
import sys
from pathlib import Path

import numpy as np

_WS = Path(__file__).resolve().parents[2]
_TOOLS = _WS / 'tools'
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from nut_detectors import Detection          # noqa: E402
from nut_vision import (RosFrameGrabber, load_params, load_snapshot,  # noqa: E402
                        run_pipeline)

DEFAULT_PARAMS = _WS / '开发资源' / 'nut_sort' / 'nut_vision.yaml'


class NutVisionDetector:
    def __init__(self, node, sub_cfg=None, K=None):
        self.node = node
        self.cfg = sub_cfg or {}
        params_path = self.cfg.get('params') or str(DEFAULT_PARAMS)
        if not Path(params_path).is_absolute():
            params_path = str(_WS / params_path)
        self.params = load_params(params_path)
        self.pick = self.cfg.get('pick', 'leftmost')
        self.grabber = None

        if node is None:
            return  # dry-run：不订话题，detect() 走快照/空列表分支

        from camera_pick_move import load_camera_K
        K_file = K
        if K_file is None:
            K_file = load_camera_K(self.cfg['_vision']['camera_info_path'])
        self.grabber = RosFrameGrabber(node, self.cfg['_vision'], K_file=K_file)

    # ---------------- 输出组装 ----------------
    @staticmethod
    def _to_detection(nt):
        return Detection(
            label=nt.label,
            p_cam=np.asarray(nt.p_cam, float).reshape(3),
            u=int(round(nt.u)),
            v=int(round(nt.v)),
            z=float(nt.p_cam[2]),
            extra={
                'diameter_mm': round(nt.flats_mm, 2),
                'corners_mm': round(nt.corners_mm, 2),
                'height_mm': round(nt.height_mm, 2),
                'area_mm2': round(nt.area_mm2, 1),
                'instance': nt.instance,
                'score': round(float(nt.score), 2),
                **{k: v for k, v in nt.extra.items() if v not in ('', None)},
            },
        )

    def _select(self, nuts, expected):
        """按 expected 过滤标签，并按 pick 策略把每档压到至多一颗。"""
        wanted = tuple(expected)
        chosen, dropped, unknown = [], [], []
        for nt in nuts:
            if nt.label == '?':
                unknown.append(nt)
            elif nt.label in wanted:
                chosen.append(nt)
        if unknown:
            print(f'[nut_vision] {len(unknown)} 颗未能分入 s/m/l，已忽略：'
                  + ', '.join(f'#{n.instance}({n.flats_mm:.1f}mm)'
                              for n in unknown))
        if self.pick == 'all':
            return chosen
        out = []
        for lab in wanted:
            group = [n for n in chosen if n.label == lab]
            if not group:
                continue
            if self.pick == 'largest':
                keep = max(group, key=lambda n: n.flats_mm)
            else:  # leftmost：画面从左到右
                keep = min(group, key=lambda n: n.u)
            out.append(keep)
            for n in group:
                if n is not keep:
                    dropped.append(n)
        if dropped:
            print('[nut_vision] 同尺寸多颗，按 pick=%s 只取一颗，舍弃：' % self.pick
                  + ', '.join(f'{n.label.upper()}#{n.instance}@x{n.u:.0f}'
                              f'({n.flats_mm:.1f}mm)' for n in dropped))
        return out

    # ---------------- 契约入口 ----------------
    def detect(self, expected):
        if self.node is None:
            snap = self.cfg.get('snapshot')
            if not snap:
                print('[nut_vision] dry-run 无 ROS 节点：跳过真机检测'
                      '（可在 detector 段配 snapshot: <快照目录> 离线核对）')
                return []
            snap = snap if Path(snap).is_absolute() else str(_WS / snap)
            color, depth, K = load_snapshot(snap)
            nuts, _ = run_pipeline(color, depth, K, self.params)
            nuts = self._select(nuts, expected)
            return [self._to_detection(n) for n in nuts]

        color, depth, K = self.grabber.get_pair()
        nuts, dbg = run_pipeline(color, depth, K, self.params)
        print(f'[nut_vision] 画面检出 {len(nuts)} 颗，平面内点率 '
              f'{(dbg.inlier_ratio or 0) * 100:.0f}%：')
        for nt in sorted(nuts, key=lambda x: (x.label, x.u)):
            pc = nt.p_cam * 1000
            warn = '  ⚠ 中心不对称，疑似相触/缺角' if nt.extra.get('warning') else ''
            print(f'  #{nt.instance} {nt.label.upper()} '
                  f'cam=({pc[0]:.0f},{pc[1]:.0f},{pc[2]:.0f})mm '
                  f'flats={nt.flats_mm:.1f}mm h={nt.height_mm:.1f}mm{warn}')
        nuts = self._select(nuts, expected)
        return [self._to_detection(n) for n in nuts]
