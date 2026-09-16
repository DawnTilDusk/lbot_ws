#!/usr/bin/env python3
"""单独测试灵巧手张/合（**不动机械臂、不跑视觉**），排查"张手不放 / 闭合夹不住"。

为什么需要它：主流程里"张手不放"这个现象隔着抓取、交接、回放段很难定位 ——
到底是命令没到手上、还是手收到了但因为受力卡住不动、还是夹持值本身就不对？
这个脚本只发手部话题，把变量降到最少。

用法：

    # 交互（推荐先跑这个）：每组值前按回车。第一组【张开】时把螺母塞进手指，
    # 回车后闭合夹住，再回车张开，看螺母掉不掉
    python3 tools/test_hand_release.py --arm left --hold 2

    # 直接复现现场那一组值：先闭合 1.0s，再张开 1.0s
    python3 tools/test_hand_release.py --arm left --hold 1.0 \\
        --seq "0,0,0,0,0,0" "185,80,255,255,255,255"

    # 反复发同一个"张开"值，看是不是发一次不生效、发几次才松
    python3 tools/test_hand_release.py --arm left --repeat 6 \\
        --seq "185,80,255,255,255,255"

先拿螺母试：让手闭合夹住螺母（`--seq "0,0,0,0,0,0"`），再单独发张开值。
  * 手指张开、螺母掉    -> 手没问题，主流程里是丢包/时序，把 --repeat 调大就行
  * 手指不动            -> 命令没到手上（话题名/命名空间）或手在保护状态
  * 手指张开但螺母还挂着 -> 夹持/张开值的问题，调 hand.open 或给该尺寸单独配 close

值和主流程完全同一套约定：[拇指弯曲, 拇指侧摆, 食, 中, 无名, 小]，255=全张 0=全闭。
不加 --open/--close 时直接读 yaml 的 hand.open / hand.close.<arm>。
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nut_robot import DEFAULT_CONFIG, TaskConfig, TaskError  # noqa: E402


def parse_vals(text, where):
    nums = str(text).replace(',', ' ').split()
    if len(nums) != 6:
        raise TaskError(f'{where} 需要 6 个数（[拇指弯曲,拇指侧摆,食,中,无名,小]），'
                        f'收到 {text!r}')
    try:
        vals = [int(v) for v in nums]
    except ValueError:
        raise TaskError(f'{where} 里有不是整数的内容：{text!r}')
    for v in vals:
        if not 0 <= v <= 255:
            raise TaskError(f'{where} 每路取值必须 0..255，收到 {v}')
    return vals


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--arm', choices=('left', 'right'), default='left')
    p.add_argument('--open', dest='open_vals', default=None,
                   help='张开值，6 个数；缺省用 yaml hand.open')
    p.add_argument('--close', dest='close_vals', default=None,
                   help='闭合值，6 个数；缺省用 yaml hand.close.<arm>')
    p.add_argument('--seq', nargs='+', default=None, metavar='VALS',
                   help='按顺序发这 6 个数一组的值，每组保持 --hold 秒；'
                        '给了就不走默认的开/合/开流程')
    p.add_argument('--hold', type=float, default=1.5, help='每组值保持多少秒（默认 1.5）')
    p.add_argument('--repeat', type=int, default=1,
                   help='每组值连发几轮（默认 1；排查丢包就调大，如 5）')
    p.add_argument('--gap', type=float, default=0.25, help='连发轮之间的间隔秒数（默认 0.25）')
    p.add_argument('--speed', default=None, help='6 个数，覆盖 yaml hand.speed')
    p.add_argument('--force', default=None, help='6 个数，覆盖 yaml hand.force')
    p.add_argument('--no-speed-force', action='store_true',
                   help='不先下发 speed/force（默认会先下发 yaml 里的 hand.speed/hand.force）')
    args = p.parse_args()

    cfg = TaskConfig(args.config)

    open_vals = (list(cfg.hand_open_vals) if args.open_vals is None
                 else parse_vals(args.open_vals, '--open'))
    close_vals = (cfg.close_for(args.arm, 'l') if args.close_vals is None
                  else parse_vals(args.close_vals, '--close'))

    if args.seq:
        steps = [(f'命令行 --seq 第 {i} 组', parse_vals(s, f'--seq 第 {i} 组'))
                 for i, s in enumerate(args.seq, 1)]
    else:
        steps = [('张开（yaml hand.open）', open_vals),
                 ('闭合（yaml/--close）', close_vals),
                 ('张开（yaml hand.open）', open_vals)]

    import rclpy
    from std_msgs.msg import UInt8MultiArray

    rclpy.init()
    node = rclpy.create_node(f'test_hand_release_{args.arm}')
    base = f'{cfg.namespace}/{args.arm}_hand'
    pubs = {k: node.create_publisher(UInt8MultiArray, f'{base}/set_l6_{k}', 10)
            for k in ('joint', 'speed', 'force')}
    print(f'手部话题：{base}/set_l6_{{joint,speed,force}}')
    print(f'臂={args.arm}  保持 {args.hold:g}s  连发 {args.repeat} 轮  间隔 {args.gap:g}s')

    def send(kind, vals):
        msg = UInt8MultiArray()
        msg.data = [int(v) for v in vals]
        for _ in range(3):      # 话题不锁存，连发防丢包（与主流程一致）
            pubs[kind].publish(msg)
            rclpy.spin_once(node, timeout_sec=0.03)

    def hold(sec):
        """保持期间持续 spin，避免话题队列积压。"""
        end = time.monotonic() + max(0.0, sec)
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        # 等驱动订阅上话题，否则头几条会丢
        print('等 1.0s 让驱动订阅上话题...')
        hold(1.0)

        if not args.no_speed_force:
            speed = (list(cfg.hand_speed) if args.speed is None
                     else parse_vals(args.speed, '--speed'))
            force = (list(cfg.hand_force) if args.force is None
                     else parse_vals(args.force, '--force'))
            print(f'下发 speed={speed}  force={force}')
            send('speed', speed)
            send('force', force)
            hold(0.3)

        for idx, (what, vals) in enumerate(steps, 1):
            if not args.seq:
                # 默认流程留出"手动塞螺母"的窗口：每组值前等一次回车
                try:
                    input(f'\n按回车发第 {idx}/{len(steps)} 组：{what} {vals} ... ')
                except EOFError:
                    pass
            print(f'\n[{idx}/{len(steps)}] {what}  {vals}')
            for r in range(max(1, args.repeat)):
                send('joint', vals)
                if r < args.repeat - 1:
                    hold(args.gap)
            hold(args.hold)
            print(f'  已保持 {args.hold:g}s，请观察手指位置与螺母是否离手')
        print('\n结束（手停在最后一组值）。')
    except KeyboardInterrupt:
        print('\n已中断。')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except TaskError as exc:
        print(f'[中止] {exc}', file=sys.stderr)
        sys.exit(1)
