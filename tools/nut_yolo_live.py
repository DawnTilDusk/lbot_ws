#!/usr/bin/env python3
"""实时 YOLO 中心及坐标预览；仅订阅相机，不创建机械臂运动客户端。"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
import sys

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from nut_robot import TaskConfig, DEFAULT_CONFIG, TaskError
from nut_yolo import WORKSPACE, locate
from camera_pick_move import load_extrinsics


def choose_pair(colors, depths, now, max_age, max_skew, after):
    """最新未处理彩色帧及时间最接近的深度；不积压历史帧。"""
    for c in reversed(colors):
        if c[0] <= after or now-c[1] > max_age:
            continue
        candidates = [d for d in depths if now-d[1] <= max_age and abs(c[0]-d[0]) <= max_skew]
        if candidates:
            return c, min(candidates, key=lambda d: abs(c[0]-d[0]))
    return None


class Worker:
    def __init__(self, cfg):
        self.temp = tempfile.TemporaryDirectory(prefix='nut_live_')
        self.folder = Path(self.temp.name)
        self.log = (self.folder/'worker.log').open('w+')
        model = Path(cfg.get('model', 'weights/nut_best.pt')).expanduser()
        if not model.is_absolute(): model = WORKSPACE/model
        python = Path(cfg.get('python','~/miniconda3/envs/nut-yolo/bin/python')).expanduser()
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env['PYTHONNOUSERSITE'] = '1'
        self.timeout = float(cfg.get('inference_timeout',30))
        self.process = subprocess.Popen([str(python),str(WORKSPACE/'tools/nut_yolo_infer.py'),
            '--serve','--model',str(model),'--device',str(cfg.get('device','cpu')),
            '--conf',str(cfg.get('confidence',.5)),'--imgsz',str(cfg.get('imgsz',640))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env,bufsize=1)

    def infer(self, frame):
        path = self.folder/'frame.png'
        if not cv2.imwrite(str(path),frame): raise TaskError('无法写入推理帧')
        self.process.stdin.write(json.dumps({'image':str(path)})+'\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout],[],[],self.timeout)[0]:
            raise TaskError('模型推理超时')
        line = self.process.stdout.readline()
        if not line:
            self.log.flush()
            raise TaskError('推理进程退出：'+(self.folder/'worker.log').read_text()[-1500:])
        result=json.loads(line)
        if 'error' in result: raise TaskError(result['error'])
        return result['detections']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.process.stdin.close(); self.process.stdout.close()
        self.log.close(); self.temp.cleanup()


def coordinates(records, pair, info, R, t, ext, cfg):
    rows=[]
    c,d=pair
    error=None
    if (info['height'],info['width']) != c[2].shape[:2]: error='内参与图像尺寸不匹配'
    if info['frame'] != ext.get('child_frame') or ext.get('parent_frame') != 'base_link':
        error='外参坐标系不匹配'
    for record in records:
        row=dict(record)
        try:
            if error: raise TaskError(error)
            det=locate([record],d[2],info['K'],c[2].shape,cfg)
            if not det: continue
            row.update(p_cam=det[0].p_cam.tolist(),p_base=(R@det[0].p_cam+t).tolist(),z=det[0].z)
        except TaskError as exc: row['error']=str(exc)
        rows.append(row)
    return rows


def render(frame, rows, status, font):
    canvas=frame.copy()
    for r in rows:
        x1,y1,x2,y2=map(round,r['bbox']);u,v=round(r['u']),round(r['v'])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.drawMarker(canvas,(u,v),(0,0,255),cv2.MARKER_CROSS,18,2)
    panel=np.zeros((max(canvas.shape[0],100+len(rows)*115),580,3),np.uint8)
    full=np.zeros((panel.shape[0],canvas.shape[1]+580,3),np.uint8)
    full[:canvas.shape[0],:canvas.shape[1]]=canvas
    image=PILImage.fromarray(cv2.cvtColor(full,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(image)
    x=canvas.shape[1]+12
    draw.text((x,10),'只读实时检测 | 坐标单位：米',font=font,fill='white')
    draw.text((x,42),status,font=font,fill='yellow')
    names={'l':'大','m':'中','s':'小'}
    for i,r in enumerate(rows):
        y=90+i*115
        draw.text((max(0,round(r['bbox'][0])),max(0,round(r['bbox'][1])-28)),
                  f'{names[r["label"]]} {r["confidence"]:.2f}',font=font,fill='yellow')
        draw.text((x,y),f'{names[r["label"]]} 置信度 {r["confidence"]:.2f}  中心 ({r["u"]:.1f}, {r["v"]:.1f})',font=font,fill='white')
        if 'p_cam' in r:
            for offset,key,label in [(30,'p_cam','相机'),(60,'p_base','基座')]:
                draw.text((x,y+offset),label+': '+', '.join(f'{v:+.4f}' for v in r[key]),font=font,fill='cyan')
        else: draw.text((x,y+30),'坐标无效：'+r.get('error','未知')[:28],font=font,fill='orange')
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--device',help='cpu 或 0（第一块 GPU）')
    p.add_argument('--conf',type=float)
    p.add_argument('--scale',type=float,default=.75)
    p.add_argument('--no-preview',action='store_true',help='无窗口测试，配合 --duration')
    p.add_argument('--duration',type=float,default=0,help='运行秒数，0 不限')
    p.add_argument('--output',type=Path,default=WORKSPACE/'recordings/yolo_live'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--font',default='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    args=p.parse_args()
    if not np.isfinite([args.scale,args.duration]).all() or args.scale<=0 or args.duration<0: p.error('scale 必须为正，duration 非负')
    if args.conf is not None and not 0<args.conf<=1: p.error('conf 必须在 (0,1]')
    cfg=TaskConfig(args.config); options=dict(cfg.detector_raw)
    if args.device is not None: options['device']=args.device
    if args.conf is not None: options['confidence']=args.conf
    R,t,ext=load_extrinsics(cfg.extrinsics_path)
    font=ImageFont.truetype(args.font,22)
    import rclpy
    from sensor_msgs.msg import Image,CameraInfo
    from rclpy.qos import qos_profile_sensor_data
    from cv_bridge import CvBridge
    rclpy.init();node=rclpy.create_node('nut_yolo_live');bridge=CvBridge()
    colors,depths=deque(maxlen=15),deque(maxlen=15);info={}
    def receive(m,buffer,encoding):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if stamp<=0:return
        try:buffer.append((stamp,time.monotonic(),bridge.imgmsg_to_cv2(m,encoding).copy()))
        except Exception as exc:node.get_logger().warning(str(exc))
    def camera(m):info.update(K=np.array(m.k).reshape(3,3),width=m.width,height=m.height,frame=m.header.frame_id)
    node.create_subscription(Image,cfg.color_topic,lambda m:receive(m,colors,'bgr8'),qos_profile_sensor_data)
    node.create_subscription(Image,cfg.depth_topic,lambda m:receive(m,depths,'passthrough'),qos_profile_sensor_data)
    node.create_subscription(CameraInfo,cfg.color_info_topic,camera,qos_profile_sensor_data)
    worker=None;executor=ThreadPoolExecutor(max_workers=1);future=None
    started=time.monotonic();last_stamp=-1;result=None;pending=None;updates=0;fps=0.
    age_limit=float(options.get('max_frame_age',1));result_limit=float(options.get('max_result_age',10))
    args.output.mkdir(parents=True,exist_ok=False)
    def save(view,rows,pair):
        stamp=datetime.now().strftime('%H%M%S_%f')
        if not cv2.imwrite(str(args.output/f'{stamp}.jpg'),view):raise TaskError('保存画面失败')
        (args.output/f'{stamp}.json').write_text(json.dumps(dict(detections=rows,color_stamp=pair[0][0],depth_stamp=pair[1][0]),ensure_ascii=False,indent=2))
    print('正在加载模型；q/ESC 退出，s 保存当前检测图及坐标。仅订阅相机。',flush=True)
    try:
        worker=Worker(options)
        while rclpy.ok() and (not args.duration or time.monotonic()-started<args.duration):
            rclpy.spin_once(node,timeout_sec=.005);now=time.monotonic()
            if future is not None and future.done():
                records=future.result();pair,intrinsics,sent=pending
                if now-min(pair[0][1],pair[1][1])<=result_limit:
                    rows=coordinates(records,pair,intrinsics,R,t,ext,options)
                    result=(pair,rows);updates+=1;fps=1/max(now-sent,1e-6)
                future=None
            if future is None and info:
                pair=choose_pair(colors,depths,now,age_limit,float(options.get('max_skew',.1)),last_stamp)
                if pair:
                    last_stamp=pair[0][0];pending=(pair,dict(info),now)
                    future=executor.submit(worker.infer,pair[0][2])
            fresh=bool(colors and depths) and now-colors[-1][1]<=age_limit and now-depths[-1][1]<=age_limit
            usable=result is not None and fresh and now-min(result[0][0][1],result[0][1][1])<=age_limit
            if usable:
                pair,rows=result
                view=render(pair[0][2],rows,f'{fps:.1f} 次/秒 | 帧龄 {now-pair[0][1]:.2f}s',font)
            else:
                frame=colors[-1][2] if colors else np.zeros((720,1280,3),np.uint8)
                view=render(frame,[],'等待新鲜配对帧/推理；旧坐标已隐藏',font)
            if not args.no_preview:
                cv2.imshow('YOLO live | q quit | s save',cv2.resize(view,None,fx=args.scale,fy=args.scale))
                key=cv2.waitKey(1)&255
                if key in (ord('q'),27):break
                if key==ord('s') and usable:save(view,rows,pair)
                if cv2.getWindowProperty('YOLO live | q quit | s save',cv2.WND_PROP_VISIBLE)<1:break
        if result is not None:
            pair,rows=result
            # Exit snapshot is explicitly timestamped, even if camera later stopped.
            save(render(pair[0][2],rows,'最后一次检测快照（非实时）',font),rows,pair)
        print(f'完成 {updates} 次坐标更新；输出：{args.output}',flush=True)
    except KeyboardInterrupt:pass
    finally:
        if worker is not None:worker.close()
        executor.shutdown(wait=True,cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        if not args.no_preview:cv2.destroyAllWindows()


if __name__=='__main__':
    try:main()
    except (TaskError,OSError,ValueError,ImportError) as exc:sys.exit(f'实时预览失败：{exc}')
