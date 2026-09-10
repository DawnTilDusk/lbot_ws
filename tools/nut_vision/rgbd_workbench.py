#!/usr/bin/env python3
"""Visible RGB-D collection and nut recognition workbench; no motion commands."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import cv2
import message_filters
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from rosidl_runtime_py.convert import message_to_ordereddict
from rgbd_core import detect, annotate


ROOT = Path(__file__).resolve().parent
OUTPUT = Path.home() / 'nut_capture'


def write_json(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def timestamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9


class CameraNode(Node):
    def __init__(self, window):
        super().__init__('nut_vision_workbench')
        self.window = window
        self.bridge = CvBridge()
        self.infos = {}
        self.last_processed = 0
        self.last_received = time.monotonic()
        self.transforms = {}
        self.robot_observations = {}
        self.publisher = self.create_publisher(String, '/nut_vision/detections', 10)
        self.info_subs = [self.create_subscription(
            CameraInfo, f'/camera/{key}/camera_info',
            lambda msg,key=key:self.infos.__setitem__(key,msg), qos_profile_sensor_data,
        ) for key in ['color','depth']]
        self.color_sub = message_filters.Subscriber(self,Image,'/camera/color/image_raw',qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self,Image,'/camera/depth/image_raw',qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color_sub,self.depth_sub],20,.02)
        self.sync.registerCallback(self.on_pair)
        qos = QoSProfile(depth=50,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.tf_sub = self.create_subscription(TFMessage,'/tf_static',self.on_tf,qos)
        self.robot_subs = []
        self.discovery_timer = self.create_timer(5,self.discover_robot)
        self.known_robot_topics = set()

    def on_tf(self,msg):
        for t in msg.transforms:
            self.transforms[(t.header.frame_id,t.child_frame_id)] = message_to_ordereddict(t)

    def discover_robot(self):
        for topic,types in self.get_topic_names_and_types():
            if not topic.startswith('/robot') or topic in self.known_robot_topics:
                continue
            cls = PoseStamped if 'geometry_msgs/msg/PoseStamped' in types else JointState if 'sensor_msgs/msg/JointState' in types else None
            if cls:
                self.robot_subs.append(self.create_subscription(cls,topic,lambda m,t=topic:self.robot_observations.__setitem__(t,message_to_ordereddict(m)),qos_profile_sensor_data))
                self.known_robot_topics.add(topic)

    def publish(self,payload):
        self.publisher.publish(String(data=json.dumps(payload,ensure_ascii=False,allow_nan=False)))

    def on_pair(self,color_msg,depth_msg):
        now = time.monotonic()
        self.last_received = now
        if now-self.last_processed < .18:
            return
        self.last_processed = now
        try:
            if len(self.infos) != 2:
                raise ValueError('Waiting for both CameraInfo streams')
            ci,di = self.infos['color'],self.infos['depth']
            if (color_msg.width,color_msg.height) != (depth_msg.width,depth_msg.height):
                raise ValueError('Color/depth dimensions differ; enable depth registration')
            if color_msg.header.frame_id != depth_msg.header.frame_id or not color_msg.header.frame_id:
                raise ValueError('Color and depth are not in the same optical frame')
            for image,info in [(color_msg,ci),(depth_msg,di)]:
                if (image.width,image.height) != (info.width,info.height) or image.header.frame_id != info.header.frame_id:
                    raise ValueError('CameraInfo does not match image dimensions/frame')
                if abs(timestamp(image)-timestamp(info))>.5:
                    raise ValueError('CameraInfo timestamp is stale')
            if not np.allclose(ci.k,di.k,rtol=1e-4,atol=1e-4):
                raise ValueError('Aligned depth and color intrinsics differ')
            delta = abs(timestamp(color_msg)-timestamp(depth_msg))*1000
            if delta > self.window.config['max_sync_ms']:
                raise ValueError('RGB/depth timestamp difference exceeds configured tolerance')
            color = self.bridge.imgmsg_to_cv2(color_msg,'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg,'passthrough')
            config = dict(self.window.config)
            if depth_msg.encoding == '16UC1':
                config['depth_scale_m'] = .001
            elif depth_msg.encoding == '32FC1':
                config['depth_scale_m'] = 1.
            else:
                raise ValueError('Unsupported depth encoding: '+depth_msg.encoding)
            info = message_to_ordereddict(ci)
            results,edges = detect(color,depth,info,config)
            self.window.update_tracks(results,now)
            payload = {
                'schema_version':1,'status':'ok','task_mode':config['task_mode'],
                'header':message_to_ordereddict(color_msg.header),'units':'m',
                'coordinate_scope':'camera_only_no_robot_extrinsics',
                'sync_delta_ms':delta,'valid_for_ms':500,'detections':results,
                'quality_is_probability':False,
            }
            self.publish(payload)
            metadata = {
                'color_header':message_to_ordereddict(color_msg.header),
                'depth_header':message_to_ordereddict(depth_msg.header),
                'color_camera_info':info,'depth_camera_info':message_to_ordereddict(di),
                'color_encoding':color_msg.encoding,'saved_color_encoding':'BGR PNG via OpenCV',
                'depth_encoding':depth_msg.encoding,'depth_scale_m':config['depth_scale_m'],
                'sync_delta_ms':delta,'parameters':config,
                'robot_observations':dict(self.robot_observations),
                'robot_observations_are_synchronized':False,
                'tf_static':list(self.transforms.values()),
                'detector_predictions':payload,
                'ground_truth':None,
            }
            self.window.present(color,depth,metadata,results,edges)
        except Exception as exc:
            self.window.status.setText('INVALID: '+str(exc))
            self.window.latest = None
            self.window.previous = []
            self.publish({'schema_version':1,'status':'invalid','error':str(exc),'detections':[]})


class Workbench(QtWidgets.QWidget):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.config = json.loads((ROOT/'vision_config.json').read_text())
        self.output = args.output
        self.output.mkdir(parents=True,exist_ok=True)
        self.setWindowTitle('Nut Vision - RGB-D 采集 / 螺母检测 / 相机坐标定位')
        self.resize(1450,900)
        self.latest = None
        self.previous = []
        self.track_counter = 0
        self.burst_remaining = 0
        self.last_save = 0
        self.scene_dir = None
        self.scene_settings = None
        self.saved_total = 0
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('相机坐标系结果 | 不控制机器人 | 黄色=分类或深度待确认；绿色=分类与定位均有效')
        layout.addWidget(title)
        views = QtWidgets.QHBoxLayout()
        self.rgb_view = QtWidgets.QLabel('等待同步 RGB-D...')
        self.depth_view = QtWidgets.QLabel('深度预览（颜色仅用于显示）')
        for view in [self.rgb_view,self.depth_view]:
            view.setMinimumSize(480,270)
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setStyleSheet('background:#182229;color:white')
            views.addWidget(view,1)
        layout.addLayout(views,4)
        self.status = QtWidgets.QLabel('等待相机...')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QtWidgets.QTableWidget(0,8)
        self.table.setHorizontalHeaderLabels(['ID','类别','外观','实测估计AF/mm','相机XYZ/m','平面RMS/mm','连续稳定','无效/未知原因'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(210)
        layout.addWidget(self.table,1)
        form = QtWidgets.QGridLayout()
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(['basic','advanced']); self.mode.setCurrentText(self.config['task_mode'])
        self.mode.currentTextChanged.connect(self.change_mode)
        self.split = QtWidgets.QComboBox(); self.split.addItems(['train','validation'])
        self.kind = QtWidgets.QComboBox();self.kind.addItems(['nuts','calibration_board'])
        self.note = QtWidgets.QLineEdit(); self.note.setPlaceholderText('填写位置/光照/背景/遮挡；标定板规格；是否手已移开')
        self.note.setText('初始场景；真实尺寸待提供；未完成人工标注')
        form.addWidget(QtWidgets.QLabel('任务'),0,0);form.addWidget(self.mode,0,1)
        form.addWidget(QtWidgets.QLabel('数据用途'),0,2);form.addWidget(self.split,0,3)
        form.addWidget(self.kind,0,4);form.addWidget(self.note,0,5,1,3)
        self.roi_edit = QtWidgets.QLineEdit(','.join(str(x) for x in self.config['roi']))
        roi_button=QtWidgets.QPushButton('应用工作区ROI');roi_button.clicked.connect(self.apply_roi)
        form.addWidget(QtWidgets.QLabel('ROI归一化 x0,y0,x1,y1'),1,0,1,2);form.addWidget(self.roi_edit,1,2,1,3);form.addWidget(roi_button,1,5)
        new_button=QtWidgets.QPushButton('新场景 / New scene');new_button.clicked.connect(self.new_scene)
        single=QtWidgets.QPushButton('采集1对 / Save pair');single.clicked.connect(self.save_pair)
        burst=QtWidgets.QPushButton('采集10对 / Capture 10 pairs');burst.clicked.connect(self.start_burst)
        stop=QtWidgets.QPushButton('停止采集');stop.clicked.connect(lambda:self.stop_capture('采集已停止'))
        form.addWidget(new_button,2,0,1,2);form.addWidget(single,2,2,1,2);form.addWidget(burst,2,4,1,2);form.addWidget(stop,2,6,1,2)
        layout.addLayout(form)
        dims=QtWidgets.QHBoxLayout();self.measurements={}
        for name,profile in self.config['profiles'].items():
            box=QtWidgets.QGroupBox(name+' 实测(mm)');fields=QtWidgets.QFormLayout(box)
            af=QtWidgets.QDoubleSpinBox();af.setRange(0,150);af.setDecimals(2);af.setValue(profile['across_flats_mm'] or 0);af.setSpecialValueText('未知')
            height=QtWidgets.QDoubleSpinBox();height.setRange(0,100);height.setDecimals(2);height.setValue(profile['height_mm'] or 0);height.setSpecialValueText('未知')
            fields.addRow('对边宽',af);fields.addRow('厚度',height);dims.addWidget(box);self.measurements[name]=(af,height)
        apply=QtWidgets.QPushButton('保存实测尺寸');apply.clicked.connect(self.apply_measurements);dims.addWidget(apply)
        layout.addLayout(dims)
        self.capture_status=QtWidgets.QLabel('未采集。不同摆放使用“新场景”；验证集必须重新摆放，不能只切换相邻帧。')
        self.capture_status.setWordWrap(True);layout.addWidget(self.capture_status)
        screenshot=QtWidgets.QPushButton('保存当前界面截图 / Screenshot')
        screenshot.clicked.connect(lambda:self.grab().save(str(self.output/'workbench_screen.png')))
        layout.addWidget(screenshot)
        self.node=CameraNode(self)
        self.timer=QtCore.QTimer();self.timer.timeout.connect(self.tick);self.timer.start(20)
        self.last_stale_message=0

    def change_mode(self,mode):
        self.config['task_mode']=mode;self.previous=[]

    def apply_roi(self):
        try:
            roi=[float(x.strip()) for x in self.roi_edit.text().split(',')]
            if len(roi)!=4 or not all(0<=x<=1 for x in roi) or not roi[0]<roi[2] or not roi[1]<roi[3]:raise ValueError()
            self.config['roi']=roi;self.previous=[]
            write_json(ROOT/'vision_config.json',self.config)
            self.capture_status.setText('ROI已更新，仅限制固定工作区，不是人工选择抓取目标。')
        except ValueError:self.capture_status.setText('ROI必须是0到1之间的 x0,y0,x1,y1，且右下角大于左上角。')

    def apply_measurements(self):
        for name,(af,height) in self.measurements.items():
            self.config['profiles'][name]['across_flats_mm']=af.value() or None
            self.config['profiles'][name]['height_mm']=height.value() or None
        write_json(ROOT/'vision_config.json',self.config)
        self.previous=[]
        self.capture_status.setText('尺寸已保存。0表示未知；算法估计尺寸不能代替卡尺实测。')

    def new_scene(self):
        self.burst_remaining=0;self.scene_dir=None;self.scene_settings=None
        self.capture_status.setText('已准备新场景。请改变摆放/条件，选训练或验证用途，填写说明后采集。')

    def stop_capture(self,message):
        self.burst_remaining=0;self.capture_status.setText(message)

    def start_burst(self):
        if self.latest is None:
            self.capture_status.setText('还没有有效同步帧，不能采集。');return
        self.burst_remaining=10
        self.capture_status.setText('开始采集10对，约5秒；正式物体样本请把手移出画面。')

    def tick(self):
        if not rclpy.ok():
            self.timer.stop()
            return
        # Four 30Hz input streams exceed a single callback per 20ms GUI tick.
        # Drain a bounded batch so image and CameraInfo queues do not starve.
        pump_start=time.monotonic()
        for _ in range(12):
            if not rclpy.ok():return
            rclpy.spin_once(self.node,timeout_sec=0)
            if time.monotonic()-pump_start>.012:break
        now=time.monotonic()
        if now-self.node.last_received>1.5:
            self.latest=None;self.previous=[]
            self.status.setText('STALE：没有新同步帧，当前画面不可作为实时结果')
            if now-self.last_stale_message>1:
                self.node.publish({'schema_version':1,'status':'stale','detections':[]});self.last_stale_message=now
            if self.burst_remaining:self.stop_capture('采集暂停：相机数据已过期')

    def update_tracks(self,results,now):
        used=set()
        for obj in results:
            candidates=[(np.linalg.norm(np.array(obj['center_pixel'])-old['center_pixel']),j,old)
                        for j,old in enumerate(self.previous) if j not in used and now-old['_time']<.6]
            match=min(candidates,key=lambda x:x[0]) if candidates else None
            stable=False;count=1
            if match and match[0]<35:
                _,j,old=match;used.add(j);identity=old['track_id']
                if obj['position_valid'] and old['position_valid'] and obj['class']==old['class']:
                    stable=np.linalg.norm(np.array(obj['position_camera_m'])-old['position_camera_m'])<.010
                    count=old['stable_frames']+1 if stable else 1
            else:
                self.track_counter+=1;identity=self.track_counter
            obj.update(track_id=identity,stable_frames=count,stable=count>=3,
                       usable_camera_observation=bool(count>=3 and obj['position_valid'] and obj['class_valid'] and obj.get('in_task', True)))
        self.previous=[dict(obj,_time=now) for obj in results]

    def image_on_label(self,label,image):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        q=QtGui.QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QtGui.QImage.Format_RGB888).copy()
        label.setPixmap(QtGui.QPixmap.fromImage(q).scaled(label.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))

    def present(self,color,depth,metadata,results,edges):
        self.latest=(color,depth,metadata,time.monotonic())
        self.image_on_label(self.rgb_view,annotate(color,results,self.config['roi']))
        meters=depth.astype(float)*metadata['depth_scale_m']
        scaled=np.clip(np.nan_to_num(meters)*255/3,0,255).astype(np.uint8)
        preview=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO);preview[(~np.isfinite(meters))|(meters<=0)]=0
        self.image_on_label(self.depth_view,preview)
        usable=sum(x['usable_camera_observation'] for x in results)
        self.status.setText(f"LIVE {color.shape[1]}x{color.shape[0]} | 同步差 {metadata['sync_delta_ms']:.1f}ms | 候选 {len(results)} | 连续稳定且分类有效 {usable} | /nut_vision/detections")
        self.table.setRowCount(len(results))
        for row,obj in enumerate(results):
            pos=obj['position_camera_m']
            values=[str(obj['track_id']),obj['class'],obj['appearance'],
                    '-' if obj['across_flats_mm'] is None else f"{obj['across_flats_mm']:.1f}",
                    '-' if pos is None else ', '.join(f'{v:.3f}' for v in pos),
                    '-' if 'plane_rms_mm' not in obj else f"{obj['plane_rms_mm']:.2f}",
                    str(obj['stable_frames']),'; '.join(obj['invalid_reasons']+([] if obj['class_valid'] else [obj['class_reason']]))]
            for col,value in enumerate(values):self.table.setItem(row,col,QtWidgets.QTableWidgetItem(value))
        if self.burst_remaining and time.monotonic()-self.last_save>.5:
            if self.save_pair():self.burst_remaining-=1

    def save_pair(self):
        if self.latest is None or time.monotonic()-self.latest[3]>.5:
            self.stop_capture('没有新鲜同步帧，未保存');return False
        if shutil.disk_usage(self.output).free<1024**3:
            self.stop_capture('磁盘剩余不足1GiB，已停止采集');return False
        current=(self.split.currentText(),self.kind.currentText(),self.note.text())
        if self.scene_settings is not None and current!=self.scene_settings:
            self.stop_capture('同一场景不能中途更改用途或说明。请先点击“新场景”。');return False
        try:
            color,depth,metadata,_=self.latest
            if self.scene_dir is None:
                scene_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
                self.scene_dir=self.output/'dataset'/current[0]/scene_id
                self.scene_dir.mkdir(parents=True,exist_ok=False)
                self.scene_settings=current
                write_json(self.scene_dir/'scene.json',{'scene_id':scene_id,'split':current[0],
                    'kind':current[1],'notes':current[2],'created_utc':datetime.now(timezone.utc).isoformat(),
                    'ground_truth_status':'unannotated','camera_to_robot_calibration':'deferred',
                    'calibration_board_specification':None,'measured_profiles':self.config['profiles']})
                driver_parameters=self.output/'camera_parameters_720p.yaml'
                if driver_parameters.exists():
                    shutil.copy2(driver_parameters,self.scene_dir/'camera_driver_parameters.yaml')
            stamp=metadata['color_header']['stamp']
            frame_dir=self.scene_dir/f"{stamp['sec']}_{stamp['nanosec']:09d}"
            if frame_dir.exists():return False
            frame_dir.mkdir()
            if not cv2.imwrite(str(frame_dir/'color.png'),color):raise RuntimeError('color PNG write failed')
            np.savez_compressed(frame_dir/'depth.npz',depth=depth)
            metadata=dict(metadata,depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                          color_sha256=hashlib.sha256(color.tobytes()).hexdigest())
            write_json(frame_dir/'metadata.json',metadata)
            self.last_save=time.monotonic();self.saved_total+=1
            self.capture_status.setText(f'已保存 {self.saved_total} 对 | 剩余 {max(0,self.burst_remaining-1)} | {self.scene_dir}')
            print('SAVED',frame_dir,flush=True)
            return True
        except Exception as exc:
            self.stop_capture('保存失败：'+str(exc));return False

    def closeEvent(self,event):
        self.timer.stop();self.node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        event.accept()


def offline(args):
    p=args.offline
    color=cv2.imread(str(p/'color.png'))
    depth=np.load(p/'depth.npz')['depth'] if (p/'depth.npz').exists() else np.load(p/'depth.npy',allow_pickle=False)
    meta=json.loads((p/'metadata.json').read_text())
    info=meta.get('color_camera_info') or meta['infos']['color']
    config=json.loads((ROOT/'vision_config.json').read_text())
    config['depth_scale_m']=meta.get('depth_scale_m',.001)
    results,_=detect(color,depth,info,config)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'offline_detections.json',results)
    cv2.imwrite(str(args.output/'offline_preview.png'),annotate(color,results,config['roi']))
    print(json.dumps([{k:v for k,v in obj.items() if k not in ('outer_contour','hole_contour')} for obj in results],indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--offline',type=Path)
    args=parser.parse_args()
    if args.offline:
        offline(args);return
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    app=QtWidgets.QApplication(sys.argv[:1]);window=Workbench(args);window.show()
    signal.signal(signal.SIGINT,lambda *_:window.close())
    signal.signal(signal.SIGTERM,lambda *_:window.close())
    sys.exit(app.exec_())


if __name__=='__main__':main()
