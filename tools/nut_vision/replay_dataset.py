#!/usr/bin/env python3
"""Replay captured lossless samples on the original ROS image topics."""
import argparse
import json
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from rosidl_runtime_py.set_message import set_message_fields


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('scene',type=Path)
    parser.add_argument('--rate',type=float,default=1.0,help='Playback speed multiplier')
    parser.add_argument('--loop',action='store_true')
    args=parser.parse_args()
    if args.rate<=0:parser.error('rate must be positive')
    paths=sorted(args.scene.glob('*/metadata.json'))
    if not paths:parser.error('No captured frames in this scene')
    rclpy.init();node=rclpy.create_node('nut_dataset_replay');bridge=CvBridge()
    pubs={}
    for name in ['color','depth']:
        pubs[name]=node.create_publisher(Image,f'/camera/{name}/image_raw',10)
        pubs[name+'_info']=node.create_publisher(CameraInfo,f'/camera/{name}/camera_info',10)
    # Allow subscriber discovery; the replay itself does not move hardware.
    start=time.monotonic()
    while time.monotonic()-start<1:rclpy.spin_once(node,timeout_sec=.05)
    try:
        while rclpy.ok():
            previous=None;play_start=time.monotonic();record_start=None
            for path in paths:
                meta=json.loads(path.read_text())
                header=meta['color_header'];stamp=header['stamp']
                recorded=stamp['sec']+stamp['nanosec']*1e-9
                if record_start is None:record_start=recorded
                due=play_start+(recorded-record_start)/args.rate
                while rclpy.ok() and time.monotonic()<due:rclpy.spin_once(node,timeout_sec=min(.05,due-time.monotonic()) if due>time.monotonic() else 0)
                color=cv2.imread(str(path.parent/'color.png'))
                depth=np.load(path.parent/'depth.npz')['depth']
                for name in ['color','depth']:
                    info=CameraInfo();set_message_fields(info,meta[name+'_camera_info']);pubs[name+'_info'].publish(info)
                    img=bridge.cv2_to_imgmsg(color if name=='color' else depth,encoding='bgr8' if name=='color' else meta['depth_encoding'])
                    set_message_fields(img.header,meta[name+'_header']);pubs[name].publish(img)
                print(path.parent.name,flush=True)
                rclpy.spin_once(node,timeout_sec=0)
            if not args.loop:
                # Let reliable large image messages finish transport before shutdown.
                drain_until=time.monotonic()+1.5
                while rclpy.ok() and time.monotonic()<drain_until:
                    rclpy.spin_once(node,timeout_sec=.05)
                break
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
