#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import os
import sys

# 导入同目录下的 sort_tracker
sys.path.append(os.path.dirname(__file__))
from sort_tracker import SORTTracker

class YOLOTrackerNode:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 获取模型路径参数
        model_path = rospy.get_param('~model_path', '')
        
        if not model_path or not os.path.exists(model_path):
            rospy.logwarn("未找到模型文件，请使用默认 YOLOv8n 或检查参数")
            self.model = YOLO('yolov8n.pt')
        else:
            rospy.loginfo(f"加载模型: {model_path}")
            self.model = YOLO(model_path)
        
        self.tracker = SORTTracker()
        
        # 订阅相机图像 (默认话题，可在launch文件中修改)
        self.sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        # 发布带跟踪框的图像
        self.pub = rospy.Publisher("/yolo/tracked_image", Image, queue_size=1)
        
        rospy.loginfo("YOLO Tracker Node Started")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(str(e))
            return

        # YOLO 推理
        # results[0].boxes.xywh 返回的是 [x_center, y_center, w, h]，正好符合 SORT 输入格式
        results = self.model(cv_image, verbose=False)
        boxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes is not None else np.empty((0, 4))
        
        detections = []
        if len(boxes) > 0:
            detections = boxes.tolist()

        # SORT 跟踪更新
        tracked_objects = self.tracker.update(detections)

        # 绘制结果
        for track in tracked_objects:
            if track.history:
                # 取最新的历史框
                x, y, w, h = track.history[-1]
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # 画框
                color = (0, 255, 0) # Green
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                
                # 画ID
                label = f"ID: {track.track_id}"
                cv2.putText(cv_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 发布结果
        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(str(e))

if __name__ == '__main__':
    rospy.init_node('yolo_tracker_node')
    node = YOLOTrackerNode()
    rospy.spin()