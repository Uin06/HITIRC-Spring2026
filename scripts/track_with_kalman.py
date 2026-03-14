import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ---------------- 1. 自定义卡尔曼滤波类 ----------------
class KalmanFilterTracker:
    def __init__(self, track_id):
        self.track_id = track_id
        # 状态向量 [x, y, w, h, vx, vy, vw, vh] (中心点坐标，宽高，速度)
        # 这里简化为只跟踪中心点 (x, y) 和 速度 (vx, vy)，假设宽高变化不大或通过其他方式处理
        # 为了演示简单性，我们跟踪 [x, y, w, h] 及其速度，共8维状态
        self.kf = cv2.KalmanFilter(8, 4) 
        
        # 状态转移矩阵 F (假设匀速运动)
        # x' = x + vx, y' = y + vy, ...
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], np.float32)

        # 测量矩阵 H (我们只能测量 x, y, w, h)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], np.float32)

        # 噪声协方差 (需要根据实际场景调整)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2  # 过程噪声
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1 # 测量噪声
        
        self.is_initialized = False
        self.predicted_bbox = None
        self.age = 0 # 跟踪器存活帧数
        self.time_since_update = 0 # 多久没被检测到

    def initialize(self, bbox):
        """用第一个检测框初始化"""
        x, y, w, h = bbox
        self.kf.statePost[:4] = np.array([x, y, w, h], np.float32).reshape(-1, 1)
        # 初始速度设为0
        self.kf.statePost[4:] = 0.0 
        self.is_initialized = True
        self.predicted_bbox = self._format_bbox(self.kf.statePost[:4])

    def predict(self):
        """预测下一帧的位置 (即使没有检测到)"""
        if not self.is_initialized:
            return None
        pred = self.kf.predict()
        self.time_since_update += 1
        self.predicted_bbox = self._format_bbox(pred[:4])
        return self.predicted_bbox

    def update(self, bbox):
        """用新的检测框修正预测"""
        if not self.is_initialized:
            self.initialize(bbox)
            return self._format_bbox(self.kf.statePost[:4])
        
        x, y, w, h = bbox
        measurement = np.array([x, y, w, h], np.float32).reshape(-1, 1)
        self.kf.correct(measurement)
        self.time_since_update = 0
        self.predicted_bbox = self._format_bbox(self.kf.statePost[:4])
        return self.predicted_bbox

    def _format_bbox(self, state):
        """将状态向量 [x,y,w,h] 转换为 (x1, y1, x2, y2)"""
        x, y, w, h = state.flatten()
        # 确保宽高为正
        w = max(0, w)
        h = max(0, h)
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        return (x1, y1, x2, y2)

# ---------------- 2. 主程序逻辑 ----------------

def iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def main():
    # 加载模型
    model_path = "/home/zhihan/Desktop/task3/runs/detect/runs/detect/door_handle_train/weights/best.pt"
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    trackers = {} # id -> KalmanFilterTracker 对象
    next_track_id = 0
    max_age = 30  # 如果超过30帧没检测到，认为目标消失
    iou_threshold = 0.3 # IoU 匹配阈值

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO 检测
        results = model(frame, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                cx, cy = x1 + w/2, y1 + h/2
                detections.append({'bbox': (cx, cy, w, h), 'xyxy': (x1, y1, x2, y2)})

        # 2. 数据关联与卡尔曼滤波更新
        matched_detections = set()
        
        # 先更新已有的跟踪器
        for tid, tracker in list(trackers.items()):
            # 预测当前位置
            pred_bbox = tracker.predict() # 返回 (x1, y1, x2, y2) 格式需要转换一下逻辑，上面类返回的是xyxy
            
            # 尝试在当前的 detections 中寻找匹配的框
            best_iou = 0
            best_det_idx = -1
            
            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                
                # 将 detection 的 (cx, cy, w, h) 转为 xyxy 用于计算 IoU
                dx, dy, dw, dh = det['bbox']
                det_xyxy = (dx - dw/2, dy - dh/2, dx + dw/2, dy + dh/2)
                
                # 注意：tracker.predicted_bbox 在类里已经是 xyxy 了
                # 但上面的类实现中 predicted_bbox 是 xyxy，我们需要统一
                # 让我们重新检查一下类里的返回值，_format_bbox 返回的是 xyxy
                
                # 为了匹配，我们需要把 detection 也转成中心点形式给 update? 
                # 不，update 接收的是 (cx, cy, w, h)
                
                curr_iou = iou(pred_bbox, det_xyxy)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_det_idx = idx
            
            if best_iou > iou_threshold:
                # 匹配成功，更新卡尔曼滤波
                matched_det = detections[best_det_idx]
                tracker.update(matched_det['bbox']) # 传入 cx, cy, w, h
                matched_detections.add(best_det_idx)
                
                # 绘制：使用修正后的位置
                final_box = tracker.predicted_bbox # 这里的 predicted_bbox 其实是 update 后最新的 state
                # 重新从 state 获取最新的 xyxy
                # 为了简单，直接用 update 返回的值或者再次 format
                # 这里我们信任 tracker 内部状态，重新格式化一下
                state = tracker.kf.statePost[:4].flatten()
                x, y, w, h = state
                x1, y1, int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # 匹配失败，继续使用预测值 (卡尔曼滤波发挥作用的地方！)
                # 如果时间太久，删除跟踪器
                if tracker.time_since_update > max_age:
                    del trackers[tid]
                    continue
                
                # 绘制预测框 (虚线表示是预测的，非检测到的)
                x1, y1, x2, y2 = map(int, tracker.predicted_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # 红色表示预测
                cv2.putText(frame, f"ID:{tid} (Pred)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 3. 创建新跟踪器 (对于未匹配的检测)
        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                cx, cy, w, h = det['bbox']
                new_tracker = KalmanFilterTracker(next_track_id)
                new_tracker.initialize((cx, cy, w, h))
                trackers[next_track_id] = new_tracker
                next_track_id += 1
                
                x1, y1, x2, y2 = det['xyxy']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"New:{next_track_id-1}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO + Kalman Filter Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()