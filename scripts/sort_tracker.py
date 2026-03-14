#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class KalmanFilter:
    def __init__(self, bbox):
        # bbox: [x, y, w, h]
        self.dt = 0.05 
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        self.x = np.array([[cx], [cy], [0], [0]]) 
        self.P = np.eye(4) * 1000
        self.F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * 10
        self.Q = np.eye(4) * 0.1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, measurement):
        z = np.array(measurement).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()

class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.kf = KalmanFilter(bbox)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = deque([bbox], maxlen=5) 

    def predict(self):
        pred = self.kf.predict()
        self.time_since_update += 1
        self.age += 1
        return pred 

    def update(self, bbox):
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        self.kf.update([cx, cy])
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)

class SORTTracker:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def iou(self, box1, box2):
        b1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
        b2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]
        xx1 = max(b1[0], b2[0]); yy1 = max(b1[1], b2[1])
        xx2 = min(b1[2], b2[2]); yy2 = min(b1[3], b2[3])
        inter = max(0, xx2-xx1) * max(0, yy2-yy1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        if len(detections) == 0:
            for t in self.tracks: t.predict()
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return []

        predictions = []
        for t in self.tracks:
            pred_center = t.predict()
            if t.history:
                last_w, last_h = t.history[-1][2], t.history[-1][3]
                pred_box = [pred_center[0]-last_w/2, pred_center[1]-last_h/2, last_w, last_h]
                predictions.append(pred_box)
            else: predictions.append([0,0,0,0])

        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self.iou(pred, det)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_track_ids, matched_det_ids = set(), set()
        result_tracks = []

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.iou_threshold:
                self.tracks[i].update(detections[j])
                matched_track_ids.add(i); matched_det_ids.add(j)
                if self.tracks[i].hits >= self.min_hits: result_tracks.append(self.tracks[i])

        for j, det in enumerate(detections):
            if j not in matched_det_ids:
                new_track = Track(self.next_id, det)
                self.next_id += 1
                self.tracks.append(new_track)
                # 如果不需要等待确认，可以直接加入 result_tracks，这里为了稳定等待3帧
                # if new_track.hits >= self.min_hits: result_tracks.append(new_track) 

        for i, t in enumerate(self.tracks):
            if i not in matched_track_ids and t.time_since_update < self.max_age:
                if t.hits >= self.min_hits: result_tracks.append(t)

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        return result_tracks