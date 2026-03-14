#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os

def train_model():
    # 1. 加载预训练模型 (推荐从 yolov8n.pt 开始微调)
    model = YOLO('yolov8n.pt')

    # 2. 设置数据集配置
    
    dataset_root = "/home/zhihan/Desktop/task3/dataset" # <--- 请根据你的实际路径修改这里
    
    # 临时创建 data.yaml 内容
    data_config = f"""
train: {dataset_root}/images
val: {dataset_root}/images  # 如果没有验证集，暂时用训练集代替，正式使用请分开
nc: 1
names: ['door_handle']
"""
    yaml_path = "data_temp.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_config)
    
    print(f"配置文件已生成：{yaml_path}")
    print("开始训练...")

    # 3. 开始训练
    results = model.train(
        data=yaml_path,
        epochs=100,          # 训练轮数
        imgsz=640,           # 输入图片大小
        batch=16,            # 批次大小 (显存不够改小，比如 8 或 4)
        device=0,            # 使用 GPU 0
        amp=False,
        project='runs/detect',
        name='door_handle_train',
        exist_ok=True
    )
    
    print("训练完成！模型保存在 runs/detect/door_handle_train/weights/best.pt")
    
    # 清理临时文件
    if os.path.exists(yaml_path):
        os.remove(yaml_path)

if __name__ == '__main__':
    train_model()