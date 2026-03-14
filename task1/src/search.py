#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice of HITIRC club - task1
Author Zhihan Wu
Date 2026.1.28
"""

import cv2
import numpy as np
import sys
import os

def detect_red_ball(image_path, output_path='result.png'):
    # Import photo in source
    print(f"Reading: {image_path}")
    img = cv2.imread(image_path)
    print(f"Get it! The size of photo: {img.shape[1]}x{img.shape[0]}")
    # build a draft
    result_img = img.copy()
    # Change the search.png into HSV format
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define what is red in HSV
    # As we all know , red is at either the beginning of the ring or the end of the ring, 
    # so we define the lowest and the highest limit of red here.
    lower_red1 = np.array([0, 100, 100])      
    upper_red1 = np.array([10, 255, 255])    
    lower_red2 = np.array([160, 100, 100])   
    upper_red2 = np.array([180, 255, 255])   
    
    # This step I use mask1 and mask2 to collect all the pixels that are labelled 'red'
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # This step I looked up some former researches to make the red zone a whole part and clear the noising red zone outside
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)  
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel) 
    
    # find the outline finally
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detection_result = {
        'found': False,
        'center_x': -1,
        'center_y': -1,
        'box_x': -1,
        'box_y': -1,
        'box_width': 0,
        'box_height': 0,
        'area': 0
    }
    
    if len(contours) > 0:
        # find the largest outline
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        # delete noise points
        MIN_AREA = 100
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            cv2.circle(result_img, (center_x, center_y), 5, (255, 0, 0), -1)
            
            cv2.putText(result_img, f"Center: ({center_x}, {center_y})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            detection_result.update({
                'found': True,
                'center_x': center_x,
                'center_y': center_y,
                'box_x': x,
                'box_y': y,
                'box_width': w,
                'box_height': h,
                'area': area
            })
    
    # Save my result
    cv2.imwrite(output_path, result_img)
    print(f"\n结果已保存到: {output_path}")
    
    return detection_result

def show_comparison(img1, title1, img2, title2):
    cv2.namedWindow(title1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
    cv2.imshow(title1, img1)
    cv2.imshow(title2, img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    input_image = 'search.png'
    output_image = 'result.png'
    result = detect_red_ball(input_image, output_image)
    if result['found']:
        print(f"Success!")
        print(f"   中心坐标: ({result['center_x']}, {result['center_y']})")
        print(f"   边界框: 左上角({result['box_x']}, {result['box_y']})")
        print(f"           尺寸: {result['box_width']}x{result['box_height']}")
    else:
        print("Not Found")
    original_img = cv2.imread(input_image)
    result_img = cv2.imread(output_image)
    if original_img is not None and result_img is not None:
        show_comparison(original_img, '原图 (search.png)', result_img, '结果 (result.png)')
if __name__ == '__main__':
    main()