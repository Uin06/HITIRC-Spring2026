#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import threading
import sys
import tty
import termios
from geometry_msgs.msg import Twist

# 全局变量用于控制速度
current_velocity = Twist()
current_velocity.linear.x = 0.0
current_velocity.linear.y = 0.0
current_velocity.angular.z = 0.0

# 线程锁确保多线程安全
velocity_lock = threading.Lock()

def getch():
    """获取单个按键输入（非阻塞）"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_listener():
    """监听键盘输入并更新速度指令"""
    print("Robot control activated! Use 'w', 's', 'a', 'd' for movement, 'q', 'e' for rotation, press any key to exit.")
    print("w: forward, s: backward, a: left, d: right, q: rotate left, e: rotate right")
    
    while not rospy.is_shutdown():
        key = getch()
        if key == 'w':
            with velocity_lock:
                current_velocity.linear.x = 0.2
                current_velocity.linear.y = 0.0
                current_velocity.angular.z = 0.0
        elif key == 's':
            with velocity_lock:
                current_velocity.linear.x = -0.2
                current_velocity.linear.y = 0.0
                current_velocity.angular.z = 0.0
        elif key == 'a':
            with velocity_lock:
                current_velocity.linear.x = 0.0
                current_velocity.linear.y = 0.2
                current_velocity.angular.z = 0.0
        elif key == 'd':
            with velocity_lock:
                current_velocity.linear.x = 0.0
                current_velocity.linear.y = -0.2
                current_velocity.angular.z = 0.0
        elif key == 'q':
            with velocity_lock:
                current_velocity.linear.x = 0.0
                current_velocity.linear.y = 0.0
                current_velocity.angular.z = 0.5
        elif key == 'e':
            with velocity_lock:
                current_velocity.linear.x = 0.0
                current_velocity.linear.y = 0.0
                current_velocity.angular.z = -0.5
        else:  # 无按键时停止
            with velocity_lock:
                current_velocity.linear.x = 0.0
                current_velocity.linear.y = 0.0
                current_velocity.angular.z = 0.0

def main():
    rospy.init_node('robot_keyboard_controller', anonymous=True)
    
    # 创建/cmd_vel发布者
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    # 启动键盘监听线程
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # 设置发布频率 (10Hz)
    rate = rospy.Rate(10)
    
    print("Robot control started. Press keys to move the robot.")
    print("Use 'w', 's', 'a', 'd' for movement, 'q', 'e' for rotation.")
    
    try:
        while not rospy.is_shutdown():
            # 获取当前速度
            with velocity_lock:
                current_vel = Twist()
                current_vel.linear.x = current_velocity.linear.x
                current_vel.linear.y = current_velocity.linear.y
                current_vel.angular.z = current_velocity.angular.z
            
            # 发布速度指令
            cmd_vel_pub.publish(current_vel)
            rate.sleep()
            
    except rospy.ROSInterruptException:
        print("\nRobot control stopped.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nControl terminated by user.")
