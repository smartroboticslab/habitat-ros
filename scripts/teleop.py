#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import curses
import math
import numpy as np
import os
import quaternion
import rospy
import sys
import time

from geometry_msgs.msg import PoseStamped
from typing import List, Tuple



class Movement:
    _fb_step = 0.25 # metres
    _ud_step = 0.25 # metres
    _lr_step = 5 # degrees

    def __init__(self, f: int=0, u: int=0, l: int=0) -> None:
        self._fb = f
        self._ud = u
        self._lr = l

    def fb(self) -> float:
        return self._fb * Movement._fb_step

    def ud(self) -> float:
        return self._ud * Movement._ud_step

    def lr(self) -> float:
        return math.radians(self._lr * Movement._lr_step)




def program_name() -> str:
    return os.path.basename(sys.argv[0]).replace('.py', '')



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description='Control the agent with the keyboard')
    return parser.parse_args()



def init_pose() -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = 'world'
    p.header.stamp = rospy.get_rostime()
    p.header.seq = 0
    p.pose.position.x = -4.0
    p.pose.position.y =  2.0
    p.pose.position.z = -3.0
    p.pose.orientation.x =  0.0
    p.pose.orientation.y =  0.0
    p.pose.orientation.z =  0.0
    p.pose.orientation.w =  1.0
    return p



def update_pose(p: PoseStamped, m: Movement):
    p.header.stamp = rospy.get_rostime()
    p.header.seq += 1
    q_current = quaternion.quaternion(p.pose.orientation.w, p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z)
    R_current = quaternion.as_rotation_matrix(q_current)
    theta_current = math.atan2(R_current[1,0], R_current[0,0])
    # Forwards/backwards movement
    p.pose.position.x += m.fb() * math.cos(theta_current)
    p.pose.position.y += m.fb() * math.sin(theta_current)
    # Up/down movement
    p.pose.position.z += m.ud()
    # Left/right rotation
    theta_new = theta_current + m.lr()
    R = np.eye(3)
    R[0,0] = math.cos(theta_new)
    R[0,1] = -math.sin(theta_new)
    R[1,0] = math.sin(theta_new)
    R[1,1] = math.cos(theta_new)
    q_new = quaternion.as_float_array(quaternion.from_rotation_matrix(R))
    p.pose.orientation.x =  q_new[1]
    p.pose.orientation.y =  q_new[2]
    p.pose.orientation.z =  q_new[3]
    p.pose.orientation.w =  q_new[0]
    return p



def read_key(window) -> Tuple[Movement, bool]:
    m = Movement(0, 0, 0)
    quit = False
    try:
        while True:
            key = window.getkey()
            if key == "q":
                quit = True
                break
            elif key == "k":
                m = Movement(1, 0, 0)
                break
            elif key == "j":
                m = Movement(-1, 0, 0)
                break
            elif key == "u":
                m = Movement(0, 1, 0)
                break
            elif key == "m":
                m = Movement(0, -1, 0)
                break
            elif key == "h":
                m = Movement(0, 0, 1)
                break
            elif key == "l":
                m = Movement(0, 0, -1)
                break
            time.sleep(0.005)
    finally:
        curses.endwin()
    return m, quit


def print_pose_stamped(p: PoseStamped, window) -> None:
    position = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    orientation = [p.pose.orientation.w, p.pose.orientation.x,
            p.pose.orientation.y, p.pose.orientation.z]
    window.move(1, 0)
    window.clrtoeol()
    window.addstr(1, 0, '  ' + ' '.join(['{: 8.3f}'.format(x) for x in position]))
    window.move(3, 0)
    window.clrtoeol()
    window.addstr(3, 0, '  ' + ' '.join(['{: 8.3f}'.format(x) for x in orientation]))



def main():
    args = parse_args()
    # Initialize ROS
    rospy.init_node('habitat_ros_' + program_name())
    pose_pub = rospy.Publisher('/habitat/external_pose', PoseStamped, queue_size=10)
    # Initialize curses
    window = curses.initscr()
    window.addstr(0, 0, 'Position:')
    window.addstr(2, 0, 'Orientation (w,x,y,z):')
    window.addstr(5, 0, 'k/j   forwards/backwards')
    window.addstr(6, 0, 'u/m   up/down')
    window.addstr(7, 0, 'h/l   rotate left/right')
    curses.noecho()
    # Main loop
    quit = False
    pose = init_pose()
    while not (rospy.is_shutdown() or quit):
        pose_pub.publish(pose)
        print_pose_stamped(pose, window)
        movement, quit = read_key(window) # blocks
        pose = update_pose(pose, movement)



if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, rospy.ROSInterruptException, _curses.error) as e:
        pass
    finally:
        curses.endwin()

