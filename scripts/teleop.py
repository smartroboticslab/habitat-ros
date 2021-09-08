#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import curses
import math
import numpy as np
import quaternion
import rospy
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from typing import Tuple



_node_name = "habitat_ros_teleop"
_pose_input_topic = "/habitat/pose"
_pose_input_topic_type = PoseStamped
_pose_output_topic = "/habitat/external_pose"
_pose_output_topic_type = PoseStamped
_path_output_topic = "/mav_sim/goal_path"
_path_output_topic_type = Path



class Movement:
    _fb_step = 0.25 # metres, forward/backward movement step
    _lr_step = 0.25 # metres, left/right movement step
    _ud_step = 0.25 # metres, up/down movement step
    _ro_step = 5 # degrees, left/right rotation step

    def __init__(self, forward: int=0, left: int=0, up: int=0, rotate_left: int=0) -> None:
        self._fb = forward
        self._lr = left
        self._ud = up
        self._ro = rotate_left

    def fb(self) -> float:
        return self._fb * Movement._fb_step

    def lr(self) -> float:
        return self._lr * Movement._lr_step

    def ud(self) -> float:
        return self._ud * Movement._ud_step

    def ro(self) -> float:
        return math.radians(self._ro * Movement._ro_step)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            description="Control the agent with the keyboard")
    parser.add_argument("-p", "--publish-path", action="store_true",
            help="Publish the goal pose as a nav_msgs::Path instead of a "
            "geometry_msgs::PoseStamped. This is meant for use with the MAV "
            "simulator.")
    return parser.parse_args()



def init_pose() -> _pose_input_topic_type:
    return rospy.wait_for_message(_pose_input_topic, _pose_input_topic_type)



def update_pose(p: PoseStamped, m: Movement) -> PoseStamped:
    new_p = PoseStamped()
    new_p.header.stamp = rospy.get_rostime()
    new_p.header.frame_id = p.header.frame_id
    new_p.pose.position = p.pose.position
    q_current = quaternion.quaternion(p.pose.orientation.w,
            p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z)
    R_current = quaternion.as_rotation_matrix(q_current)
    theta_current = math.atan2(R_current[1,0], R_current[0,0])
    # Forwards/backwards movement
    new_p.pose.position.x += m.fb() * math.cos(theta_current)
    new_p.pose.position.y += m.fb() * math.sin(theta_current)
    # Left/right movement
    new_p.pose.position.x += m.lr() * math.cos(theta_current + math.pi/2)
    new_p.pose.position.y += m.lr() * math.sin(theta_current + math.pi/2)
    # Up/down movement
    new_p.pose.position.z = p.pose.position.z + m.ud()
    # Left/right rotation
    theta_new = theta_current + m.ro()
    R = np.eye(3)
    R[0,0] = math.cos(theta_new)
    R[0,1] = -math.sin(theta_new)
    R[1,0] = math.sin(theta_new)
    R[1,1] = math.cos(theta_new)
    q_new = quaternion.as_float_array(quaternion.from_rotation_matrix(R))
    new_p.pose.orientation.x = q_new[1]
    new_p.pose.orientation.y = q_new[2]
    new_p.pose.orientation.z = q_new[3]
    new_p.pose.orientation.w = q_new[0]
    return new_p



def pose_to_path(pose: PoseStamped, new_pose: PoseStamped) -> Path:
    path = Path()
    path.header.stamp = rospy.get_rostime()
    path.header.frame_id = pose.header.frame_id
    path.poses.append(pose)
    path.poses.append(new_pose)
    return path



def wait_for_key(window) -> Tuple[Movement, bool]:
    m = Movement()
    quit = False
    try:
        while True:
            key = window.getkey()
            if key == "Q":
                quit = True
                break
            elif key == "w":
                m = Movement(forward=1)
                break
            elif key == "s":
                m = Movement(forward=-1)
                break
            elif key == "a":
                m = Movement(left=1)
                break
            elif key == "d":
                m = Movement(left=-1)
                break
            elif key == " ":
                m = Movement(up=1)
                break
            elif key == "c":
                m = Movement(up=-1)
                break
            elif key == "q":
                m = Movement(rotate_left=1)
                break
            elif key == "e":
                m = Movement(rotate_left=-1)
                break
            time.sleep(0.05)
    finally:
        curses.endwin()
    return m, quit



def print_waiting_for_pose(window) -> None:
    window.clear()
    window.addstr(1, 0, "Waiting for initial pose of type {} on topic {}".format(_pose_input_topic_type, _pose_input_topic))
    window.refresh()



def print_help(window) -> None:
    window.addstr(0, 0, "Position:")
    window.addstr(2, 0, "Orientation (w,x,y,z):")
    window.addstr(5, 0, "w/s       forwards/backwards")
    window.addstr(6, 0, "a/d       left/right")
    window.addstr(7, 0, "space/c   up/down")
    window.addstr(8, 0, "q/e       rotate left/right")
    window.addstr(9, 0, "Q         quit")



def print_pose_stamped(p: PoseStamped, window) -> None:
    position = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    orientation = [p.pose.orientation.w, p.pose.orientation.x,
            p.pose.orientation.y, p.pose.orientation.z]
    window.move(1, 0)
    window.clrtoeol()
    window.addstr(1, 0, "  " + " ".join(["{: 8.3f}".format(x) for x in position]))
    window.move(3, 0)
    window.clrtoeol()
    window.addstr(3, 0, "  " + " ".join(["{: 8.3f}".format(x) for x in orientation]))



def main_manual(args: argparse.Namespace) -> None:
    # Initialize ROS
    rospy.init_node(_node_name)
    if args.publish_path:
        path_pub = rospy.Publisher(_path_output_topic, _path_output_topic_type, queue_size=10)
    else:
        pose_pub = rospy.Publisher(_pose_output_topic, _pose_output_topic_type, queue_size=10)
    # Initialize curses
    window = curses.initscr()
    try:
        curses.noecho()
        # Wait for the initial pose
        print_waiting_for_pose(window)
        pose = init_pose()
        # Main loop
        quit = False
        print_help(window)
        while not (rospy.is_shutdown() or quit):
            print_pose_stamped(pose, window)
            movement, quit = wait_for_key(window)
            new_pose = update_pose(pose, movement)
            if args.publish_path:
                path_pub.publish(pose_to_path(pose, new_pose))
            else:
                pose_pub.publish(new_pose)
            pose = new_pose
    finally:
        curses.endwin()



if __name__ == "__main__":
    args = parse_args()
    main_manual(args)

