#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np
import os
import quaternion
import rospy
import sys

from geometry_msgs.msg import PoseStamped
from math import cos, sin, pi
from typing import List



def printerr(*args, **kwargs) -> None:
    """Print to stderr prefixed with the program name"""
    error_prefix = os.path.basename(sys.argv[0]) + ': error:'
    print(error_prefix, *args, file=sys.stderr, **kwargs)



def program_name() -> str:
    return os.path.basename(sys.argv[0]).replace('.py', '')



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Publish debug poses '
            'depending on the provided dataset')
    parser.add_argument('dataset', metavar='DATASET', nargs='?',
            default='1LXtFkjw3qL',
            help='(default: 1LXtFkjw3qL)')
    args = parser.parse_args()
    return args



def pose_1LXtFkjw3qL() -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = rospy.get_rostime()
    p.pose.position.x = -4.0
    p.pose.position.y =  2.0
    p.pose.position.z = -3.0
    p.pose.orientation.x =  0.0
    p.pose.orientation.y =  0.0
    p.pose.orientation.z =  0.0
    p.pose.orientation.w =  1.0
    return p



def generate_pose_stamped(dataset: str):
    return eval('pose_' + dataset + '()')



def print_pose_stamped(p: PoseStamped) -> None:
    position = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    orientation = [p.pose.orientation.w, p.pose.orientation.x,
            p.pose.orientation.y, p.pose.orientation.z]
    position_str    = 'Position:              ' + ' '.join([str(x) for x in position])
    orientation_str = 'Orientation (w,x,y,z): ' + ' '.join([str(x) for x in orientation])
    rospy.loginfo(position_str)
    rospy.loginfo(orientation_str)



def main():
    args = parse_args()
    rospy.init_node('habitat_ros_' + program_name())
    pose_pub = rospy.Publisher('/habitat/external_pose', PoseStamped, queue_size=10)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pose = generate_pose_stamped(args.dataset)
        pose_pub.publish(pose)
        rospy.loginfo('Published pose')
        print_pose_stamped(pose)
        rate.sleep()



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

