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



def str_to_center(s: str) -> List[float]:
    tokens = s.split(',')
    if len(tokens) != 3:
        return None
    center = []
    for coord in tokens:
        if coord.lstrip('-').replace('.', '').isdigit():
            center.append(float(coord))
        else:
            return None
    return center



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Publish poses along a '
            'horizontal circle')
    parser.add_argument('-r', '--radius', type=float, default=1,
            help='The radius of the circle in meters (default: 1)')
    parser.add_argument('-c', '--center', default='0,0,1',
            help='The coordinates of the circle\'s center in meters '
            '(default: 0,0,1)')
    args = parser.parse_args()
    # Convert string to circle center coordinates
    args.center = str_to_center(args.center)
    if args.center is None:
        printerr('The circle center must be in the format X,Y,Z')
        sys.exit(2)
    return args



def generate_pose_stamped(center: List[float], radius: float, t: float):
    f = 0.1
    theta = 2 * pi * f * t
    R = np.array([(cos(theta + pi/2), -sin(theta + pi/2), 0.0),
            (sin(theta + pi/2), cos(theta + pi/2), 0.0), (0.0, 0.0, 1.0)])
    q = quaternion.from_rotation_matrix(R)
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = rospy.get_rostime()
    p.pose.position.x = center[0] + radius * cos(theta)
    p.pose.position.y = center[1] + radius * sin(theta)
    p.pose.position.z = center[2]
    p.pose.orientation.x = q.x
    p.pose.orientation.y = q.y
    p.pose.orientation.z = q.z
    p.pose.orientation.w = q.w
    return p



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
        t = rospy.get_rostime().to_sec()
        pose = generate_pose_stamped(args.center, args.radius, t)
        pose_pub.publish(pose)
        rospy.loginfo('Published pose')
        print_pose_stamped(pose)
        rate.sleep()



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

