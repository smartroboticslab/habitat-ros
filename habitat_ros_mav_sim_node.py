#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import math
import threading

import numpy as np
import quaternion
import rospy
import tf2_ros

from collections import deque
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped
from nav_msgs.msg import Path
from typing import Any, Dict, Tuple, Union

from habitat_ros_node import (Config, read_config, print_config, split_pose,
        combine_pose, msg_to_pose, msg_to_transform, find_tf)



def wrap_angle_2pi(angle_rad: float) -> float:
    """Wrap an angle in radians to the interval [0,2pi)."""
    angle = math.fmod(angle_rad, math.tau)
    if angle < 0:
        angle += math.tau
    return angle

def wrap_angle_pi(angle_rad: float) -> float:
    """Wrap an angle in radians to the interval [-pi,pi]."""
    angle = math.fmod(angle_rad + math.pi, math.tau)
    if angle < 0:
        angle += math.tau
    return angle - math.pi



def angle_diff(start_angle_rad: float, end_angle_rad: float) -> float:
    """Compute the smallest angle in the interval [-pi,pi] that if added to
    start_angle_rad will result in end_angle_rad or an equivalent angle."""
    # Compute the angle difference in the interval (-2pi,2pi)
    start_angle_2pi = wrap_angle_2pi(start_angle_rad)
    end_angle_2pi = wrap_angle_2pi(end_angle_rad)
    diff = end_angle_2pi - start_angle_2pi
    # Select a smaller angle if needed
    if diff > math.pi:
        diff -= math.tau
    elif diff < -math.pi:
        diff += math.tau
    return diff



def yaw_B_to_C_WB(yaw_B: float) -> np.array:
    C_WB = np.identity(3)
    C_WB[0,0] =  math.cos(yaw_B)
    C_WB[0,1] = -math.sin(yaw_B)
    C_WB[1,0] =  math.sin(yaw_B)
    C_WB[1,1] =  math.cos(yaw_B)
    return C_WB

def C_WB_to_yaw_B(C_WB: np.array) -> float:
    pitch_B = math.asin(-C_WB[2,0])
    cos_pitch_B = math.cos(pitch_B)
    return math.atan2(C_WB[1,0] / cos_pitch_B, C_WB[0,0] / cos_pitch_B)



def trajectory_time_x(x0: float, xf: float, a_max:float) -> float:
    """Compute the time required to go from x0 to xf while moving at +-a_max."""
    return 2.0 * math.sqrt(abs(xf - x0) / a_max)



def simulate_x(t: float, x0: float, xf: float, a_max: float) -> float:
    """Return the position at time t for the trajectory starting from x0 and
    ending at xf with constant acceleration +-a_max"""
    tf = trajectory_time_x(x0, xf, a_max)
    if t > tf:
        return xf
    if x0 <= xf:
        if t <= tf / 2.0:
            x = 0.5 * a_max * t**2 + x0
        else:
            x = -0.5 * a_max * t**2 + a_max * tf * t - a_max * tf**2 / 2.0 + xf
    else:
        if t <= tf / 2.0:
            x = -0.5 * a_max * t**2 + x0
        else:
            x = 0.5 * a_max * t**2 - a_max * tf * t + a_max * tf**2 / 2.0 + xf
    return x



def trajectory_time(T_0: np.array, T_f: np.array, a_max: np.array, w_max: np.array) -> float:
    """Return the time required to move from pose T_0 to pose T_f while moving
    at +-a_max and rotating at +-w_max."""
    translation_times = np.fromiter((trajectory_time_x(x0, xf, a) for x0, xf, a
        in zip(T_0[0:3,3], T_f[0:3,3], a_max)), dtype=np.float64)
    yaw_0 = C_WB_to_yaw_B(T_0[0:3,0:3])
    yaw_f = C_WB_to_yaw_B(T_f[0:3,0:3])
    yaw_diff = angle_diff(yaw_0, yaw_f)
    yaw_time = trajectory_time_x(yaw_0, yaw_0 + yaw_diff, w_max[2])
    rotation_times = np.array([0.0, 0.0, yaw_time])
    return float(np.amax(np.maximum(translation_times, rotation_times)))



class SimpleMAVSimNode:
    # Published topic names
    _pose_topic = "/mav_sim/pose"
    # Subscribed topic names
    _goal_path_topic = "/mav_sim/goal_path"
    _init_pose_topic = "/habitat/pose"
    _default_config = {
            "a_max": [1.0, 1.0, 0.5],
            "w_max": [0.1, 0.1, 0.05],
            "sim_freq": 60,
            "world_frame_id": "map"}



    def __init__(self) -> None:
        rospy.init_node("habitat_ros_mav_sim")
        # Read the configuration parameters
        self._config = read_config(self._default_config, "habitat_ros_mav_sim")
        rospy.loginfo("Simple MAV simulator parameters:")
        print_config(self._config)
        # Initialize the transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Initialize data members
        self._pose_mutex = threading.Lock()
        self._start_T_WB_time = rospy.get_time()
        self._goal_T_WBs = deque()
        # Initialize the pose from the habitat node
        T_FB_msg = rospy.wait_for_message(self._init_pose_topic, PoseStamped)
        T_WF = find_tf(self.tf_buffer, T_FB_msg.header.frame_id, self._config['world_frame_id'])
        if T_WF is None:
            rospy.logfatal('Could not find transform from frame '
                    + T_FB_msg.header.frame_id + ' to frame '
                    + self._config['world_frame_id'])
            raise rospy.ROSException
        T_FB = msg_to_pose(T_FB_msg.pose)
        self._T_WB = T_WF.dot(T_FB)
        self._start_T_WB = self._T_WB
        # Setup publishers and subscribers
        self._pub = rospy.Publisher(self._pose_topic, PoseStamped, queue_size=10)
        rospy.Subscriber(self._goal_path_topic, Path, self._path_callback)
        # Main loop
        rospy.loginfo("Simple MAV simulator ready")
        if self._config["sim_freq"] > 0:
            rate = rospy.Rate(self._config["sim_freq"])
        while not rospy.is_shutdown():
            self._simulate()
            self._publish_pose()
            if self._config["sim_freq"] > 0:
                rate.sleep()



    def _path_callback(self, path: Path) -> None:
        # Ignore empty paths
        if not path.poses:
            return
        # Transform the pose to the correct frame
        T_WF = find_tf(self.tf_buffer, path.header.frame_id, self._config['world_frame_id'])
        if T_WF is None:
            rospy.logfatal('Could not find transform from frame '
                    + path.header.frame_id + ' to frame '
                    + self._config['world_frame_id'])
            return
        self._pose_mutex.acquire()
        # Set the current and start poses to the first path vertex
        first_T_FB = msg_to_pose(path.poses[0].pose)
        self._T_WB = T_WF.dot(first_T_FB)
        self._start_T_WB = self._T_WB
        self._start_T_WB_time = rospy.get_time()
        # Clear the queue of any previous paths and add the goal poses
        self._goal_T_WBs.clear()
        for i in range(1, len(path.poses)):
            p = path.poses[i].pose
            self._goal_T_WBs.append(msg_to_pose(p))
        self._pose_mutex.release()



    def _simulate(self) -> None:
        self._pose_mutex.acquire()
        # Exit if there are no goals left
        if not self._goal_T_WBs:
            self._pose_mutex.release()
            return
        # Simulate a single line segment at a time
        T_0 = self._start_T_WB
        T_f = self._goal_T_WBs[0]
        # Simulation parameters
        t0 = self._start_T_WB_time
        tf = t0 + trajectory_time(T_0, T_f, self._config["a_max"],
                self._config["w_max"])
        t = rospy.get_time()
        # Update the position
        self._T_WB[0,3] = simulate_x(t-t0, T_0[0,3], T_f[0,3], self._config["a_max"][0])
        self._T_WB[1,3] = simulate_x(t-t0, T_0[1,3], T_f[1,3], self._config["a_max"][1])
        self._T_WB[2,3] = simulate_x(t-t0, T_0[2,3], T_f[2,3], self._config["a_max"][2])
        # Update the yaw making sure to use the yaw_diff to avoid wrap-around
        # issues
        yaw_0 = C_WB_to_yaw_B(T_0[0:3,0:3])
        yaw_f = C_WB_to_yaw_B(T_f[0:3,0:3])
        yaw_diff = angle_diff(yaw_0, yaw_f)
        yaw = simulate_x(t-t0, yaw_0, yaw_0 + yaw_diff, self._config["w_max"][2])
        self._T_WB[0:3,0:3] = yaw_B_to_C_WB(yaw)
        # Pop the goal pose if it has been reached and update the start pose
        if t >= tf:
            self._goal_T_WBs.popleft()
            self._start_T_WB = self._T_WB
            self._start_T_WB_time = rospy.get_time()
        self._pose_mutex.release()



    def _publish_pose(self) -> None:
        # Extract the position vector and orientation quaternion from the
        # current pose
        self._pose_mutex.acquire()
        position, orientation = split_pose(self._T_WB)
        self._pose_mutex.release()
        # Populate the message fields
        msg = PoseStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = self._config["world_frame_id"]
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = orientation.x
        msg.pose.orientation.y = orientation.y
        msg.pose.orientation.z = orientation.z
        msg.pose.orientation.w = orientation.w
        self._pub.publish(msg)



if __name__ == "__main__":
    try:
        node = SimpleMAVSimNode()
    except rospy.ROSInterruptException:
        pass

