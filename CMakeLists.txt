# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required(VERSION 3.1)

project(habitat_ros)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    rospy
    geometry_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS
        rospy
        message_runtime
        geometry_msgs
        sensor_msgs
        std_msgs
)

catkin_install_python(
    PROGRAMS habitat_ros_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

