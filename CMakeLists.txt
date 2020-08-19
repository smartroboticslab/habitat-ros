# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required(VERSION 3.1)

project(habitat_ros)

# Set the build type and warn if other than Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE set to ${CMAKE_BUILD_TYPE}. Performance may be terrible.")
else()
  message(STATUS "Building with build type '${CMAKE_BUILD_TYPE}'")
endif()

# Compile habitat-sim
# This assumes Ubuntu 20.04 and a conda environment already set-up with the
# habitat-sim dependencies and activated
add_custom_target(habitat-sim ALL
    python3 setup.py install --prefix ~/.local --with-cuda --cmake-args="-DUSE_SYSTEM_ASSIMP=ON"
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/habitat_sim
)

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

