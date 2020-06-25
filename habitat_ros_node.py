#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

# TODO function given a pose render the rgb depth and sem
# TODO function to generate a valid random pose close the the given one

import os
import sys

import cv2
import habitat_sim as hs
import numpy as np
import rospkg
import rospy

from typing import Dict



def print_node_config(config: Dict):
    rospy.loginfo('Habitat node parameters:')
    for name, val in config.items():
        rospy.loginfo('  {: <25} {}'.format(name + ':', str(val)))



def read_node_config() -> Dict:
    # Available parameter names and default values
    param_names = ['rgb_topic_name', 'depth_topic_name', 'semantics_topic_name',
        'habitat_pose_topic_name', 'external_pose_topic_name',
        'external_pose_topic_type', 'scene_file', 'enable_external_pose']
    param_default_values = ['/habitat/rgb', '/habitat/depth',
        '/habitat/semantics', '/habitat/pose', '/habitat/ext_pose',
        'geometry_msgs::PoseStamped', '', False]
    # Read the parameters
    config = {}
    for name, val in zip(param_names, param_default_values):
        config[name] = rospy.get_param('~' + name, val)
    # Get an absolute path from the supplied scene file
    config['scene_file'] = os.path.expanduser(config['scene_file'])
    if not os.path.isabs(config['scene_file']):
        # The scene file path is relative, assuming relative to the ROS package
        package_path = rospkg.RosPack().get_path('habitat_ros') + '/'
        config['scene_file'] = package_path + config['scene_file']
    print_node_config(config)
    # Ensure a valid scene file was supplied
    if not config['scene_file']:
        rospy.logfatal('No scene file supplied')
        raise rospy.ROSException
    elif not os.path.isfile(config['scene_file']):
        rospy.logfatal('Scene file ' + config['scene_file'] + ' does not exist')
        raise rospy.ROSException
    return config



def color_sensor_config(config: Dict, sensor_name: str='color_sensor'):
    # Documentation for SensorSpec here
    #   https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.SensorSpec.html
    color_sensor_spec = hs.SensorSpec()
    color_sensor_spec.uuid = sensor_name
    color_sensor_spec.sensor_type = hs.SensorType.COLOR
    color_sensor_spec.resolution = [480, 640]
    # TODO set the position?
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    #color_sensor_spec.position = 1.5 * hs.geo.UP + 0.25 * hs.geo.LEFT
    return color_sensor_spec



def depth_sensor_config(config: Dict, sensor_name: str='depth_sensor'):
    depth_sensor_spec = hs.SensorSpec()
    depth_sensor_spec.uuid = sensor_name
    depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
    depth_sensor_spec.resolution = [480, 640]
    return depth_sensor_spec



def semantic_sensor_config(config: Dict, sensor_name: str='semantic_sensor'):
    semantic_sensor_spec = hs.SensorSpec()
    semantic_sensor_spec.uuid = sensor_name
    semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [480, 640]
    return semantic_sensor_spec



def init_habitat(config: Dict):
    backend_config = hs.SimulatorConfiguration()
    backend_config.scene.id = (config['scene_file'])
    agent_config = hs.AgentConfiguration()
    agent_config.sensor_specifications = [color_sensor_config(config),
            depth_sensor_config(config), semantic_sensor_config(config)]
    sim = hs.Simulator(hs.Configuration(backend_config, [agent_config]))
    rospy.loginfo('Habitat initialized')
    return sim



def render(sim):
    for _ in range(100):
        # Just spin in a circle
        observation = sim.step("turn_right")
        # Change from RGBA to RGB and then to BGR
        color_render = observation['color_sensor'][..., 0:3][..., ::-1]
        # Normalize the depth for visualization
        depth_render = np.clip(observation['depth_sensor'], 0.0, 10.0)
        depth_render /= 10.0
        depth_render = cv2.cvtColor(depth_render, cv2.COLOR_GRAY2RGB)
        # TODO render the semantics for visualization (create function)
        # Combine into a single render
        render = np.concatenate([color_render / 255.0, depth_render], axis=1)
        cv2.imshow("render", render)
        k = cv2.waitKey()
        if k == ord("q"):
            break
        # TODO Return the color/depth/semantics



def run_node():
    # Read the node configuration
    rospy.init_node('habitat_ros', anonymous=True)
    config = read_node_config()
    # Initialize the habitat simulator
    sim = init_habitat(config)
    # TODO setup publishers and subscribers
    # TODO remove render from here
    render(sim)
    rospy.spin()



if __name__ == "__main__":
    try:
        run_node()
    except rospy.ROSInterruptException:
        pass

