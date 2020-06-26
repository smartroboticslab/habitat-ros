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

from typing import Dict, Tuple



def print_node_config(config: Dict) -> None:
    """Print a dictionary containing the configuration to the ROS info log"""
    rospy.loginfo('Habitat node parameters:')
    for name, val in config.items():
        rospy.loginfo('  {: <25} {}'.format(name + ':', str(val)))



def read_node_config() -> Dict:
    """Read the node parameters, print them and return a dictionary"""
    # Available parameter names and default values
    param_names = ['rgb_topic_name', 'depth_topic_name', 'semantics_topic_name',
        'habitat_pose_topic_name', 'external_pose_topic_name',
        'external_pose_topic_type', 'width', 'height', 'scene_file',
        'enable_external_pose']
    param_default_values = ['/habitat/rgb', '/habitat/depth',
        '/habitat/semantics', '/habitat/pose', '/habitat/ext_pose',
        'geometry_msgs::PoseStamped', 640, 480, '', False]

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

    # Ensure a valid scene file was supplied
    if not config['scene_file']:
        rospy.logfatal('No scene file supplied')
        raise rospy.ROSException
    elif not os.path.isfile(config['scene_file']):
        rospy.logfatal('Scene file ' + config['scene_file'] + ' does not exist')
        raise rospy.ROSException

    print_node_config(config)
    return config



def rgb_sensor_config(config: Dict, name: str='rgb') -> hs.SensorSpec:
    """Return the configuration for a Habitat color sensor"""
    # Documentation for SensorSpec here
    #   https://aihabitat.org/docs/habitat-sim/habitat_sim.sensor.SensorSpec.html
    rgb_sensor_spec = hs.SensorSpec()
    rgb_sensor_spec.uuid = name
    rgb_sensor_spec.sensor_type = hs.SensorType.COLOR
    rgb_sensor_spec.resolution = [config['height'], config['width']]
    # TODO set the position?
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    #rgb_sensor_spec.position = 1.5 * hs.geo.UP + 0.25 * hs.geo.LEFT
    return rgb_sensor_spec



def depth_sensor_config(config: Dict, name: str='depth') -> hs.SensorSpec:
    """Return the configuration for a Habitat depth sensor"""
    depth_sensor_spec = hs.SensorSpec()
    depth_sensor_spec.uuid = name
    depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
    depth_sensor_spec.resolution = [config['height'], config['width']]
    return depth_sensor_spec



def semantic_sensor_config(config: Dict, name: str='semantics') -> hs.SensorSpec:
    """Return the configuration for a Habitat semantic sensor"""
    semantic_sensor_spec = hs.SensorSpec()
    semantic_sensor_spec.uuid = name
    semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [config['height'], config['width']]
    return semantic_sensor_spec



def init_habitat(config: Dict) -> hs.Simulator:
    """Initialize the Habitat simulator with sensors and scene file"""
    backend_config = hs.SimulatorConfiguration()
    backend_config.scene.id = (config['scene_file'])
    agent_config = hs.AgentConfiguration()
    agent_config.sensor_specifications = [rgb_sensor_config(config),
            depth_sensor_config(config), semantic_sensor_config(config)]
    sim = hs.Simulator(hs.Configuration(backend_config, [agent_config]))
    rospy.loginfo('Habitat simulator initialized')
    return sim



def render(sim: hs.Simulator) -> None:
    for _ in range(100):
        # Just spin in a circle
        observation = sim.step("turn_right")
        # Change from RGBA to RGB and then to BGR
        rgb_render = observation['rgb'][..., 0:3][..., ::-1]
        # Normalize the depth for visualization
        depth_render = np.clip(observation['depth'], 0.0, 10.0)
        depth_render /= 10.0
        depth_render = cv2.cvtColor(depth_render, cv2.COLOR_GRAY2RGB)
        # TODO render the semantics for visualization (create function)
        # Combine into a single render
        render = np.concatenate([rgb_render / 255.0, depth_render], axis=1)
        cv2.imshow("render", render)
        k = cv2.waitKey()
        if k == ord("q"):
            break
        # TODO Return the rgb/depth/semantics



def init_node() -> Tuple[Dict, hs.Simulator]:
    """Initialize the ROS node and Habitat simulator"""
    rospy.init_node('habitat_ros')
    config = read_node_config()
    sim = init_habitat(config)
    return config, sim



def run_publisher_node(config: Dict, sim: hs.Simulator) -> None:
    """Start the ROS publisher node"""
    # TODO setup publishers and subscribers
    # TODO remove render from here
    render(sim)
    rospy.spin()



def main() -> None:
    config, sim = init_node()
    run_publisher_node(config, sim)



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

