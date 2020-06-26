#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import cv2
import habitat_sim as hs
import numpy as np
import rospkg
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
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



def observation_to_posemsg(obs: hs.sensor.Observation) -> PoseStamped:
    """Convert the agent pose in the observation to a PoseStamped message"""
    p = PoseStamped()
    p.header.frame_id = 'world'
    # Return the current ROS time since the habitat simulator does not provide
    # one
    p.header.stamp = rospy.get_rostime()
    p.pose.position.x = obs['t_WC'][0]
    p.pose.position.y = obs['t_WC'][1]
    p.pose.position.z = obs['t_WC'][2]
    p.pose.orientation.x = obs['q_WC'].x
    p.pose.orientation.y = obs['q_WC'].y
    p.pose.orientation.z = obs['q_WC'].z
    p.pose.orientation.w = obs['q_WC'].w
    return p


def render(sim: hs.Simulator) -> hs.sensor.Observation:
    # Just spin in a circle
    # TODO move in a more meaningful way
    observation = sim.step("turn_right")
    # Change from RGBA to RGB
    observation['rgb'] = observation['rgb'][..., 0:3]
    # TODO process depth
    # TODO process semantics
    # Return the agent ground truth position (t_WC) and orientation (q_WC)
    # TODO convert to z-forward, x-right
    observation['t_WC'] = sim.get_agent(0).get_state().position
    observation['q_WC'] = sim.get_agent(0).get_state().rotation
    # TODO Return the simulation timestamp. sim.get_world_time() returns 0
    return observation



def init_node() -> Tuple[Dict, hs.Simulator]:
    """Initialize the ROS node and Habitat simulator"""
    rospy.init_node('habitat_ros')
    config = read_node_config()
    sim = init_habitat(config)
    return config, sim



def run_publisher_node(config: Dict, sim: hs.Simulator) -> None:
    """Start the ROS publisher node"""
    # Setup the image and pose publishers
    rgb_pub = rospy.Publisher(config['rgb_topic_name'], Image, queue_size=10)
    pose_pub = rospy.Publisher(config['habitat_pose_topic_name'], PoseStamped, queue_size=10)
    # TODO setup depth publisher
    # TODO setup semantics publisher
    # Main publishing loop
    # TODO decouple movement rate from camera framerate, read both from config
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        observation = render(sim)
        rgb_pub.publish(CvBridge().cv2_to_imgmsg(observation['rgb'], "rgb8"))
        pose_pub.publish(observation_to_posemsg(observation))
        # TODO publish depth
        # TODO publish semantics
        # TODO publish rgb/depth/semantics visualisations. Use $TOPICNAME_render
        rate.sleep()



def main() -> None:
    config, sim = init_node()
    # TODO select whether to run publisher or service
    run_publisher_node(config, sim)



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

