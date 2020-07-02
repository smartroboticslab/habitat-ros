#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import os

import cv2
import habitat_sim as hs
import numpy as np
import quaternion
import rospkg
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from typing import Any, Dict, List, Tuple



# Custom type definitions
Config = Dict[str, Any]



# Convert Matterport3D class ID to class name
class_id_to_name = {
    0: 'void',
    1: 'wall',
    2: 'floor',
    3: 'chair',
    4: 'door',
    5: 'table',
    6: 'picture',
    7: 'cabinet',
    8: 'cushion',
    9: 'window',
    10: 'sofa',
    11: 'bed',
    12: 'curtain',
    13: 'chest_of_drawers',
    14: 'plant',
    15: 'sink',
    16: 'stairs',
    17: 'ceiling',
    18: 'toilet',
    19: 'stool',
    20: 'towel',
    21: 'mirror',
    22: 'tv_monitor',
    23: 'shower',
    24: 'column',
    25: 'bathtub',
    26: 'counter',
    27: 'fireplace',
    28: 'lighting',
    29: 'beam',
    30: 'railing',
    31: 'shelving',
    32: 'blinds',
    33: 'gym_equipment',
    34: 'seating',
    35: 'board_panel',
    36: 'furniture',
    37: 'appliances',
    38: 'clothes',
    39: 'objects',
    40: 'misc',
    41: 'unlabeled',
}

# Matterport3D class RGB colors
class_colors = np.array([
    [0xff, 0xff, 0xff],
    [0xae, 0xc7, 0xe8],
    [0x70, 0x80, 0x90],
    [0x98, 0xdf, 0x8a],
    [0xc5, 0xb0, 0xd5],
    [0xff, 0x7f, 0x0e],
    [0xd6, 0x27, 0x28],
    [0x1f, 0x77, 0xb4],
    [0xbc, 0xbd, 0x22],
    [0xff, 0x98, 0x96],
    [0x2c, 0xa0, 0x2c],
    [0xe3, 0x77, 0xc2],
    [0xde, 0x9e, 0xd6],
    [0x94, 0x67, 0xbd],
    [0x8c, 0xa2, 0x52],
    [0x84, 0x3c, 0x39],
    [0x9e, 0xda, 0xe5],
    [0x9c, 0x9e, 0xde],
    [0xe7, 0x96, 0x9c],
    [0x63, 0x79, 0x39],
    [0x8c, 0x56, 0x4b],
    [0xdb, 0xdb, 0x8d],
    [0xd6, 0x61, 0x6b],
    [0xce, 0xdb, 0x9c],
    [0xe7, 0xba, 0x52],
    [0x39, 0x3b, 0x79],
    [0xa5, 0x51, 0x94],
    [0xad, 0x49, 0x4a],
    [0xb5, 0xcf, 0x6b],
    [0x52, 0x54, 0xa3],
    [0xbd, 0x9e, 0x39],
    [0xc4, 0x9c, 0x94],
    [0xf7, 0xb6, 0xd2],
    [0x6b, 0x6e, 0xcf],
    [0xff, 0xbb, 0x78],
    [0xc7, 0xc7, 0xc7],
    [0x8c, 0x6d, 0x31],
    [0xe7, 0xcb, 0x94],
    [0xce, 0x6d, 0xbd],
    [0x17, 0xbe, 0xcf],
    [0x7f, 0x7f, 0x7f],
    [0x00, 0x00, 0x00]
])

# Instantiate a single CvBridge object for all conversions
_bridge = CvBridge()
# Published topic names
_rgb_topic_name = '/habitat/rgb'
_depth_topic_name = '/habitat/depth'
_semantic_class_topic_name = '/habitat/semantic_class'
_semantic_instance_topic_name = '/habitat/semantic_instance'
_habitat_pose_topic_name = '/habitat/pose'



def print_node_config(config: Config) -> None:
    """Print a dictionary containing the configuration to the ROS info log"""
    rospy.loginfo('Habitat node parameters:')
    for name, val in config.items():
        rospy.loginfo('  {: <25} {}'.format(name + ':', str(val)))



def read_node_config() -> Config:
    """Read the node parameters, print them and return a dictionary"""
    # Available parameter names and default values
    param_names = ['external_pose_topic_name', 'external_pose_topic_type',
            'width', 'height', 'scene_file', 'publisher_rate',
            'enable_external_pose', 'enable_semantics', 'visualize_semantics']
    param_default_values = ['/habitat/ext_pose', 'geometry_msgs::PoseStamped',
            640, 480, '', 0, False, False, False]

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



def rgb_sensor_config(config: Config, name: str='rgb') -> hs.SensorSpec:
    """Return the configuration for a Habitat color sensor"""
    rgb_sensor_spec = hs.SensorSpec()
    rgb_sensor_spec.uuid = name
    rgb_sensor_spec.sensor_type = hs.SensorType.COLOR
    rgb_sensor_spec.resolution = [config['height'], config['width']]
    return rgb_sensor_spec



def depth_sensor_config(config: Config, name: str='depth') -> hs.SensorSpec:
    """Return the configuration for a Habitat depth sensor"""
    depth_sensor_spec = hs.SensorSpec()
    depth_sensor_spec.uuid = name
    depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
    depth_sensor_spec.resolution = [config['height'], config['width']]
    return depth_sensor_spec



def semantic_sensor_config(config: Config, name: str='semantics') -> hs.SensorSpec:
    """Return the configuration for a Habitat semantic sensor"""
    semantic_sensor_spec = hs.SensorSpec()
    semantic_sensor_spec.uuid = name
    semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [config['height'], config['width']]
    return semantic_sensor_spec



def init_habitat(config: Config) -> hs.Simulator:
    """Initialize the Habitat simulator with sensors and scene file"""
    backend_config = hs.SimulatorConfiguration()
    backend_config.scene.id = (config['scene_file'])
    agent_config = hs.AgentConfiguration()
    agent_config.sensor_specifications = [rgb_sensor_config(config),
            depth_sensor_config(config), semantic_sensor_config(config)]
    sim = hs.Simulator(hs.Configuration(backend_config, [agent_config]))
    # Get the intrinsic camera parameters
    hfov = float(agent_config.sensor_specifications[0].parameters['hfov'])
    fx = 1.0 / np.tan(hfov / 2.0)
    config['K'] = np.array([[fx, 0.0, 0.0, 0.0], [0.0, fx, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    rospy.loginfo('Habitat simulator initialized')
    return sim



def pose_to_msg(T_WB: np.ndarray) -> PoseStamped:
    """Convert the agent pose in the observation to a PoseStamped message"""
    position = T_WB[0:3, 3]
    orientation = quaternion.from_rotation_matrix(T_WB[0:3, 0:3])
    p = PoseStamped()
    p.header.frame_id = 'map'
    # Return the current ROS time since the habitat simulator does not provide
    # one. sim.get_world_time() always returns 0
    p.header.stamp = rospy.get_rostime()
    p.pose.position.x = position[0]
    p.pose.position.y = position[1]
    p.pose.position.z = position[2]
    p.pose.orientation.x = orientation.x
    p.pose.orientation.y = orientation.y
    p.pose.orientation.z = orientation.z
    p.pose.orientation.w = orientation.w
    return p



def rgb_to_msg(rgb: np.ndarray) -> Image:
    """Convert RGB image to ROS Image message"""
    return _bridge.cv2_to_imgmsg(rgb, "rgb8")



def depth_to_msg(depth: np.ndarray) -> Image:
    """Convert depth image to ROS Image message"""
    return _bridge.cv2_to_imgmsg(depth, "32FC1")



def sem_instances_to_msg(sem_instances: np.ndarray) -> Image:
    """Convert instance ID image to ROS Image message"""
    return _bridge.cv2_to_imgmsg(sem_instances.astype(np.uint16), "16UC1")



def sem_classes_to_msg(sem_classes: np.ndarray) -> Image:
    """Convert class ID image to ROS Image message"""
    return _bridge.cv2_to_imgmsg(sem_classes.astype(np.uint8), "8UC1")



def render_sem_instances_to_msg(sem_instances: np.ndarray) -> Image:
    """Render an instance ID image to a ROS Image message with pretty colours"""
    color_img_shape = [sem_instances.shape[0], sem_instances.shape[1] , 3]
    color_img = np.zeros(color_img_shape, dtype=np.uint8)
    # TODO make this conversion more efficient
    for y in range(color_img.shape[1]):
        for x in range(color_img.shape[0]):
            color_img[x, y, :] = class_colors[sem_instances[x, y] % 41, :]
    return _bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")



def render_sem_classes_to_msg(sem_classes: np.ndarray) -> Image:
    """Render a class ID image to a ROS Image message with pretty colours"""
    color_img_shape = [sem_classes.shape[0], sem_classes.shape[1] , 3]
    color_img = np.zeros(color_img_shape, dtype=np.uint8)
    # TODO make this conversion more efficient
    for y in range(color_img.shape[1]):
        for x in range(color_img.shape[0]):
            color_img[x, y, :] = class_colors[sem_classes[x, y], :]
    return _bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")



def render(sim: hs.Simulator, config: Config) -> hs.sensor.Observation:
    """Move the camera and return the sensor observations and ground truth pose"""
    # TODO move in a more meaningful way
    if '1LXtFkjw3qL' in config['scene_file']:
        # Show a scene for debugging instances/classes
        agent = sim.get_agent(0)
        agent_state = hs.agent.AgentState([0.0239539,-2.91559,6.6636], [0, -0.77301, 0, -0.634393])
        agent.set_state(agent_state)
        observation = sim.get_sensor_observations()
    else:
        # Move around in circles
        sim.step("turn_right")
        sim.step("move_forward")
        observation = sim.step("move_forward")

    # Change from RGBA to RGB
    observation['rgb'] = observation['rgb'][..., 0:3]

    if config['enable_semantics'] and config['instance_to_class'].size > 0:
        # Assuming the scene has no more than 65534 objects
        observation['sem_instances'] = np.clip(observation['semantics'].astype(np.uint16), 0, 65535)
        del observation['semantics']
        # Convert instance IDs to class IDs
        observation['sem_classes'] = np.zeros(observation['sem_instances'].shape, dtype=np.uint8)
        # TODO make this conversion more efficient
        for y in range(observation['sem_classes'].shape[1]):
            for x in range(observation['sem_classes'].shape[0]):
                observation['sem_classes'][x, y] = config['instance_to_class'][observation['sem_instances'][x, y]]
        observation['sem_classes'] = observation['sem_classes'].astype(np.uint8)

    # Get the camera ground truth pose (T_HC) in the habitat frame from the
    # position and orientation
    t_HC = sim.get_agent(0).get_state().position
    q_HC = sim.get_agent(0).get_state().rotation
    T_HC = np.identity(4)
    T_HC[0:3, 0:3] = quaternion.as_rotation_matrix(q_HC)
    T_HC[0:3, 3] = t_HC
    # Change from the habitat frame (y-up) to the world frame (z-up)
    T_WH = np.identity(4)
    T_WH[0:3, 0:3] = quaternion.as_rotation_matrix(hs.utils.common.quat_from_two_vectors(
            hs.geo.GRAVITY, np.array([0.0, 0.0, -1.0])))
    T_WC = np.dot(T_WH, T_HC)
    # Change from the camera frame (-z-forward, y-up) to the ROS body frame
    # (x-forward, z-up)
    T_CB = np.array([(0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)])
    T_WB = np.dot(T_WC, T_CB)
    observation['T_WB'] = T_WB
    return observation



def generate_instance_to_class_map(objects: List[hs.scene.SemanticObject]) -> np.ndarray:
    """Given the objects in the scene, create an array that maps instance IDs to
    class IDs"""
    map = np.zeros(len(objects), dtype=np.uint8)
    for instance_id in range(len(objects)):
        map[instance_id] = objects[instance_id].category.index()
        if map[instance_id] > 40:
            rospy.logwarn(''.join(['Invalid object class ID/name: ',
                str(map[instance_id]), '/"',
                objects[instance_id].category.name(), '"\nReplacing with 0']))
            map[instance_id] = 0
    return map



def init_node() -> Tuple[Dict, hs.Simulator]:
    """Initialize the ROS node and Habitat simulator"""
    rospy.init_node('habitat_ros')
    config = read_node_config()
    sim = init_habitat(config)
    return config, sim



def run_publisher_node(config: Config, sim: hs.Simulator) -> None:
    """Start the ROS publisher node"""
    # Setup the instance/class conversion map
    config['instance_to_class'] = generate_instance_to_class_map(sim.semantic_scene.objects)
    if config['enable_semantics'] and config['instance_to_class'].size == 0:
        rospy.logwarn('The scene contains no semantics')

    # Setup the image and pose publishers
    pose_pub = rospy.Publisher(_habitat_pose_topic_name, PoseStamped, queue_size=10)
    rgb_pub = rospy.Publisher(_rgb_topic_name, Image, queue_size=10)
    depth_pub = rospy.Publisher(_depth_topic_name, Image, queue_size=10)
    if config['enable_semantics'] and config['instance_to_class'].size > 0:
        # Only publish semantics if the scene contains semantics
        sem_class_pub = rospy.Publisher(_semantic_class_topic_name, Image, queue_size=10)
        sem_instance_pub = rospy.Publisher(_semantic_instance_topic_name, Image, queue_size=10)
        if config['visualize_semantics']:
            sem_class_render_pub = rospy.Publisher(_semantic_class_topic_name + '_render', Image, queue_size=10)
            sem_instance_render_pub = rospy.Publisher(_semantic_instance_topic_name + '_render', Image, queue_size=10)

    # Main publishing loop
    # TODO decouple movement rate from camera framerate, read both from config
    if config['publisher_rate'] > 0:
        rate = rospy.Rate(config['publisher_rate'])
    while not rospy.is_shutdown():
        observation = render(sim, config)
        pose_pub.publish(pose_to_msg(observation['T_WB']))
        rgb_pub.publish(rgb_to_msg(observation['rgb']))
        depth_pub.publish(depth_to_msg(observation['depth']))
        if config['enable_semantics'] and config['instance_to_class'].size > 0:
            sem_class_pub.publish(sem_classes_to_msg(observation['sem_classes']))
            sem_instance_pub.publish(sem_instances_to_msg(observation['sem_instances']))
            # Publish semantics visualisations
            if config['visualize_semantics']:
                sem_class_render_pub.publish(render_sem_classes_to_msg(observation['sem_classes']))
                sem_instance_render_pub.publish(render_sem_instances_to_msg(observation['sem_instances']))
        if config['publisher_rate'] > 0:
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

