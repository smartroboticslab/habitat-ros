#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import threading

import cv2
import habitat_sim as hs
import numpy as np
import quaternion
import rospkg
import rospy
import tf2_ros

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from typing import Any, Dict, List, Tuple, Union



# Custom type definitions
Config = Dict[str, Any]
Observation = hs.sensor.Observation
Publishers = Dict[str, rospy.Publisher]
Sim = hs.Simulator



def read_config(config: Config) -> Config:
    """Read the ROS parameters from the namespace into the configuration
    dictionary and return the result. Parameters that don't exist in the ROS
    namespace retain their initial values."""
    new_config = config.copy()
    for name, val in config.items():
        new_config[name] = rospy.get_param("~habitat/" + name, val)
    return new_config

def print_config(config: Config) -> None:
    """Print a dictionary containing the configuration to the ROS info log."""
    for name, val in config.items():
        rospy.loginfo("  {: <25} {}".format(name + ":", str(val)))



def split_pose(T: np.array) -> Tuple[np.array, quaternion.quaternion]:
    """Split a pose in a 4x4 matrix into a position vector and an orientation
    quaternion."""
    return T[0:3, 3], quaternion.from_rotation_matrix(T[0:3, 0:3])

def combine_pose(t: np.array, q: quaternion.quaternion) -> np.array:
    """Combine a position vector and an orientation quaternion into a 4x4 pose
    matrix."""
    T = np.identity(4)
    T[0:3, 3] = t
    T[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    return T

def msg_to_pose(msg: Pose) -> np.array:
    """Convert a ROS Pose message to a 4x4 pose matrix."""
    t = [msg.position.x, msg.position.y, msg.position.z]
    q = quaternion.quaternion(msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z)
    return combine_pose(t, q)

def msg_to_transform(msg: Transform) -> np.array:
    """Convert a ROS Transform message to a 4x4 transform matrix."""
    t = [msg.translation.x, msg.translation.y, msg.translation.z]
    q = quaternion.quaternion(msg.rotation.w, msg.rotation.x,
            msg.rotation.y, msg.rotation.z)
    return combine_pose(t, q)

def list_to_pose(l: List) -> Union[np.array, None]:
    """Convert a list to a pose represented by a 4x4 homogeneous matrix. The
    list may have a varying number of elements:
    - 3 (translation: x, y, z)
    - 4 (orientation quaternion: qx, qy, qz, qw)
    - 7 (translation, orientation quaternion)
    - 16 (4x4 homogeneous matrix in row-major from)"""
    n = len(l)
    if n == 3:
        # Position: tx, ty, tz
        T = np.identity(4)
        T[0:3,3] = np.array(l).T
    elif n == 4:
        # Orientation quaternion: qx, qy, qz, qw
        q = quaternion.quaternion(l[3], l[0], l[1], l[2])
        T = np.identity(4)
        T[0:3,0:3] = quaternion.as_rotation_matrix(q)
    elif n == 7:
        # Position and orientation quaternion: tx, ty, tz, qx, qy, qz, qw
        q = quaternion.quaternion(l[6], l[3], l[4], l[5])
        T = np.identity(4)
        T[0:3,3] = np.array(l[0:3]).T
        T[0:3,0:3] = quaternion.as_rotation_matrix(q)
    elif n == 16:
        # 4x4 pose matrix in row-major order
        T = np.array(l)
        T = T.reshape((4, 4))
        rospy.logwarn(T)
    else:
        T = None
    return T



def hfov_to_fx(hfov: float, width: int) -> float:
    """Convert horizontal field of view in degrees to focal length in pixels.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return 1.0 / (2.0 / float(width) * math.tan(math.radians(hfov) / 2.0))

def fx_to_hfov(fx: float, width: int) -> float:
    """Convert focal length in pixels to horizontal field of view in degrees.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * fx)))



def find_tf(tf_buffer: tf2_ros.Buffer, from_frame: str, to_frame: str) -> Union[np.array, None]:
    """Return the transformation relating the 2 frames."""
    try:
        return msg_to_transform(tf_buffer.lookup_transform(from_frame, to_frame, rospy.Time()).transform)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return None



class HabitatROSNode:
    # Dictionary from Matterport3D class IDs to class names
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
    _rgb_topic_name = '~rgb/'
    _depth_topic_name = '~depth/'
    _sem_class_topic_name = '~semantic_class/'
    _sem_instance_topic_name = '~semantic_instance/'
    _habitat_pose_topic_name = '~pose'

    # Subscribed topic names
    _external_pose_topic_name = '~external_pose'

    # Transforms between the habitat frame H (y-up) and the world frame W (z-up)
    _T_WH = np.identity(4)
    _T_WH[0:3, 0:3] = quaternion.as_rotation_matrix(hs.utils.common.quat_from_two_vectors(
            hs.geo.GRAVITY, np.array([0.0, 0.0, -1.0])))
    _T_HW = np.linalg.inv(_T_WH)

    # Transforms between the habitat camera frame C (-z-forward, y-up) and the
    # ROS body frame B (x-forward, z-up)
    _T_CB = np.array([(0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)])
    _T_BC = np.linalg.inv(_T_CB)

    # The default node options
    _default_config = {
            'width': 640,
            'height': 480,
            'near_plane': 0.1,
            'far_plane': 10.0,
            'fx': 525.0,
            'fps': 30,
            'enable_semantics': False,
            'allowed_classes': [],
            'scene_file': '',
            'initial_T_WB': [],
            'world_frame_id': 'map',
            'visualize_semantics': False}



    def __init__(self):
        # Initialize the node, habitat-sim and publishers
        rospy.init_node('habitat')
        self.config = self._read_node_config()
        self.sim = self._init_habitat(self.config)
        self.pub = self._init_publishers(self.config)
        # Initialize the pose mutex
        self.T_WB_mutex = threading.Lock()
        # Initialize the transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Setup the external pose subscriber
        rospy.Subscriber(self._external_pose_topic_name, PoseStamped,
                self._pose_callback, queue_size=1)
        rospy.loginfo('Habitat node ready')
        # Main loop
        if self.config['fps'] > 0:
            rate = rospy.Rate(self.config['fps'])
        while not rospy.is_shutdown():
            # Move, observe and publish
            self._teleport()
            observation = self._render(self.sim, self.config)
            self._publish_observation(observation, self.pub, self.config)
            if self.config['fps'] > 0:
                rate.sleep()



    def _read_node_config(self) -> Config:
        """Read the node parameters, print them and return a dictionary."""
        # Read the parameters
        config = read_config(self._default_config)
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
        # Create the initial T_WB matrix
        T = list_to_pose(config['initial_T_WB'])
        if T is None and config['initial_T_WB']:
            rospy.logerr('Invalid initial T_WB. Expected list of 3, 4, 7 or 16 elements')
        config['initial_T_WB'] = T
        rospy.loginfo('Habitat node parameters:')
        print_config(config)
        return config



    def _init_habitat(self, config: Config) -> Sim:
        """Initialize the Habitat simulator, create the sensors and load the
        scene file."""
        backend_config = hs.SimulatorConfiguration()
        backend_config.scene.id = (config['scene_file'])
        agent_config = hs.AgentConfiguration()
        agent_config.sensor_specifications = [self._rgb_sensor_config(config),
                self._depth_sensor_config(config), self._semantic_sensor_config(config)]
        sim = Sim(hs.Configuration(backend_config, [agent_config]))
        # Get the intrinsic camera parameters
        hfov = float(agent_config.sensor_specifications[0].parameters['hfov'])
        fx = hfov_to_fx(hfov, config['width'])
        cx = config['width'] / 2.0 - 0.5
        cy = config['height'] / 2.0 - 0.5
        config['K'] = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64)
        config['P'] = np.array([[fx, 0.0, cx, 0.0], [0.0, fx, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64)
        # Setup the instance/class conversion map
        config['instance_to_class'] = self._generate_instance_to_class_map(sim.semantic_scene.objects)
        if config['enable_semantics'] and config['instance_to_class'].size == 0:
            rospy.logwarn('The scene contains no semantics')
        # Get or set the initial agent pose
        agent = sim.get_agent(0)
        if config['initial_T_WB'] is None:
            t_HC = agent.get_state().position
            q_HC = agent.get_state().rotation
            T_HC = combine_pose(t_HC, q_HC)
            self.T_WB = self._T_HC_to_T_WB(T_HC)
        else:
            self.T_WB = config['initial_T_WB']
            t_HC, q_HC = split_pose(self._T_WB_to_T_HC(self.T_WB))
            agent_state = hs.agent.AgentState(t_HC, q_HC)
            agent.set_state(agent_state)
        t_WB, q_WB = split_pose(self.T_WB)
        rospy.loginfo('Habitat initial t_WB:           ' + str(t_WB))
        rospy.loginfo('Habitat initial q_WB (w,x,y,z): ' + str(q_WB))
        return sim



    def _rgb_sensor_config(self, config: Config) -> hs.SensorSpec:
        """Return the configuration for a Habitat color sensor."""
        rgb_sensor_spec = hs.SensorSpec()
        rgb_sensor_spec.uuid = 'rgb'
        rgb_sensor_spec.sensor_type = hs.SensorType.COLOR
        rgb_sensor_spec.resolution = [config['height'], config['width']]
        rgb_sensor_spec.parameters['near'] = str(0.00001)
        rgb_sensor_spec.parameters['far'] = str(1000)
        rgb_sensor_spec.parameters['hfov'] = str(fx_to_hfov(config['fx'], config['width']))
        return rgb_sensor_spec



    def _depth_sensor_config(self, config: Config) -> hs.SensorSpec:
        """Return the configuration for a Habitat depth sensor."""
        depth_sensor_spec = hs.SensorSpec()
        depth_sensor_spec.uuid = 'depth'
        depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
        depth_sensor_spec.resolution = [config['height'], config['width']]
        depth_sensor_spec.parameters['near'] = str(config['near_plane'])
        depth_sensor_spec.parameters['far'] = str(config['far_plane'])
        depth_sensor_spec.parameters['hfov'] = str(fx_to_hfov(config['fx'], config['width']))
        return depth_sensor_spec



    def _semantic_sensor_config(self, config: Config) -> hs.SensorSpec:
        """Return the configuration for a Habitat semantic sensor."""
        semantic_sensor_spec = hs.SensorSpec()
        semantic_sensor_spec.uuid = 'semantic'
        semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [config['height'], config['width']]
        semantic_sensor_spec.parameters['near'] = str(0.00001)
        semantic_sensor_spec.parameters['far'] = str(1000)
        semantic_sensor_spec.parameters['hfov'] = str(fx_to_hfov(config['fx'], config['width']))
        return semantic_sensor_spec



    def _generate_instance_to_class_map(self, objects: List[hs.scene.SemanticObject]) -> np.ndarray:
        """Given the objects in the scene, create an array that maps instance
        IDs to class IDs."""
        map = np.zeros(len(objects), dtype=np.uint8)
        for instance_id in range(len(objects)):
            map[instance_id] = objects[instance_id].category.index()
            if map[instance_id] > 40:
                rospy.logwarn(''.join(['Invalid object class ID/name ',
                    str(map[instance_id]), '/"',
                    objects[instance_id].category.name(), '", replacing with 0']))
                map[instance_id] = 0
        return map



    def _init_publishers(self, config: Config) -> Publishers:
        """Initialize and return the image and pose publishers."""
        image_queue_size = 10
        pub = {}
        # Pose publisher
        pub['pose'] = rospy.Publisher(self._habitat_pose_topic_name, PoseStamped, queue_size=10)
        # Image publishers
        pub['rgb'] = rospy.Publisher(self._rgb_topic_name + 'image_raw',
                Image, queue_size=image_queue_size)
        pub['depth'] = rospy.Publisher(self._depth_topic_name + 'image_raw',
                Image, queue_size=image_queue_size)
        if config['enable_semantics'] and config['instance_to_class'].size > 0:
            # Only publish semantics if the scene contains semantics
            pub['sem_class'] = rospy.Publisher(self._sem_class_topic_name + 'image_raw',
                    Image, queue_size=image_queue_size)
            pub['sem_instance'] = rospy.Publisher(self._sem_instance_topic_name + 'image_raw',
                    Image, queue_size=image_queue_size)
            if config['visualize_semantics']:
                pub['sem_class_render'] = rospy.Publisher(self._sem_class_topic_name + 'image_color',
                        Image, queue_size=image_queue_size)
                pub['sem_instance_render'] = rospy.Publisher(self._sem_instance_topic_name + 'image_color',
                        Image, queue_size=image_queue_size)
        # Publish the camera info for each image topic
        image_topics = [self._rgb_topic_name, self._depth_topic_name]
        if config['enable_semantics'] and config['instance_to_class'].size > 0:
            image_topics += [self._sem_class_topic_name, self._sem_instance_topic_name]
        for topic in image_topics:
            pub[topic + '_camera_info'] = rospy.Publisher(topic + 'camera_info',
                CameraInfo, queue_size=1, latch=True)
            pub[topic + '_camera_info'].publish(self._camera_intrinsics_to_msg(config))
        return pub



    def _pose_callback(self, pose: PoseStamped) -> None:
        """Callback for receiving external pose messages. It updates the agent
        pose."""
        # Find the required transform from some frame F to the world frame W as
        # specified in self.config['world_frame_id']
        T_WF = find_tf(self.tf_buffer, pose.header.frame_id, self.config['world_frame_id'])
        if T_WF is None:
            rospy.logerr_once('Could not find transform from frame '
                    + pose.header.frame_id + ' to frame '
                    + self.config['world_frame_id'])
            return
        # Transform the pose
        T_FB = msg_to_pose(pose.pose)
        T_WB = T_WF @ T_FB
        # Update the pose
        self.T_WB_mutex.acquire()
        self.T_WB = T_WB
        self.T_WB_mutex.release()



    def _filter_sem_classes(self, observation: Observation) -> None:
        """Remove object detections whose classes are not in the allowed class
        list. Their class and instance IDs are set to 0."""
        # Generate a per-pixel boolean matrix
        allowed = np.vectorize(lambda x: x in self.config['allowed_classes'])
        allowed_pixels = allowed(observation['sem_classes'])
        # Set all False pixels to 0 on the class and instance images
        class_zeros = np.zeros(observation['sem_classes'].shape, dtype=observation['sem_classes'].dtype)
        instance_zeros = np.zeros(observation['sem_instances'].shape, dtype=observation['sem_instances'].dtype)
        observation['sem_classes'] = np.where(allowed_pixels, observation['sem_classes'], class_zeros)
        observation['sem_instances'] = np.where(allowed_pixels, observation['sem_instances'], instance_zeros)



    def _pose_to_msg(self, observation: Observation) -> PoseStamped:
        """Convert the agent pose from the observation to a ROS PoseStamped
        message."""
        position = observation['T_WB'][0:3, 3]
        orientation = quaternion.from_rotation_matrix(observation['T_WB'][0:3, 0:3])
        p = PoseStamped()
        p.header.frame_id = self.config['world_frame_id']
        # Return the current ROS time since the habitat simulator does not provide
        # one. sim.get_world_time() always returns 0
        p.header.stamp = observation['timestamp']
        p.pose.position.x = position[0]
        p.pose.position.y = position[1]
        p.pose.position.z = position[2]
        p.pose.orientation.x = orientation.x
        p.pose.orientation.y = orientation.y
        p.pose.orientation.z = orientation.z
        p.pose.orientation.w = orientation.w
        return p



    def _rgb_to_msg(self, observation: Observation) -> Image:
        """Convert the RGB image from the observation to a ROS Image message."""
        msg = self._bridge.cv2_to_imgmsg(observation['rgb'], "rgb8")
        msg.header.stamp = observation['timestamp']
        return msg



    def _depth_to_msg(self, observation: Observation) -> Image:
        """Convert the depth image from the observation to a ROS Image
        message."""
        msg = self._bridge.cv2_to_imgmsg(observation['depth'], "32FC1")
        msg.header.stamp = observation['timestamp']
        return msg



    def _sem_instances_to_msg(self, observation: Observation) -> Image:
        """Convert the instance ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 16-bit per-pixel instance ID images.
        msg = self._bridge.cv2_to_imgmsg(observation['sem_instances'].astype(np.uint16), "16UC1")
        msg.header.stamp = observation['timestamp']
        return msg



    def _sem_classes_to_msg(self, observation: Observation) -> Image:
        """Convert the class ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 8-bit per-pixel class ID images.
        msg = self._bridge.cv2_to_imgmsg(observation['sem_classes'].astype(np.uint8), "8UC1")
        msg.header.stamp = observation['timestamp']
        return msg



    def _render_sem_instances_to_msg(self, observation: Observation) -> Image:
        """Visualize an instance ID image to a ROS Image message with
        per-instance colours."""
        color_img = self.class_colors[observation['sem_instances'] % 41]
        msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation['timestamp']
        return msg



    def _render_sem_classes_to_msg(self, observation: Observation) -> Image:
        """Visualize a class ID image to a ROS Image message with per-class
        colours."""
        color_img = self.class_colors[observation['sem_classes']]
        msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation['timestamp']
        return msg



    def _camera_intrinsics_to_msg(self, config: Config) -> CameraInfo:
        """Return a ROS message containing the Habitat-Sim camera intrinsic
        parameters."""
        # TODO Set parameters in the message header?
        # http://docs.ros.org/electric/api/sensor_msgs/html/msg/CameraInfo.html
        msg = CameraInfo()
        msg.width = config['width']
        msg.height = config['height']
        msg.K = config['K'].flatten().tolist()
        msg.P = config['P'].flatten().tolist()
        return msg



    def _T_HC_to_T_WB(self, T_HC: np.array) -> np.array:
        """Convert T_HC to T_WB."""
        return self._T_WH @ T_HC @ self._T_CB



    def _T_WB_to_T_HC(self, T_WB: np.array) -> np.array:
        """Convert T_WB to T_HC."""
        return self._T_HW @ T_WB @ self._T_BC



    def _teleport(self) -> None:
        """Move the habitat sensor to the pose contained in self.T_WB."""
        self.T_WB_mutex.acquire()
        t_HC, q_HC = split_pose(self._T_WB_to_T_HC(self.T_WB))
        self.T_WB_mutex.release()
        agent = self.sim.get_agent(0)
        agent_state = hs.agent.AgentState(t_HC, q_HC)
        agent.set_state(agent_state)



    def _render(self, sim: Sim, config: Config) -> Observation:
        """Return the sensor observations and ground truth pose"""
        observation = sim.get_sensor_observations()
        observation['timestamp'] = rospy.get_rostime()
        # Change from RGBA to RGB
        observation['rgb'] = observation['rgb'][..., 0:3]
        if config['enable_semantics'] and config['instance_to_class'].size > 0:
            # Assuming the scene has no more than 65534 objects
            observation['sem_instances'] = np.clip(observation['semantic'].astype(np.uint16), 0, 65535)
            del observation['semantic']
            # Convert instance IDs to class IDs
            observation['sem_classes'] = np.array(
                    [config['instance_to_class'][x] for x in observation['sem_instances']],
                    dtype=np.uint8)
        # Get the camera ground truth pose (T_HC) in the habitat frame from the
        # position and orientation
        t_HC = sim.get_agent(0).get_state().position
        q_HC = sim.get_agent(0).get_state().rotation
        T_HC = combine_pose(t_HC, q_HC)
        observation['T_WB'] = self._T_HC_to_T_WB(T_HC)
        return observation



    def _publish_observation(self, obs: Observation, pub: Publishers, config: Config) -> None:
        """Publish the sensor observations and ground truth pose."""
        pub['pose'].publish(self._pose_to_msg(obs))
        pub['rgb'].publish(self._rgb_to_msg(obs))
        pub['depth'].publish(self._depth_to_msg(obs))
        if config['enable_semantics'] and config['instance_to_class'].size > 0:
            if config['allowed_classes']:
                self._filter_sem_classes(obs)
            pub['sem_class'].publish(self._sem_classes_to_msg(obs))
            pub['sem_instance'].publish(self._sem_instances_to_msg(obs))
            # Publish semantics visualisations
            if config['visualize_semantics']:
                pub['sem_class_render'].publish(self._render_sem_classes_to_msg(obs))
                pub['sem_instance_render'].publish(self._render_sem_instances_to_msg(obs))



if __name__ == "__main__":
    try:
        node = HabitatROSNode()
    except rospy.ROSInterruptException:
        pass

