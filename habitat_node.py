#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

import copy
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
        rospy.loginfo("  {: <20} {}".format(name + ":", str(val)))



def split_pose(T: np.array) -> Tuple[np.array, quaternion.quaternion]:
    """Split a pose in a 4x4 matrix into a position vector and an orientation
    quaternion."""
    return T[0:3, 3], quaternion.from_rotation_matrix(T[0:3, 0:3]).normalized()

def combine_pose(t: np.array, q: quaternion.quaternion) -> np.array:
    """Combine a position vector and an orientation quaternion into a 4x4 pose
    matrix."""
    T = np.identity(4)
    T[0:3, 3] = t
    T[0:3, 0:3] = quaternion.as_rotation_matrix(q.normalized())
    return T

def msg_to_pose(msg: Pose) -> np.array:
    """Convert a ROS Pose message to a 4x4 pose matrix."""
    t = [msg.position.x, msg.position.y, msg.position.z]
    q = quaternion.quaternion(msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z).normalized()
    return combine_pose(t, q)

def msg_to_transform(msg: Transform) -> np.array:
    """Convert a ROS Transform message to a 4x4 transform matrix."""
    t = [msg.translation.x, msg.translation.y, msg.translation.z]
    q = quaternion.quaternion(msg.rotation.w, msg.rotation.x,
            msg.rotation.y, msg.rotation.z).normalized()
    return combine_pose(t, q)

def transform_to_msg(T_TF: np.array, from_frame: str, to_frame: str) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = from_frame;
    msg.child_frame_id = to_frame;
    msg.transform.translation.x = T_TF[0,3]
    msg.transform.translation.y = T_TF[1,3]
    msg.transform.translation.z = T_TF[2,3]
    q_TF = quaternion.from_rotation_matrix(T_TF[0:3, 0:3]).normalized()
    msg.transform.rotation.x = q_TF.x
    msg.transform.rotation.y = q_TF.y
    msg.transform.rotation.z = q_TF.z
    msg.transform.rotation.w = q_TF.w
    return msg

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
        q = quaternion.quaternion(l[3], l[0], l[1], l[2]).normalized()
        T = np.identity(4)
        T[0:3,0:3] = quaternion.as_rotation_matrix(q)
    elif n == 7:
        # Position and orientation quaternion: tx, ty, tz, qx, qy, qz, qw
        q = quaternion.quaternion(l[6], l[3], l[4], l[5]).normalized()
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



def hfov_to_f(hfov: float, width: int) -> float:
    """Convert horizontal field of view in degrees to focal length in pixels.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return 1.0 / (2.0 / float(width) * math.tan(math.radians(hfov) / 2.0))

def f_to_hfov(f: float, width: int) -> float:
    """Convert focal length in pixels to horizontal field of view in degrees.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * f)))



def find_tf(tf_buffer: tf2_ros.Buffer, from_frame: str, to_frame: str) -> Union[np.array, None]:
    """Return the transformation relating the 2 frames."""
    try:
        return msg_to_transform(tf_buffer.lookup_transform(from_frame, to_frame, rospy.Time()).transform)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logfatal('Could not find transform from frame "' + from_frame
                + '" to frame "' + to_frame + '"')
        raise



def remove_invalid_objects(objects: List[hs.scene.SemanticObject]) -> List[hs.scene.SemanticObject]:
    return [x for x in objects if x is not None and x.category is not None]

def get_instance_id(o: hs.scene.SemanticObject) -> int:
    """Return the instance ID of the object."""
    s = o.id.strip("_")
    if "_" in s:
        return [int(x) for x in s.split("_")][2]
    else:
        return int(s)



class HabitatROSNode:
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
        [0x7f, 0x7f, 0x7f]
    ])

    # Instantiate a single CvBridge object for all conversions
    _bridge = CvBridge()

    # Published topic names
    _rgb_topic_name = "~rgb/"
    _depth_topic_name = "~depth/"
    _sem_class_topic_name = "~semantic_class/"
    _sem_instance_topic_name = "~semantic_instance/"
    _habitat_pose_topic_name = "~pose"

    # Subscribed topic names
    _external_pose_topic_name = "~external_pose"

    # Transforms between the internal habitat frame I (y-up) and the exported
    # habitat frame H (z-up)
    _T_HI = np.identity(4)
    _T_HI[0:3, 0:3] = quaternion.as_rotation_matrix(hs.utils.common.quat_from_two_vectors(
            hs.geo.GRAVITY, np.array([0.0, 0.0, -1.0])))
    _T_IH = np.linalg.inv(_T_HI)

    # Transforms between the habitat camera frame C (-z-forward, y-up) and the
    # ROS body frame B (x-forward, z-up)
    _T_CB = np.array([(0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)])
    _T_BC = np.linalg.inv(_T_CB)

    # The default node options
    _default_config = {
            "width": 640,
            "height": 480,
            "near_plane": 0.1,
            "far_plane": 10.0,
            "f": 525.0,
            "fps": 30,
            "enable_semantics": False,
            "allowed_classes": [],
            "scene_file": "",
            "initial_T_HB": [],
            "pose_frame_id": "habitat",
            "pose_frame_at_initial_T_HB": False,
            "visualize_semantics": False}



    def __init__(self):
        # Initialize the node, habitat-sim and publishers
        rospy.init_node("habitat")
        self.config = self._read_node_config()
        self.sim = self._init_habitat(self.config)
        self.pub = self._init_publishers(self.config)
        # Initialize the pose mutex
        self.T_HB_mutex = threading.Lock()
        # Initialize the transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # Publish the T_HP transform so that the P frame coincides with the
        # initial pose
        if self.config["pose_frame_at_initial_T_HB"] and self.config["pose_frame_id"] != "habitat":
            T_HP = self.T_HB
            T_HP_msg = transform_to_msg(T_HP, "habitat", self.config["pose_frame_id"])
            self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()
            self.tf_static_broadcaster.sendTransform(T_HP_msg)
            # Wait for the listener to pick up the transform
            rospy.sleep(0.1)
        # Setup the external pose subscriber
        rospy.Subscriber(self._external_pose_topic_name, PoseStamped,
                self._pose_callback, queue_size=1)
        rospy.loginfo("Habitat node ready")
        # Main loop
        if self.config["fps"] > 0:
            rate = rospy.Rate(self.config["fps"])
        while not rospy.is_shutdown():
            # Move, observe and publish
            observation = self._move_and_render(self.sim, self.config)
            self._publish_observation(observation, self.pub, self.config)
            if self.config["fps"] > 0:
                rate.sleep()



    def _read_node_config(self) -> Config:
        """Read the node parameters, print them and return a dictionary."""
        # Read the parameters
        config = read_config(self._default_config)
        # Get an absolute path from the supplied scene file
        config["scene_file"] = os.path.expanduser(config["scene_file"])
        if not os.path.isabs(config["scene_file"]):
            # The scene file path is relative, assuming relative to the ROS package
            package_path = rospkg.RosPack().get_path("habitat_ros") + "/"
            config["scene_file"] = package_path + config["scene_file"]
        # Ensure a valid scene file was supplied
        if not config["scene_file"]:
            rospy.logfatal("No scene file supplied")
            raise rospy.ROSException
        elif not os.path.isfile(config["scene_file"]):
            rospy.logfatal("Scene file " + config["scene_file"] + " does not exist")
            raise rospy.ROSException
        # Create the initial T_HB matrix
        T = list_to_pose(config["initial_T_HB"])
        if T is None and config["initial_T_HB"]:
            rospy.logerr("Invalid initial T_HB. Expected list of 3, 4, 7 or 16 elements")
        config["initial_T_HB"] = T
        rospy.loginfo("Habitat node parameters:")
        print_config(config)
        return config



    def _init_habitat(self, config: Config) -> Sim:
        """Initialize the Habitat simulator, create the sensors and load the
        scene file."""
        backend_config = hs.SimulatorConfiguration()
        backend_config.scene_id = (config["scene_file"])
        agent_config = hs.AgentConfiguration()
        agent_config.sensor_specifications = [self._rgb_sensor_config(config),
                self._depth_sensor_config(config), self._semantic_sensor_config(config)]
        agent_config.height = 0.0
        agent_config.radius = 0.0
        sim = Sim(hs.Configuration(backend_config, [agent_config]))
        # Get the intrinsic camera parameters
        hfov = float(agent_config.sensor_specifications[0].hfov)
        f = hfov_to_f(hfov, config["width"])
        cx = config["width"] / 2.0 - 0.5
        cy = config["height"] / 2.0 - 0.5
        config["K"] = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64)
        config["P"] = np.array([[f, 0.0, cx, 0.0], [0.0, f, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64)
        self.class_id_to_name = self._class_id_to_name_map(sim.semantic_scene.categories)
        # Setup the instance/class conversion map
        if config["enable_semantics"]:
            config["instance_to_class"] = self._instance_to_class_map(
                    remove_invalid_objects(sim.semantic_scene.objects), self.class_id_to_name)
            if config["instance_to_class"].size == 0:
                rospy.logwarn("The scene contains no semantics")
        # Get or set the initial agent pose
        agent = sim.get_agent(0)
        if config["initial_T_HB"] is None:
            t_IC = agent.get_state().position
            q_IC = agent.get_state().rotation
            T_IC = combine_pose(t_IC, q_IC)
            self.T_HB = self._T_IC_to_T_HB(T_IC)
        else:
            self.T_HB = config["initial_T_HB"]
            t_IC, q_IC = split_pose(self._T_HB_to_T_IC(self.T_HB))
            agent_state = hs.agent.AgentState(t_IC, q_IC)
            agent.set_state(agent_state)
        t_HB, q_HB = split_pose(self.T_HB)
        # Initialize the current pose timestamp to zero.
        self.T_HB_stamp = rospy.Time()
        self.T_HB_received = False
        rospy.loginfo("Habitat initial t_HB (x,y,z):   {}, {}, {}".format(
            t_HB[0], t_HB[1], t_HB[2]))
        rospy.loginfo("Habitat initial q_HB (x,y,z,w): {}, {}, {}, {}".format(
            q_HB.x, q_HB.y, q_HB.z, q_HB.w))
        return sim



    def _rgb_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat color sensor."""
        rgb_sensor_spec = hs.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = hs.SensorType.COLOR
        rgb_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        rgb_sensor_spec.resolution = [config["height"], config["width"]]
        rgb_sensor_spec.near = 0.00001
        rgb_sensor_spec.far = 1000
        rgb_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        rgb_sensor_spec.position = np.zeros((3, 1))
        rgb_sensor_spec.orientation = np.zeros((3, 1))
        return rgb_sensor_spec



    def _depth_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat depth sensor."""
        depth_sensor_spec = hs.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
        depth_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        depth_sensor_spec.resolution = [config["height"], config["width"]]
        depth_sensor_spec.near = config["near_plane"]
        depth_sensor_spec.far = config["far_plane"]
        depth_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        depth_sensor_spec.position = np.zeros((3, 1))
        depth_sensor_spec.orientation = np.zeros((3, 1))
        return depth_sensor_spec



    def _semantic_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat semantic sensor."""
        semantic_sensor_spec = hs.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic"
        semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
        semantic_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        semantic_sensor_spec.resolution = [config["height"], config["width"]]
        semantic_sensor_spec.near = 0.00001
        semantic_sensor_spec.far = 1000
        semantic_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        semantic_sensor_spec.position = np.zeros((3, 1))
        semantic_sensor_spec.orientation = np.zeros((3, 1))
        return semantic_sensor_spec



    def _class_id_to_name_map(self, categories: List) -> Dict[int, str]:
        """Generate a dictionary from class IDs to class names."""
        return {x.index(): x.name() for x in categories if x is not None}



    def _instance_to_class_map(self, objects: List[hs.scene.SemanticObject], classes: Dict[int, str]) -> np.ndarray:
        """Given the objects in the scene, create an array that maps instance
        IDs to class IDs."""
        # Default is -1 so that an empty array is created in the following line
        # if there are no objects.
        max_instance_id = max([get_instance_id(x) for x in objects], default=-1)
        mapping = np.zeros(max_instance_id + 1, dtype=np.uint8)
        for object in objects:
            instance_id = get_instance_id(object)
            mapping[instance_id] = object.category.index()
            if mapping[instance_id] not in classes.keys():
                rospy.logwarn('Invalid object class ID/name {}/"{}", replacing with 0/"{}"'.format(
                    mapping[instance_id], object.category.name(), classes[0]))
                mapping[instance_id] = 0
        return mapping



    def _init_publishers(self, config: Config) -> Publishers:
        """Initialize and return the image and pose publishers."""
        image_queue_size = 10
        pub = {}
        # Pose publisher
        pub["pose"] = rospy.Publisher(self._habitat_pose_topic_name, PoseStamped, queue_size=10)
        # Image publishers
        pub["rgb"] = rospy.Publisher(self._rgb_topic_name + "image_raw",
                Image, queue_size=image_queue_size)
        pub["depth"] = rospy.Publisher(self._depth_topic_name + "image_raw",
                Image, queue_size=image_queue_size)
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            # Only publish semantics if the scene contains semantics
            pub["sem_class"] = rospy.Publisher(self._sem_class_topic_name + "image_raw",
                    Image, queue_size=image_queue_size)
            pub["sem_instance"] = rospy.Publisher(self._sem_instance_topic_name + "image_raw",
                    Image, queue_size=image_queue_size)
            if config["visualize_semantics"]:
                pub["sem_class_render"] = rospy.Publisher(self._sem_class_topic_name + "image_color",
                        Image, queue_size=image_queue_size)
                pub["sem_instance_render"] = rospy.Publisher(self._sem_instance_topic_name + "image_color",
                        Image, queue_size=image_queue_size)
        # Publish the camera info for each image topic
        image_topics = [self._rgb_topic_name, self._depth_topic_name]
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            image_topics += [self._sem_class_topic_name, self._sem_instance_topic_name]
        for topic in image_topics:
            pub[topic + "_camera_info"] = rospy.Publisher(topic + "camera_info",
                CameraInfo, queue_size=1, latch=True)
            pub[topic + "_camera_info"].publish(self._camera_intrinsics_to_msg(config))
        return pub



    def _pose_callback(self, pose: PoseStamped) -> None:
        """Callback for receiving external pose messages. It updates the agent
        pose."""
        # Find the transform from the pose frame F to the habitat frame H
        T_HE = find_tf(self.tf_buffer, "habitat", pose.header.frame_id)
        # Transform the pose
        T_EB = msg_to_pose(pose.pose)
        T_HB = T_HE @ T_EB
        # Update the pose
        self.T_HB_mutex.acquire()
        self.T_HB = T_HB
        self.T_HB_stamp = pose.header.stamp
        self.T_HB_received = True
        self.T_HB_mutex.release()



    def _filter_sem_classes(self, observation: Observation) -> None:
        """Remove object detections whose classes are not in the allowed class
        list. Their class and instance IDs are set to 0."""
        # Generate a per-pixel boolean matrix
        allowed = np.vectorize(lambda x: x in self.config["allowed_classes"])
        allowed_pixels = allowed(observation["sem_classes"])
        # Set all False pixels to 0 on the class and instance images
        class_zeros = np.zeros(observation["sem_classes"].shape, dtype=observation["sem_classes"].dtype)
        instance_zeros = np.zeros(observation["sem_instances"].shape, dtype=observation["sem_instances"].dtype)
        observation["sem_classes"] = np.where(allowed_pixels, observation["sem_classes"], class_zeros)
        observation["sem_instances"] = np.where(allowed_pixels, observation["sem_instances"], instance_zeros)



    def _pose_to_msg(self, observation: Observation) -> PoseStamped:
        """Convert the agent pose from the observation to a ROS PoseStamped
        message."""
        T_PH = find_tf(self.tf_buffer, self.config["pose_frame_id"], "habitat")
        t_HB, q_HB = split_pose(T_PH @ observation["T_HB"])
        p = PoseStamped()
        p.header.frame_id = self.config["pose_frame_id"]
        p.header.stamp = observation["timestamp"]
        p.pose.position.x = t_HB[0]
        p.pose.position.y = t_HB[1]
        p.pose.position.z = t_HB[2]
        p.pose.orientation.x = q_HB.x
        p.pose.orientation.y = q_HB.y
        p.pose.orientation.z = q_HB.z
        p.pose.orientation.w = q_HB.w
        return p



    def _rgb_to_msg(self, observation: Observation) -> Image:
        """Convert the RGB image from the observation to a ROS Image message."""
        msg = self._bridge.cv2_to_imgmsg(observation["rgb"], "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg



    def _depth_to_msg(self, observation: Observation) -> Image:
        """Convert the depth image from the observation to a ROS Image
        message."""
        msg = self._bridge.cv2_to_imgmsg(observation["depth"], "32FC1")
        msg.header.stamp = observation["timestamp"]
        return msg



    def _sem_instances_to_msg(self, observation: Observation) -> Image:
        """Convert the instance ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 16-bit per-pixel instance ID images.
        msg = self._bridge.cv2_to_imgmsg(observation["sem_instances"].astype(np.uint16), "16UC1")
        msg.header.stamp = observation["timestamp"]
        return msg



    def _sem_classes_to_msg(self, observation: Observation) -> Image:
        """Convert the class ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 8-bit per-pixel class ID images.
        msg = self._bridge.cv2_to_imgmsg(observation["sem_classes"].astype(np.uint8), "8UC1")
        msg.header.stamp = observation["timestamp"]
        return msg



    def _render_sem_instances_to_msg(self, observation: Observation) -> Image:
        """Visualize an instance ID image to a ROS Image message with
        per-instance colours."""
        color_img = self.class_colors[observation["sem_instances"] % len(self.class_colors)]
        color_img = color_img / 2 + observation["rgb"] / 2
        msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg



    def _render_sem_classes_to_msg(self, observation: Observation) -> Image:
        """Visualize a class ID image to a ROS Image message with per-class
        colours."""
        color_img = self.class_colors[observation["sem_classes"] % len(self.class_colors)]
        color_img = color_img / 2 + observation["rgb"] / 2
        msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg



    def _camera_intrinsics_to_msg(self, config: Config) -> CameraInfo:
        """Return a ROS message containing the Habitat-Sim camera intrinsic
        parameters."""
        # TODO Set parameters in the message header?
        # http://docs.ros.org/electric/api/sensor_msgs/html/msg/CameraInfo.html
        msg = CameraInfo()
        msg.width = config["width"]
        msg.height = config["height"]
        msg.K = config["K"].flatten().tolist()
        msg.P = config["P"].flatten().tolist()
        return msg



    def _T_IC_to_T_HB(self, T_IC: np.array) -> np.array:
        """Convert T_IC to T_HB."""
        return self._T_HI @ T_IC @ self._T_CB



    def _T_HB_to_T_IC(self, T_HB: np.array) -> np.array:
        """Convert T_HB to T_IC."""
        return self._T_IH @ T_HB @ self._T_BC



    def _move_and_render(self, sim: Sim, config: Config) -> Observation:
        """Move the habitat sensor and return its observations and ground truth
        pose."""
        # Receive the latest pose.
        self.T_HB_mutex.acquire()
        T_HB = np.copy(self.T_HB)
        stamp = copy.deepcopy(self.T_HB_stamp)
        T_HB_received = self.T_HB_received
        self.T_HB_received = False
        self.T_HB_mutex.release()
        # Move the sensor to the pose contained in self.T_HB.
        t_IC, q_IC = split_pose(self._T_HB_to_T_IC(T_HB))
        agent_state = hs.agent.AgentState(t_IC, q_IC)
        self.sim.get_agent(0).set_state(agent_state)
        # Render the sensor observations.
        observation = sim.get_sensor_observations()
        if T_HB_received:
            # Set the observation timestamp to that of the received pose to keep
            # them in sync.
            observation["timestamp"] = stamp
        else:
            # No new pose received yet, use the current timestamp.
            observation["timestamp"] = rospy.get_rostime()
        # Change from RGBA to RGB
        observation["rgb"] = observation["rgb"][..., 0:3]
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            # Assuming the scene has no more than 65534 objects
            observation["sem_instances"] = np.clip(observation["semantic"].astype(np.uint16), 0, 65535)
            del observation["semantic"]
            # Convert instance IDs to class IDs
            observation["sem_classes"] = np.array(
                    [config["instance_to_class"][x] for x in observation["sem_instances"]],
                    dtype=np.uint8)
        # Get the camera ground truth pose (T_IC) in the habitat frame from the
        # position and orientation
        t_IC = sim.get_agent(0).get_state().position
        q_IC = sim.get_agent(0).get_state().rotation
        T_IC = combine_pose(t_IC, q_IC)
        observation["T_HB"] = self._T_IC_to_T_HB(T_IC)
        return observation



    def _publish_observation(self, obs: Observation, pub: Publishers, config: Config) -> None:
        """Publish the sensor observations and ground truth pose."""
        pub["pose"].publish(self._pose_to_msg(obs))
        pub["rgb"].publish(self._rgb_to_msg(obs))
        pub["depth"].publish(self._depth_to_msg(obs))
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            if config["allowed_classes"]:
                self._filter_sem_classes(obs)
            pub["sem_class"].publish(self._sem_classes_to_msg(obs))
            pub["sem_instance"].publish(self._sem_instances_to_msg(obs))
            # Publish semantics visualisations
            if config["visualize_semantics"]:
                pub["sem_class_render"].publish(self._render_sem_classes_to_msg(obs))
                pub["sem_instance_render"].publish(self._render_sem_instances_to_msg(obs))



if __name__ == "__main__":
    try:
        node = HabitatROSNode()
    except rospy.ROSInterruptException:
        pass

