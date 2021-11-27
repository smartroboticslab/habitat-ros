# Habitat-ROS

A ROS wrapper for
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim). It allows
getting RGB, depth and semantic renders from ROS. It also contains a simplified
MAV simulator.



## Build

### Install dependencies

Only tested on Ubuntu 20.04 and ROS Noetic. Older versions of ROS use Python2
which will complicate things when trying to use Habitat which is written on
Python3.

``` bash
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev \
    libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev \
    python3-pip python3-attr python3-numba python3-numpy python3-pil \
    python3-scipy python3-tqdm python3-matplotlib python3-git
pip3 install --user numpy-quaternion
# The habitat-sim install script calls pip expecting pip3. The following command
# will fail if /usr/local/bin/pip exists.
sudo ln --symbolic /usr/bin/pip3 /usr/local/bin/pip
```

### Build package

This is a normal ROS package so all you have to do is create a workspace, clone
the repository and build. This will build Habitat-Sim which will take some time
and require a lot of RAM and CPU time.

``` bash
mkdir -p ~/catkin_ws/src/ && cd ~/catkin_ws/src/
source /opt/ros/noetic/setup.bash
catkin init
git clone https://bitbucket.org/smartroboticslab/habitat-ros.git
catkin build -DCMAKE_BUILD_TYPE=Release habitat_ros
```



## Dataset download

Get the Matterport3D download script from
[here](https://niessner.github.io/Matterport/). Then download the Matterport3D
dataset ready for use with Habitat by running

``` bash
python3 download_mp.py --task habitat -o /path/to/download/
```



## Usage

### habitat\_node.py

This is the main node and its respective launch file is
`launch/habitat.launch`.

#### Coordinate Frames

- The Habitat frame (H) is z-up and its origin and x-axis direction depend on
  the scene that is being used.
- The Body frame (B) is the standard ROS x-forward, z-up frame.
- The Pose frame (P) is the frame at which ground truth poses are being
  published. Its frame ID can be set using `habitat/pose_frame_id`.
- The External pose frame (E) is the frame at which external poses are being
  received. A transformation from this frame to the Habitat frame (frame ID
  `"habitat"`) will be searched for using
  [`tf2_ros`](http://wiki.ros.org/tf2_ros).

#### Published topics

This node publishes the ground truth pose, RGB, depth and optionally semantics
at a constant rate configurable from `habitat/fps`.

| Topic name                               | Type                          | Description |
| :--------------------------------------- | :---------------------------- | :---------- |
| `/habitat/pose`                          | `geometry_msgs::PoseStamped`  | The pose (T\_PB) where the images are rendered from. |
| `/habitat/depth/image_raw`               | `sensor_msgs::Image (32FC1)`  | The rendered depth image in metres. |
| `/habitat/rgb/image_raw`                 | `sensor_msgs::Image (rgb8)`   | The rendered RGB image. |
| `/habitat/semantic_class/image_raw`      | `sensor_msgs::Image (mono8)`  | The per-pixel semantic class IDs. Each class ID is in the range [0-41]. Published only if `enable_semantics` is `true`. |
| `/habitat/semantic_instance/image_raw`   | `sensor_msgs::Image (mono16)` | The per-pixel semantic instance IDs. Published only if `enable_semantics` is `true`. |
| `/habitat/semantic_class/image_color`    | `sensor_msgs::Image (rgb8)`   | A color render of the semantic class ID image. Useful for visualization and debugging. Published only if `visualize_semantics` is `true`. |
| `/habitat/semantic_instance/image_color` | `sensor_msgs::Image (rgb8)`   | A color render of the semantic instance ID image. Useful for visualization and debugging. Published only if `visualize_semantics` is `true`. |

#### Subscribed topics

The node subscribes to a single topic which is used to set the pose the images
are rendered from. Once a new pose is received, the Habitat-Sim camera is
immediately moved there.

| Topic name               | Type                         | Description |
| :----------------------- | :--------------------------- | :---------- |
| `/habitat/external_pose` | `geometry_msgs::PoseStamped` | The pose (T\_EB) where the next images should be rendered from. |

#### Settings

| Setting name                         | Type    | Description |
| :----------------------------------- | :------ | :---------- |
| `habitat/width`                      | `int`   | The width of all rendered images in pixels. |
| `habitat/height`                     | `int`   | The height of all rendered images in pixels. |
| `habitat/near_plane`                 | `float` | The near plane of the depth sensor in metres. No depth values smaller than `near_plane` will be produced. |
| `habitat/far_plane`                  | `float` | The far plane of the depth sensor in metres. No depth values greater than `far_plane` will be produced. |
| `habitat/f`                          | `float` | The focal length of the sensors in pixels. Habitat-Sim doesn't currently support different focal lengths between the x and y axes. |
| `habitat/fps`                        | `float` | The rate at which the ground truth pose and rendered images are published in Hz. Set to 0 to publish as fast as possible. |
| `habitat/enable_semantics`           | `bool`  | Enable publishing of the semantic class and instance IDs. |
| `habitat/allowed_classes`            | `List`  | Only class IDs present in this list will be present in the output images. All other object classes will have a class and instance ID of 0. Leave empty to return all the available classes. Having a non-empty list significantly impacts performance so its suggested to only use this option for debugging. |
| `habitat/scene_file`                 | `str`   | The path to the .glb scene file to load. The path can be absolute, relative to the habitat\_ros package or it may start with `~` to indicate the home directory of the current user. |
| `habitat/initial_T_HB`               | `List`  | The initial body pose. Can be a translation only `[tx, ty, tz]`, rotation only `[qx, qy, qz, qw]`, translation and rotation `[tx, ty, tz, qx, qy, qz, qw]` or the 16 elements of a homogeneous transformation matrix in row-major order. |
| `habitat/pose_frame_id`              | `str`   | The ID of the frame for poses published in `/habitat/pose`. |
| `habitat/pose_frame_at_initial_T_HB` | `bool`  | Enable publishing a static transform so that the Pose frame (P) coincides with the initial T\_HB pose. This results in the initial published pose being the identity matrix. |
| `habitat/visualize_semantics`        | `bool`  | Generate and publish visualizations of the semantic class and instance IDs. Useful for debugging. |



## Performance

- Enabling the semantic class and instance publishing will reduce performance so
  it is advised to keep them disabled if semantics are not needed.
- Enabling semantic class and instance visualization publishing will reduce
  performance even more so it is advised to keep them disabled if not needed.



## License

Copyright © 2020-2021 Smart Robotics Lab, Imperial College London

Copyright © 2020-2021 Sotiris Papatheodorou

Distributed under the BSD 3-Clause license.

