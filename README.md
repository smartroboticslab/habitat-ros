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

### habitat\_ros\_node.py

This is the main node and its respective launch file is
`launch/habitat_ros.launch`.

#### Published topics

This node publishes the ground truth pose, RGB, depth and optionally semantics
at a constant rate.

| Topic name                               | Type                          | Description |
| :--------------------------------------- | :---------------------------- | :---------- |
| `/habitat/pose`                          | `geometry_msgs::PoseStamped`  | The pose (T\_WB) where the images are rendered from. |
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
| `/habitat/external_pose` | `geometry_msgs::PoseStamped` | The pose (T\_WB) where the next images should be rendered from. |

#### Settings

| Setting name | Type    | Description |
| :----------- | :------ | :---------- |
| `width`      | `int`   | The width of all rendered images in pixels. |
| `height`     | `int`   | The height of all rendered images in pixels. |
| `near_plane` | `float` | The near plane of the depth sensor in metres. No depth values smaller than `near_plane` will be produced. |
| `far_plane`  | `float` | The far plane of the depth sensor in metres. No depth values greater than `far_plane` will be produced. |
| `fx`         | `float` | The focal lendth of the sensors in pixels. fy will be the same since Habitat-Sim doesn't currently support different focal lengths between the x and y axes. |
| `fps`        | `float` | The rate at which the ground truth pose and rendered images are published in Hz. Set to 0 to publish as fast as possible. |



## Notes

### Coordinate Frames

- The World (W) frame is z-up and its origin and x-axis direction depend on the
  scene that is being used.
- The Body (B) frame is the standard ROS x-forward, z-up frame.

### Performance

- Enabling the semantic class and instance publishing will reduce performance so
  it is advised to keep them disabled if semantics are not needed.
- Enabling semantic class and instance visualization publishing will reduce
  performance even more so it is advised to keep them disabled if not needed.



## License

Copyright © 2020-2021 Smart Robotics Lab, Imperial College London

Copyright © 2020-2021 Sotiris Papatheodorou

Distributed under the BSD 3-Clause license.

