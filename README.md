# Habitat-ROS

A ROS wrapper for
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim).



## Build

### Install dependencies

Only tested on Ubuntu 20.04 and ROS Noetic. ROS Melodic on Ubuntu 18.04 used
Python2 which will complicate things when trying to use Habitat which is written
on Python3.

``` bash
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev \
    libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev \
    python3-pip python3-attr python3-numba python3-numpy python3-pil \
    python3-scipy python3-tqdm python3-matplotlib python3-git
pip3 install --user numpy-quaternion
# The habitat-sim install script calls pip expecting pip3
sudo ln --symbolic /usr/bin/pip3 /usr/bin/pip
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
catkin build -DCMAKE_BUILD_TYPE=Release habitat-ros
```



## Dataset download

Get the Matterport3D download script from
[here](https://niessner.github.io/Matterport/). Then download the Matterport3D
dataset ready for use with Habitat by running

``` bash
python3 download_mp.py --task habitat -o /path/to/download/
```



## License

Copyright © 2020 Smart Robotics Lab, Imperial College London

Copyright © 2020 Sotiris Papatheodorou

Distributed under the BSD 3-Clause license.

