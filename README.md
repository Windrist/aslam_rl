# ASLAM With ADNet and DRL
ASLAM for Robotics, combine with Object Tracking and Deep Reinforcement Learning!

## Core Feature:
```bash
Frontier Detection supported by ADNet Object Tracking
Gmapping SLAM
Navigation by DRL (Developing)
Turtlebot3 Gazebo
...
```

## Requirements
- Ubuntu 16.04 / 18.04 / 20.04
- ROS - Robot Operating System
- Python 3.6 / Anaconda
- GPU Recommended

## Installation
```bash
mkdir -p Workspace/ThichNghi_ws/src && cd Workspace/ThichNghi_ws
cd src && git clone https://github.com/Windrist/aslam_rl
cd aslam_rl
git submodule update
conda env create -f environment.yml
conda activate ThichNghi
```

## Usage

#### To Run on Real Environment:
```bash
roslaunch aslam_rl turtlebot3_house.launch
roslaunch aslam_rl control.launch
roslaunch aslam_rl autonomous_explorer.launch
```
#### To Train:
```bash
roslaunch aslam_rl train.launch
```
#### To Evaluate:
```bash
roslaunch aslam_rl evaluate.launch
```
#### Check Log Folder for Previous Result:
```bash
tensorboard --logdir log
```

## Credits
This is Reinforcement Learning Project and Research for UET Course!
Code is Maintaining and Developing!

Members:
- Tran Huu Quoc Dong
- Ngo Thi Ngoc Quyen