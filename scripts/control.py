#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import numpy as np
import math
import gym
import time
from stable_baselines3 import DQN, DDPG, PPO, SAC, TD3
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel


# Constants
THRESHOLD = 0.1
GRID = 3.
FRONTIER_THRESH = 2.

class Control(gym.Env):
    """
    Continuous Action Space Gym Environment w/ Obstacles
    State - relative x, y, theta, [sectorized lidar scan]
    Action - linear vel range {0. , MAX_SPEED}, 
             angular vel range {-MAX_STEER, MAX_STEER}
    Reward - 
    """
    def __init__(self):
        super(Control,self).__init__()

        # Get Parameters for Configurations
        self.algorithm = rospy.get_param('~algorithm', 'DQN') # Algorithm for Training (Include: DQN, DDPG, PPO, SAC, TD3)
        self.env_name = rospy.get_param('~env_name', 'DiscreteObs') # Turtlebot Gazebo Gym Environment (Include: Continuous, Discrete, ContinuousObs, DiscreteObs)
        self.device = rospy.get_param('~device', 'cpu') # Run on cpu or cuda (Include: cpu or cuda)

        self.target = [0., 0., 0.]
        self.done = False

        self.pose = np.zeros(3) # pose_callback
        self.sector_scan = np.zeros(36) # scan_callback
        self.depth = np.zeros(1843200) # cam_callback
        # Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
        
        self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

        # Initialize ROS nodes
        self.sub_model = rospy.Subscriber('/gazebo/model_states', ModelStates, self.checkGoalModel)
        self.sub = [0, 0, 0, 0]
        self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        self.sub[1] = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
        self.sub[3] = rospy.Subscriber("/frontier_goal", MoveBaseGoal, self.goal_callback, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        self.rate = rospy.Rate(10)
        
        self.modelGoalPath = rospkg.RosPack().get_path('turtlebot3_gazebo') + '/models/turtlebot3_square/goal_box/model.sdf'
        self.f = open(self.modelGoalPath, 'r')
        self.modelGoal = self.f.read()
        self.check_model = False

        rospack = rospkg.RosPack()
        modelPath = rospack.get_path('aslam_rl') + '/scripts/sbmodels/' + self.algorithm + '_' + self.env_name
        
        if self.algorithm == "DQN":
            self.model = DQN.load(modelPath, device=self.device)
        elif self.algorithm == "DDPG":
            self.model = DDPG.load(modelPath, device=self.device)
        elif self.algorithm == "PPO":
            self.model = PPO.load(modelPath, device=self.device)
        elif self.algorithm == "SAC":
            self.model = SAC.load(modelPath, device=self.device)
        elif self.algorithm == "TD3":
            self.model = TD3.load(modelPath, device=self.device)

    def pose_callback(self, pose_data):
        # ROS Callback function for the /odom topic
        orient = pose_data.pose.pose.orientation
        q = (orient.x, orient.y, orient.z, orient.w)
        euler = self.euler_from_quaternion(q[0], q[1], q[2], q[3])
        self.pose = np.array([pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, euler[2]])

    def scan_callback(self, scan_data):
        # ROS Callback function for the /scan topic
        scan = np.array(scan_data.ranges)
        scan = np.nan_to_num(scan, copy=False, nan=0.0, posinf=5., neginf=0.)
        self.sector_scan = np.min(scan.reshape(-1, 10), axis=1) # Sectorizes the lidar data to 36 sectors of 10 degrees each
    
    def goal_callback(self, goal_data):
        temp = [goal_data.target_pose.pose.position.x, goal_data.target_pose.pose.position.y]
        if self.get_distance(self.target, temp) > FRONTIER_THRESH:
            self.target[0] = goal_data.target_pose.pose.position.x
            self.target[1] = goal_data.target_pose.pose.position.y
            self.done = False

            obs = self.reset()
            while not self.done:
                action, _ = self.model.predict(obs)
                obs, self.done = self.step(action)
                self.rate.sleep()

    def cam_callback(self, cam_data): 
        # ROS Callback function for the /camera/depth/image_raw topic
        self.depth = cam_data.data

    def euler_from_quaternion(self, x, y, z, w):
        '''
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        '''
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

    def reset(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        print("New Goal : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]
        
        self.deleteGoalModel()
        time.sleep(0.5)
        self.respawnGoalModel()

        return np.concatenate((np.array(obs), self.sector_scan))

    def get_distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    def get_heading(self, x1, x2):
        return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

    def check_goal(self):
        done = False
        if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD):
            done = True
            print("Goal Reached!")
            self.stop_bot()

        return done

    def step(self, action):
        done = False

        self.action = [round(x, 2) for x in action]
        msg = Twist()
        msg.linear.x = self.action[0]
        msg.angular.z = self.action[1]
        self.pub.publish(msg)

        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        done = self.check_goal()

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]

        return np.concatenate((np.array(obs), self.sector_scan)), done

    def stop_bot(self):
        msg = Twist()
        msg.linear.x = 0.
        msg.linear.y = 0.
        msg.linear.z = 0.
        msg.angular.x = 0.
        msg.angular.y = 0.
        msg.angular.z = 0.
        self.pub.publish(msg)

    def close(self):
        pass
    
    def checkGoalModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True
                break

    def respawnGoalModel(self):
        while not self.check_model:
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            self.goal_position = Pose()
            self.goal_position.position.x = self.target[0]
            self.goal_position.position.y = self.target[1]
            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox('goal', self.modelGoal, 'robots_name_space', self.goal_position, "world")
            break

    def deleteGoalModel(self):
        while self.check_model:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox('goal')
            break
        
if __name__ == '__main__':
    try:
        rospy.init_node('control', anonymous=True)
        driving = Control()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
