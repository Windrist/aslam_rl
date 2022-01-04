#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
from gym import spaces
import rospy
import time
import math
import random
import collections
import rospkg
from std_msgs.msg import Bool, Float32, Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from gazebo_msgs.msg import ModelStates
import numpy as np
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

# Constants
MAX_STEER = 2.84
MAX_SPEED = 0.22
THRESHOLD = 0.1
GRID = 3.
MAX_EP_LEN = 1000
OBS_THRESH = 0.15
GOAL_X_LIST = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
GOAL_Y_LIST = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]
GOAL_X_LIST_WORLD = [-0.4, 0.6, 0.6, -1.4, -1.4, 2.0, 2.0, -2.5, 1.0, -1.0]
GOAL_Y_LIST_WORLD = [-0.4, 0.6, -1.4, 0.6, -1.4, 1.0, -1.0, 0.0, 2.0, 2.0]

class ContinuousTurtleGym(gym.Env):
    """
    Continuous Action Space Gym Environment
    State - relative x, y, theta, [sectorized lidar scan]
    Action - linear vel range {-0.22 , 0.22}, 
             angular vel range {-2.84, 2.84}
    Reward - 
    """
    def __init__(self, train_state = 1):
        super(ContinuousTurtleGym,self).__init__()
        metadata = {'render.modes': ['console']}
        print("Initialising Turtlebot 3 Continuous Gym Environment...")
        self.action_space = spaces.Box(np.array([-MAX_SPEED, -MAX_STEER]), np.array([MAX_SPEED, MAX_STEER]), dtype = np.float16) # max rotational velocity of burger is 2.84 rad/s
        
        low = np.concatenate((np.array([-1., -1., -4.]), np.zeros(36)))
        high = np.concatenate((np.array([1., 1., 4.]), np.ones(36)*5.))
        self.observation_space = spaces.Box(low, high, dtype=np.float16)
        self.target = [0., 0., 0.]
        self.ep_steps = 0
        self.train_state = train_state

        self.pose = np.zeros(3) # pose_callback
        self.sector_scan = np.zeros(36) # scan_callback
        self.depth = np.zeros(1843200) # cam_callback. 
        # Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
        
        self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

        # Initialize ROS nodes
        self.sub_model = rospy.Subscriber('/gazebo/model_states', ModelStates, self.checkGoalModel)
        self.sub = [0, 0, 0]
        self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        self.sub[1] = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        self.rate = rospy.Rate(10)

        # Gazebo Services
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        self.modelPath = rospkg.RosPack().get_path('turtlebot3_gazebo') + '/models/turtlebot3_square/goal_box/model.sdf'
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.check_model = False
        
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = 0.0
        self.last_goal_y = 0.0

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
        self.sector_scan = np.mean(scan.reshape(-1, 10), axis=1) # Sectorizes the lidar data to 36 sectors of 10 degrees each

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
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.pause()
            self.reset_simulation_proxy()
            self.unpause()
            print('Simulation reset')
        except rospy.ServiceException as exc:
            print("Reset Service did not process request: " + str(exc))

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        
        if self.train_state == 4:
            index = random.randrange(0, 13)
            x = GOAL_X_LIST[index]
            y = GOAL_Y_LIST[index]
        elif self.train_state == 5:
            index = random.randrange(0, 10)
            x = GOAL_X_LIST_WORLD[index]
            y = GOAL_Y_LIST_WORLD[index]
        else:
            position_check = True
            while position_check:
                x = random.uniform(-1.5, 1.5)
                y = random.uniform(-1.5, 1.5)
                if abs(x - self.obstacle_1[0]) <= 0.4 and abs(y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_2[0]) <= 0.4 and abs(y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_3[0]) <= 0.4 and abs(y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_4[0]) <= 0.4 and abs(y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(x - 0.0) <= 0.4 and abs(y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False
                if abs(x - self.last_goal_x) < 1 and abs(y - self.last_goal_y) < 1:
                    position_check = True
        self.target[0], self.target[1] = [x, y]
        self.last_goal_x = x
        self.last_goal_y = y

        print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]
        self.ep_steps = 0

        self.deleteGoalModel()
        time.sleep(0.5)
        self.respawnGoalModel()

        return np.concatenate((np.array(obs), self.sector_scan))

    def get_distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    def get_heading(self, x1, x2):
        return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

    def get_reward(self):
        yaw_car = self.pose[2]
        head_to_target = self.get_heading(self.pose, self.target)

        alpha = head_to_target - yaw_car
        ld = self.get_distance(self.pose, self.target)
        crossTrackError = math.sin(alpha) * ld

        headingError = abs(alpha)
        alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])

        # return -1*(abs(crossTrackError)**2 + alongTrackError + 3*headingError/1.57)/6
        return -alongTrackError

    def check_goal(self):
        done = False
        reward = self.get_reward()

        if self.ep_steps > MAX_EP_LEN:
            print("Reached Max Episode Length!")
            reward -= 100
            done = True
            self.stop_bot()
        else:
            if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD):
                done = True
                reward += 1000
                print("Goal Reached!")
                self.stop_bot()
            else:
                if np.min(self.sector_scan) < OBS_THRESH:
                    print("Collision Detected!")
                    reward -= 500
                    self.stop_bot()
                    done = True

        return done, reward

    def step(self, action):
        reward = 0
        done = False
        info = {}
        self.ep_steps += 1

        self.action = [round(x, 2) for x in action]
        msg = Twist()
        msg.linear.x = self.action[0]
        msg.angular.z = self.action[1]
        self.pub.publish(msg)
        self.rate.sleep()

        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        done, reward = self.check_goal()

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]

        return np.concatenate((np.array(obs), self.sector_scan)), reward, done, info

    def stop_bot(self):
        msg = Twist()
        msg.linear.x = 0.
        msg.linear.y = 0.
        msg.linear.z = 0.
        msg.angular.x = 0.
        msg.angular.y = 0.
        msg.angular.z = 0.
        self.pub.publish(msg)
        self.rate.sleep()

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
            spawn_model_prox('goal', self.model, 'robots_name_space', self.goal_position, "world")
            break

    def deleteGoalModel(self):
        while self.check_model:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox('goal')
            break

class DiscreteTurtleGym(gym.Env):
    """
    Discrete Action Space Gym Environment
    State - relative x, y, theta, [sectorized lidar scan]
    Action - 5 actions - 
                0: [0., 0.], 1: [MAX_SPEED, 0.], 2: [-MAX_SPEED, 0.], 3: [0., MAX_STEER/2], 4: [0., -MAX_STEER/2]
             25 actions - 
                 0: [0., -MAX_STEER], 1: [0., -MAX_STEER/2], 2: [0., 0.], 3: [0., MAX_STEER/2], 4: [0., MAX_STEER],
                5: [MAX_SPEED/2, -MAX_STEER], 6: [MAX_SPEED/2, -MAX_STEER/2], 7: [MAX_SPEED/2, 0.], 8: [MAX_SPEED/2, MAX_STEER/2], 9: [MAX_SPEED/2, MAX_STEER],
                10: [MAX_SPEED, -MAX_STEER], 11: [MAX_SPEED, -MAX_STEER/2], 12: [MAX_SPEED, 0.], 13: [MAX_SPEED, MAX_STEER/2], 14: [MAX_SPEED, MAX_STEER],
                15: [-MAX_SPEED/2, -MAX_STEER], 16: [-MAX_SPEED/2, -MAX_STEER/2], 17: [-MAX_SPEED/2, 0.], 18: [-MAX_SPEED/2, MAX_STEER/2], 19: [-MAX_SPEED/2, MAX_STEER],
                20: [-MAX_SPEED, -MAX_STEER], 21: [-MAX_SPEED, -MAX_STEER/2], 22: [-MAX_SPEED, 0.], 23: [-MAX_SPEED, MAX_STEER/2], 24: [-MAX_SPEED, MAX_STEER]
    Reward - 
    """
    def __init__(self, n_actions = 'Minimum', train_state = 1):
        super(DiscreteTurtleGym,self).__init__()		
        metadata = {'render.modes': ['console']}
        print("Initialising Turtlebot 3 Discrete Gym Environment...")
        
        self.actSpace = collections.defaultdict(list)
        if n_actions == 'Minimum':	
            self.action_space = spaces.Discrete(5)		
            self.actSpace = {
                0: [0., 0.], 1: [MAX_SPEED, 0.], 2: [-MAX_SPEED, 0.], 3: [0., MAX_STEER/2], 4: [0., -MAX_STEER/2]
            }
        elif n_actions == 'Maximum':
            self.action_space = spaces.Discrete(25)
            self.actSpace = {
                0: [0., -MAX_STEER], 1: [0., -MAX_STEER/2], 2: [0., 0.], 3: [0., MAX_STEER/2], 4: [0., MAX_STEER],
                5: [MAX_SPEED/2, -MAX_STEER], 6: [MAX_SPEED/2, -MAX_STEER/2], 7: [MAX_SPEED/2, 0.], 8: [MAX_SPEED/2, MAX_STEER/2], 9: [MAX_SPEED/2, MAX_STEER],
                10: [MAX_SPEED, -MAX_STEER], 11: [MAX_SPEED, -MAX_STEER/2], 12: [MAX_SPEED, 0.], 13: [MAX_SPEED, MAX_STEER/2], 14: [MAX_SPEED, MAX_STEER],
                15: [-MAX_SPEED/2, -MAX_STEER], 16: [-MAX_SPEED/2, -MAX_STEER/2], 17: [-MAX_SPEED/2, 0.], 18: [-MAX_SPEED/2, MAX_STEER/2], 19: [-MAX_SPEED/2, MAX_STEER],
                20: [-MAX_SPEED, -MAX_STEER], 21: [-MAX_SPEED, -MAX_STEER/2], 22: [-MAX_SPEED, 0.], 23: [-MAX_SPEED, MAX_STEER/2], 24: [-MAX_SPEED, MAX_STEER]
            }

        low = np.concatenate((np.array([-1., -1., -4.]), np.zeros(36)))
        high = np.concatenate((np.array([1., 1., 4.]), np.ones(36)*5.))
        self.observation_space = spaces.Box(low, high, dtype=np.float16)
        self.target = [0., 0., 0.]
        self.ep_steps = 0
        self.train_state = train_state

        self.pose = np.zeros(3) # pose_callback
        self.sector_scan = np.zeros(36) # scan_callback
        self.depth = np.zeros(1843200) # cam_callback. 
        # Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
        
        self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

        # Initialize ROS nodes
        self.sub_model = rospy.Subscriber('/gazebo/model_states', ModelStates, self.checkGoalModel)
        self.sub = [0, 0, 0]
        self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        self.sub[1] = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        self.rate = rospy.Rate(10)

        # Gazebo Services
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.modelPath = rospkg.RosPack().get_path('turtlebot3_gazebo') + '/models/turtlebot3_square/goal_box/model.sdf'
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.check_model = False

        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = 0.0
        self.last_goal_y = 0.0

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
        self.sector_scan = np.mean(scan.reshape(-1, 10), axis=1) # Sectorizes the lidar data to 36 sectors of 10 degrees each

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
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.pause()
            self.reset_simulation_proxy()
            self.unpause()
            print('Simulation reset')
        except rospy.ServiceException as exc:
            print("Reset Service did not process request: " + str(exc))

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.train_state == 4:
            index = random.randrange(0, 13)
            x = GOAL_X_LIST[index]
            y = GOAL_Y_LIST[index]
        elif self.train_state == 5:
            index = random.randrange(0, 10)
            x = GOAL_X_LIST_WORLD[index]
            y = GOAL_Y_LIST_WORLD[index]
        else:
            position_check = True
            while position_check:
                x = random.uniform(-1.5, 1.5)
                y = random.uniform(-1.5, 1.5)
                if abs(x - self.obstacle_1[0]) <= 0.4 and abs(y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_2[0]) <= 0.4 and abs(y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_3[0]) <= 0.4 and abs(y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_4[0]) <= 0.4 and abs(y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(x - 0.0) <= 0.4 and abs(y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False
                if abs(x - self.last_goal_x) < 1 and abs(y - self.last_goal_y) < 1:
                    position_check = True
        self.target[0], self.target[1] = [x, y]
        self.last_goal_x = x
        self.last_goal_y = y

        print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]
        self.ep_steps = 0

        self.deleteGoalModel()
        time.sleep(0.5)
        self.respawnGoalModel()

        return np.concatenate((np.array(obs), self.sector_scan))

    def get_distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    def get_heading(self, x1, x2):
        return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

    def get_reward(self):
        yaw_car = self.pose[2]
        head_to_target = self.get_heading(self.pose, self.target)

        alpha = head_to_target - yaw_car
        ld = self.get_distance(self.pose, self.target)
        crossTrackError = math.sin(alpha) * ld

        headingError = abs(alpha)
        alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])

        # return -1*(abs(crossTrackError)**2 + alongTrackError + 3*headingError/1.57)/6
        return -alongTrackError

    def check_goal(self):
        done = False
        reward = self.get_reward()

        if self.ep_steps > MAX_EP_LEN:
            print("Reached Max Episode Length!")
            reward -= 100
            done = True
            self.stop_bot()
        else:
            if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD):
                done = True
                reward += 1000
                print("Goal Reached!")
                self.stop_bot()
            else:
                if np.min(self.sector_scan) < OBS_THRESH:
                    print("Collision Detected!")
                    reward -= 500
                    self.stop_bot()
                    done = True

        return done, reward

    def step(self, discrete_action):
        reward = 0
        done = False
        info = {}
        self.ep_steps += 1

        action = self.actSpace[discrete_action]
        
        self.action = [round(x, 2) for x in action]
        msg = Twist()
        msg.linear.x = self.action[0]
        msg.angular.z = self.action[1]
        self.pub.publish(msg)
        self.rate.sleep()

        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        done, reward = self.check_goal()

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]

        return np.concatenate((np.array(obs), self.sector_scan)), reward, done, info

    def stop_bot(self):
        msg = Twist()
        msg.linear.x = 0.
        msg.linear.y = 0.
        msg.linear.z = 0.
        msg.angular.x = 0.
        msg.angular.y = 0.
        msg.angular.z = 0.
        self.pub.publish(msg)
        self.rate.sleep()

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
            spawn_model_prox('goal', self.model, 'robots_name_space', self.goal_position, "world")
            break

    def deleteGoalModel(self):
        while self.check_model:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox('goal')
            break

class ContinuousTurtleObsGym(gym.Env):
    """
    Continuous Action Space Gym Environment w/ Obstacles
    State - relative x, y, theta, [sectorized lidar scan]
    Action - linear vel range {0. , MAX_SPEED}, 
             angular vel range {-MAX_STEER, MAX_STEER}
    Reward - 
    """
    def __init__(self, train_state = 1):
        super(ContinuousTurtleObsGym,self).__init__()		
        metadata = {'render.modes': ['console']}
        print("Initialising Turtlebot 3 Continuous Gym Obstacle Environment...")
        
        self.action_space = spaces.Box(np.array([0., -MAX_STEER]), np.array([MAX_SPEED, MAX_STEER]), dtype = np.float16) # max rotational velocity of burger is 2.84 rad/s
        
        low = np.concatenate((np.array([-1., -1., -4.]), np.zeros(36)))
        high = np.concatenate((np.array([1., 1., 4.]), np.ones(36)*5.))
        self.observation_space = spaces.Box(low, high, dtype=np.float16)
        self.target = [0., 0., 0.]
        self.ep_steps = 0
        self.train_state = train_state

        self.pose = np.zeros(3) # pose_callback
        self.sector_scan = np.zeros(36) # scan_callback
        self.depth = np.zeros(1843200) # cam_callback
        # Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
        
        self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

        # Initialize ROS nodes
        self.sub_model = rospy.Subscriber('/gazebo/model_states', ModelStates, self.checkGoalModel)
        self.sub = [0, 0, 0]
        self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        self.sub[1] = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        self.rate = rospy.Rate(10)

        # Gazebo Services
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.modelPath = rospkg.RosPack().get_path('turtlebot3_gazebo') + '/models/turtlebot3_square/goal_box/model.sdf'
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.check_model = False

        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = 0.0
        self.last_goal_y = 0.0

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
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.pause()
            self.reset_simulation_proxy()
            self.unpause()
            print('Simulation reset')
        except rospy.ServiceException as exc:
            print("Reset Service did not process request: " + str(exc))

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        
        if self.train_state == 4:
            index = random.randrange(0, 13)
            x = GOAL_X_LIST[index]
            y = GOAL_Y_LIST[index]
        elif self.train_state == 5:
            index = random.randrange(0, 10)
            x = GOAL_X_LIST_WORLD[index]
            y = GOAL_Y_LIST_WORLD[index]
        else:
            position_check = True
            while position_check:
                x = random.uniform(-1.5, 1.5)
                y = random.uniform(-1.5, 1.5)
                if abs(x - self.obstacle_1[0]) <= 0.4 and abs(y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_2[0]) <= 0.4 and abs(y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_3[0]) <= 0.4 and abs(y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_4[0]) <= 0.4 and abs(y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(x - 0.0) <= 0.4 and abs(y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False
                if abs(x - self.last_goal_x) < 1 and abs(y - self.last_goal_y) < 1:
                    position_check = True
        self.target[0], self.target[1] = [x, y]
        self.last_goal_x = x
        self.last_goal_y = y

        print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]
        self.ep_steps = 0

        self.deleteGoalModel()
        time.sleep(0.5)
        self.respawnGoalModel()

        return np.concatenate((np.array(obs), self.sector_scan))

    def get_distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    def get_heading(self, x1, x2):
        return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

    def get_reward(self):
        yaw_car = self.pose[2]
        head_to_target = self.get_heading(self.pose, self.target)

        alpha = head_to_target - yaw_car
        ld = self.get_distance(self.pose, self.target)
        crossTrackError = math.sin(alpha) * ld

        headingError = abs(alpha)
        alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])

        # return -1*(abs(crossTrackError)**2 + alongTrackError + 3*headingError/1.57)/6
        return -alongTrackError

    def check_goal(self):
        done = False
        reward = self.get_reward()

        if self.ep_steps > MAX_EP_LEN:
            print("Reached Max Episode Length!")
            reward -= 100
            done = True
            self.stop_bot()
        else:
            if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD):
                done = True
                reward += 1000
                print("Goal Reached!")
                self.stop_bot()
            else:
                if np.min(self.sector_scan) < OBS_THRESH:
                    print("Collision Detected!")
                    reward -= 500
                    self.stop_bot()
                    done = True

        return done, reward

    def step(self, action):
        reward = 0
        done = False
        info = {}
        self.ep_steps += 1

        self.action = [round(x, 2) for x in action]
        msg = Twist()
        msg.linear.x = self.action[0]
        msg.angular.z = self.action[1]
        self.pub.publish(msg)
        self.rate.sleep()

        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        done, reward = self.check_goal()

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]

        return np.concatenate((np.array(obs), self.sector_scan)), reward, done, info

    def stop_bot(self):
        msg = Twist()
        msg.linear.x = 0.
        msg.linear.y = 0.
        msg.linear.z = 0.
        msg.angular.x = 0.
        msg.angular.y = 0.
        msg.angular.z = 0.
        self.pub.publish(msg)
        self.rate.sleep()

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
            spawn_model_prox('goal', self.model, 'robots_name_space', self.goal_position, "world")
            break

    def deleteGoalModel(self):
        while self.check_model:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox('goal')
            break

class DiscreteTurtleObsGym(gym.Env):
    """
    Discrete Action Space Gym Environment w/ Obstacles
    State - relative x, y, theta, [sectorized lidar scan]
    Action - 4 actions - 
                0: [MAX_SPEED, 0.], 1: [0., MAX_STEER/2], 2: [-MAX_SPEED, 0.], 3: [0., -MAX_STEER/2]
             15 actions - 
                 0: [0., -MAX_STEER], 1: [0., -MAX_STEER/2], 2: [0., 0.], 3: [0., MAX_STEER/2], 4: [0., MAX_STEER],
                5: [MAX_SPEED/2, -MAX_STEER], 6: [MAX_SPEED/2, -MAX_STEER/2], 7: [MAX_SPEED/2, 0.], 8: [MAX_SPEED/2, MAX_STEER/2], 9: [MAX_SPEED/2, MAX_STEER],
                10: [MAX_SPEED, -MAX_STEER], 11: [MAX_SPEED, -MAX_STEER/2], 12: [MAX_SPEED, 0.], 13: [MAX_SPEED, MAX_STEER/2], 14: [MAX_SPEED, MAX_STEER]
    Reward - 
    """
    def __init__(self, n_actions = 'Minimum', train_state = 1):
        super(DiscreteTurtleObsGym,self).__init__()		
        metadata = {'render.modes': ['console']}
        print("Initialising Turtlebot 3 Discrete Obs Gym Environment...")
         
        self.actSpace = collections.defaultdict(list)
        if n_actions == 'Minimum':	
            self.action_space = spaces.Discrete(4)		
            self.actSpace = {
                0: [0., 0.], 1: [MAX_SPEED, 0.], 2: [0., MAX_STEER/2], 3: [0., -MAX_STEER/2]
            }
        elif n_actions == 'Maximum':
            self.action_space = spaces.Discrete(15)
            self.actSpace = {
                0: [0., -MAX_STEER], 1: [0., -MAX_STEER/2], 2: [0., 0.], 3: [0., MAX_STEER/2], 4: [0., MAX_STEER],
                5: [MAX_SPEED/2, -MAX_STEER], 6: [MAX_SPEED/2, -MAX_STEER/2], 7: [MAX_SPEED/2, 0.], 8: [MAX_SPEED/2, MAX_STEER/2], 9: [MAX_SPEED/2, MAX_STEER],
                10: [MAX_SPEED, -MAX_STEER], 11: [MAX_SPEED, -MAX_STEER/2], 12: [MAX_SPEED, 0.], 13: [MAX_SPEED, MAX_STEER/2], 14: [MAX_SPEED, MAX_STEER]
            }

        low = np.concatenate((np.array([-1., -1., -4.]), np.zeros(36)))
        high = np.concatenate((np.array([1., 1., 4.]), np.ones(36)*5.))
        self.observation_space = spaces.Box(low, high, dtype=np.float16)
        self.target = [0., 0., 0.]
        self.ep_steps = 0
        self.train_state = train_state

        self.pose = np.zeros(3) # pose_callback
        self.sector_scan = np.zeros(36) # scan_callback
        self.depth = np.zeros(1843200) # cam_callback. 
        # Depth image is compressed image of size 1280x720. data length = 1843200 (720rows x 2560step)
        
        self.action = [0., 0.] # Publisher is /cmd_vel Twist message for linear and angular velocity

        # Initialize ROS nodes
        self.sub_model = rospy.Subscriber('/gazebo/model_states', ModelStates, self.checkGoalModel)
        self.sub = [0, 0, 0]
        self.sub[0] = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        self.sub[1] = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # self.sub[2] = rospy.Subscriber("/camera/depth/image_raw", Image, self.cam_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        self.rate = rospy.Rate(10)

        # Gazebo Services
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.modelPath = rospkg.RosPack().get_path('turtlebot3_gazebo') + '/models/turtlebot3_square/goal_box/model.sdf'
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.check_model = False

        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = 0.0
        self.last_goal_y = 0.0

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
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.pause()
            self.reset_simulation_proxy()
            self.unpause()
            print('Simulation reset')
        except rospy.ServiceException as exc:
            print("Reset Service did not process request: " + str(exc))

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.train_state == 4:
            index = random.randrange(0, 13)
            x = GOAL_X_LIST[index]
            y = GOAL_Y_LIST[index]
        elif self.train_state == 5:
            index = random.randrange(0, 10)
            x = GOAL_X_LIST_WORLD[index]
            y = GOAL_Y_LIST_WORLD[index]
        else:
            position_check = True
            while position_check:
                x = random.uniform(-1.5, 1.5)
                y = random.uniform(-1.5, 1.5)
                if abs(x - self.obstacle_1[0]) <= 0.4 and abs(y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_2[0]) <= 0.4 and abs(y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_3[0]) <= 0.4 and abs(y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(x - self.obstacle_4[0]) <= 0.4 and abs(y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(x - 0.0) <= 0.4 and abs(y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False
                if abs(x - self.last_goal_x) < 1 and abs(y - self.last_goal_y) < 1:
                    position_check = True
        self.target[0], self.target[1] = [x, y]
        self.last_goal_x = x
        self.last_goal_y = y

        print("Reset target to : [{:.2f}, {:.2f}]".format(self.target[0], self.target[1]))
        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]
        self.ep_steps = 0

        self.deleteGoalModel()
        time.sleep(0.5)
        self.respawnGoalModel()

        return np.concatenate((np.array(obs), self.sector_scan))

    def get_distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    def get_heading(self, x1, x2):
        return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))

    def get_reward(self):
        yaw_car = self.pose[2]
        head_to_target = self.get_heading(self.pose, self.target)

        alpha = head_to_target - yaw_car
        ld = self.get_distance(self.pose, self.target)
        crossTrackError = math.sin(alpha) * ld

        headingError = abs(alpha)
        alongTrackError = abs(self.pose[0] - self.target[0]) + abs(self.pose[1] - self.target[1])

        # return -1*(abs(crossTrackError)**2 + alongTrackError + 5*headingError/1.57)/6
        return -alongTrackError

    def check_goal(self):
        done = False
        reward = self.get_reward()
        
        if self.ep_steps > MAX_EP_LEN:
            print("Reached Max Episode Length!")
            reward -= 100
            done = True
            self.stop_bot()
        else:
            if (abs(self.pose[0] - self.target[0]) < THRESHOLD and abs(self.pose[1] - self.target[1]) < THRESHOLD):
                done = True
                reward += 1000
                print("Goal Reached!")
                self.stop_bot()
            else:				
                if np.min(self.sector_scan) < OBS_THRESH:
                    print("Collision Detected!")
                    reward -= 500
                    self.stop_bot()
                    done = True

        return done, reward

    def step(self, discrete_action):
        reward = 0
        done = False
        info = {}
        self.ep_steps += 1

        action = self.actSpace[discrete_action]
        
        self.action = [round(x, 2) for x in action]
        msg = Twist()
        msg.linear.x = self.action[0]
        msg.angular.z = self.action[1]
        self.pub.publish(msg)
        self.rate.sleep()

        head_to_target = self.get_heading(self.pose, self.target)
        heading = head_to_target - self.pose[2]
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        done, reward = self.check_goal()

        obs = [(self.target[0] - self.pose[0]) / GRID, (self.target[1] - self.pose[1]) / GRID, heading]
        obs = [round(x, 2) for x in obs]

        return np.concatenate((np.array(obs), self.sector_scan)), reward, done, info

    def stop_bot(self):
        msg = Twist()
        msg.linear.x = 0.
        msg.linear.y = 0.
        msg.linear.z = 0.
        msg.angular.x = 0.
        msg.angular.y = 0.
        msg.angular.z = 0.
        self.pub.publish(msg)
        self.rate.sleep()

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
            spawn_model_prox('goal', self.model, 'robots_name_space', self.goal_position, "world")
            break

    def deleteGoalModel(self):
        while self.check_model:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox('goal')
            break