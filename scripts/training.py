#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import rospkg
from tb3env import ContinuousTurtleGym, DiscreteTurtleGym, ContinuousTurtleObsGym, DiscreteTurtleObsGym
from stable_baselines3 import DQN, DDPG, PPO, SAC, TD3
from stable_baselines3.common.logger import configure


# Train Function
def train(env, algorithm, num_steps, device, old_model, env_name):
	rospack = rospkg.RosPack()
	modelPath = rospack.get_path('aslam_rl') + '/scripts/sbmodels/' + algorithm  + '_' + env_name
	tensorlogPath = rospack.get_path('aslam_rl') + '/log/' + algorithm + '_' + env_name + '/'
	
	if not old_model:
		if algorithm == "DQN":
			model = DQN("MlpPolicy", env = env, device=device)
		elif algorithm == "DDPG":
			model = DDPG("MlpPolicy", env = env, device=device)
		elif algorithm == "PPO":
			model = PPO("MlpPolicy", env = env, device=device)
		elif algorithm == "SAC":
			model = SAC("MlpPolicy", env = env, device=device)
		elif algorithm == "TD3":
			model = TD3("MlpPolicy", env = env, device=device)

		model.set_logger(configure(tensorlogPath + algorithm + '_PreTrained', ["stdout", "csv", "log", "tensorboard", "json"]))
	else:
		if algorithm == "DQN":
			model = DQN.load(modelPath, env = env, device=device)
		elif algorithm == "DDPG":
			model = DDPG.load(modelPath, env = env, device=device)
		elif algorithm == "PPO":
			model = PPO.load(modelPath, env = env, device=device)
		elif algorithm == "SAC":
			model = SAC.load(modelPath, env = env, device=device)
		elif algorithm == "TD3":
			model = TD3.load(modelPath, env = env, device=device)
		
		logPath = algorithm +  '_Extended_' + str(len(os.listdir(tensorlogPath)) - 1)
		model.set_logger(configure(tensorlogPath + logPath, ["stdout", "csv", "log", "tensorboard", "json"]))
	
	model.learn(total_timesteps = num_steps)
	model.save(modelPath)
	env.stop_bot()
	rospy.signal_shutdown('Training Completed! Shutdown ROS!')


if __name__ == '__main__':
	try:
		rospy.init_node('train', anonymous=True)

		# Get Parameters for Configurations
		state = rospy.get_param('~state', '1') # State Environtment for Training (Include: 1, 2, 3, 4, 'world')
		algorithm = rospy.get_param('~algorithm', 'DQN') # Algorithm for Training (Include: DQN, DDPG, PPO, SAC, TD3)
		env_name = rospy.get_param('~env_name', 'DiscreteObs') # Turtlebot Gazebo Gym Environment (Include: Continuous, Discrete, ContinuousObs, DiscreteObs)
		n_actions = rospy.get_param('~n_actions', 'Minimum') # Number of Discrete Actions (Include: Minimum or Maximum)
		num_steps = rospy.get_param('~num_steps', 500000) # Total of Time Steps to Train
		device = rospy.get_param('~device', 'cpu') # Run on cpu or cuda (Include: cpu or cuda)
		old_model = rospy.get_param('~old_model', False) # Training based on PreTrained Model (Include: True or False)
		
		if state == 'world':
			state = 5
		if env_name == "Continuous":
			env =  ContinuousTurtleGym(state)
		elif env_name == "Discrete":
			env = DiscreteTurtleGym(n_actions, state)
		elif env_name == "ContinuousObs":
			env =  ContinuousTurtleObsGym(state)
		elif env_name == "DiscreteObs":
			env = DiscreteTurtleObsGym(n_actions, state)
		
		train(env, algorithm, num_steps, device, old_model, env_name)
		
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
