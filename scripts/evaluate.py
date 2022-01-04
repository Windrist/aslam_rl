#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import rospkg
from tb3env import ContinuousTurtleGym, DiscreteTurtleGym, ContinuousTurtleObsGym, DiscreteTurtleObsGym
from stable_baselines3 import DQN, DDPG, PPO, SAC, TD3


def evaluate(env, algorithm, device, env_name):
    rospack = rospkg.RosPack()
    modelPath = rospack.get_path('aslam_rl') + '/scripts/sbmodels/' + algorithm + '_' + env_name
    
    if algorithm == "DQN":
        model = DQN.load(modelPath, device=device)
    elif algorithm == "DDPG":
        model = DDPG.load(modelPath, device=device)
    elif algorithm == "PPO":
        model = PPO.load(modelPath, device=device)
    elif algorithm == "SAC":
        model = SAC.load(modelPath, device=device)
    elif algorithm == "TD3":
        model = TD3.load(modelPath, device=device)
    
    episode_rewards = []
    for _ in range(10):
        done = False
        obs = env.reset()
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            # print(obs)
            # rospy.sleep(0.05)
            ep_reward += reward
            if done:
                episode_rewards.append(ep_reward)
                continue
    mean_10ep_reward = round(np.mean(episode_rewards[-10:]), 1)
    print("Mean reward:", mean_10ep_reward, "Num episodes:", len(episode_rewards))


if __name__ == '__main__':
	try:
		rospy.init_node('evaluate', anonymous=True)

		# Get Parameters for Configurations
		state = rospy.get_param('~state', '1') # State Environtment for Training (Include: 1, 2, 3, 4, 'world')
		algorithm = rospy.get_param('~algorithm', 'DQN') # Algorithm for Training (Include: DQN, DDPG, PPO, SAC, TD3)
		env_name = rospy.get_param('~env_name', 'DiscreteObs') # Turtlebot Gazebo Gym Environment (Include: Continuous, Discrete, ContinuousObs, DiscreteObs)
		n_actions = rospy.get_param('~n_actions', 'Minimum') # Number of Discrete Actions (Include: Minimum or Maximum)
		device = rospy.get_param('~device', 'cpu') # Run on cpu or cuda (Include: cpu or cuda)
		
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
		
		evaluate(env, algorithm, device, env_name)
		
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
