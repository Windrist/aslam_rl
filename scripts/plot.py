#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospkg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ALGORITHM = 'PPO'
ENV_NAME = 'ContinuousObs'

rospack = rospkg.RosPack()
logFullPath = rospack.get_path('aslam_rl') + '/Backup/V3/log/'
logPath = rospack.get_path('aslam_rl') + '/log/'

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, algorithm='DQN', env_name='DiscreteObs', title='Training Result Smoothed |'):
    """
    Plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    
    log_folder += algorithm + '_' + env_name + '/'

    dfp = []
    dfp.append(pd.read_csv(log_folder + algorithm + '_PreTrained/progress.csv'))
    for i in range(len(os.listdir(log_folder)) - 1):
        dfp.append(pd.read_csv(log_folder + algorithm + '_Extended' + '_' + str(i) + '/progress.csv'))

    for i in range(1, len(os.listdir(log_folder))):
        dfp[i]['time/total_timesteps'] += dfp[i-1]['time/total_timesteps'].iloc[-1]
    for i in range(len(os.listdir(log_folder))):
        dfp[i]['time/total_timesteps'] /= 1000
        dfp[i]['time/total_timesteps'].astype('int')
    df = pd.concat(dfp, ignore_index=True)

    timesteps = df['time/total_timesteps'].values
    rewards = df['rollout/ep_rew_mean'].values
    lendrive = df['rollout/ep_len_mean'].values

    rewards = moving_average(rewards, window=50)
    lendrive = moving_average(lendrive, window=50)
    # Truncate x
    timesteps = timesteps[len(timesteps) - len(rewards):]

    plt.figure(1)
    plt.plot(timesteps, rewards)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title + ' Rewards')

    plt.figure(2)
    plt.plot(timesteps, lendrive)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Steps to reach Goal')
    plt.title(title + ' Steps to reach Goal')

    plt.show()

def plot_all(log_folder, title='Training Result Smoothed |'):
    """
    Plot All results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    
    config = {'algorithm': ['PPO', 'DQN', 'SAC'], 'env_name': ['DiscreteObs', 'ContinuousObs']}
    full_df = {'PPO': {'DiscreteObs': pd.DataFrame(), 'ContinuousObs': pd.DataFrame()},
               'DQN': {'DiscreteObs': pd.DataFrame()},
               'SAC': {'ContinuousObs': pd.DataFrame()}}

    for algorithm in config['algorithm']:
        for env_name in config['env_name']:
            if algorithm == 'DQN' and env_name == 'ContinuousObs':
                continue
            elif algorithm == 'SAC' and env_name == 'DiscreteObs':
                continue
            else:
                temp_folder = log_folder + algorithm + '_' + env_name + '/'
                dfp = []
                dfp.append(pd.read_csv(temp_folder + algorithm + '_PreTrained/progress.csv'))
                for i in range(len(os.listdir(temp_folder)) - 1):
                    dfp.append(pd.read_csv(temp_folder + algorithm + '_Extended' + '_' + str(i) + '/progress.csv'))

                for i in range(1, len(os.listdir(temp_folder))):
                    dfp[i]['time/total_timesteps'] += dfp[i-1]['time/total_timesteps'].iloc[-1]
                for i in range(len(os.listdir(temp_folder))):
                    dfp[i]['time/total_timesteps'] /= 1000
                    dfp[i]['time/total_timesteps'].astype('int')
                df = pd.concat(dfp, ignore_index=True)
                full_df[algorithm][env_name] = df

    plt.figure(1)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')
    plt.title(title + ' Rewards')
    plt.figure(2)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Steps to reach Goal')
    plt.title(title + ' Steps to reach Goal')
    
    for algorithm in config['algorithm']:
        for env_name in config['env_name']:
            if algorithm == 'DQN' and env_name == 'ContinuousObs':
                continue
            elif algorithm == 'SAC' and env_name == 'DiscreteObs':
                continue
            else:
                timesteps = full_df[algorithm][env_name]['time/total_timesteps'].values
                rewards = full_df[algorithm][env_name]['rollout/ep_rew_mean'].values
                lendrive = full_df[algorithm][env_name]['rollout/ep_len_mean'].values

                rewards = moving_average(rewards, window=50)
                lendrive = moving_average(lendrive, window=50)
                # Truncate x
                timesteps = timesteps[len(timesteps) - len(rewards):]
                
                plt.figure(1)
                plt.plot(timesteps, rewards, label=algorithm + '_' + env_name)
                plt.legend()
                plt.figure(2)
                plt.plot(timesteps, lendrive, label=algorithm + '_' + env_name)
                plt.legend()
    
    plt.show()


if __name__ == '__main__':
    plot_results(logPath, ALGORITHM, ENV_NAME, title='Training Result Smoothed |')
    # plot_all(logFullPath, title='Training Result Smoothed |')