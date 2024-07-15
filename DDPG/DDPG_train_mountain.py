#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : DDPG_train_mountain.py
# Time       ：2024/7/15 10:43
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
https://blog.csdn.net/dgvv4/article/details/129479878
"""
import os
import random
import time

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from DDPG.ddpg_agent import DDPGAgent

# Initialize environment
env = gym.make(id='MountainCarContinuous-v0', render_mode="human")
STATE_DIM = env.observation_space.shape[0]  # 2: X & speed
ACTION_DIM = env.action_space.shape[0]  # 1
action_bound = env.action_space.high[0]  # max of action is 1

agent = DDPGAgent(STATE_DIM, ACTION_DIM, action_bound=action_bound)

# Hyperparameters
NUM_EPISODE = 50
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000  # 1/2 * NUM_EPISODE * NUM_STEP

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
REWARD_BUFFER_MEAN = []  # for smoother line in tha figure
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # initialize a random process epsilon for action exploration, linear interpretation
        epsilon = np.interp(x=episode_i * NUM_STEP + step_i, xp=[0, EPSILON_DECAY], fp=[EPSILON_START, EPSILON_END])

        # Select action a_t = µ(s_t|θ^µ) + N_t according to the current policy, this is why it's called "deterministic'
        random_sample = random.random()
        if random_sample <= epsilon:
            action = np.random.uniform(low=-1, high=1, size=ACTION_DIM)
        else:
            action = agent.get_action(state)

        # Execute action a_t and observe reward r_t and observe new state s_(t+1)
        next_state, reward, done, truncation, info = env.step(action)

        # Store transition (s_t, a_t, r_t, s_t+1) in R
        agent.replay_buffer.add_memo(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Sample a random minibatch of N transitions (s_i, a_i, r_i, s_(i+1)) from R to Update the target networks
        agent.update()

    REWARD_BUFFER[episode_i] = episode_reward
    REWARD_BUFFER_MEAN.append(np.mean(REWARD_BUFFER[-5:]))
    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")

# save models
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d%H%M%S")
torch.save(agent.actor.state_dict(), model + f"mountaincar_actor_{timestamp}.pth")
torch.save(agent.critic.state_dict(), model + f"mountaincar_critic_{timestamp}.pth")

env.close()

x_range = list(range(len(REWARD_BUFFER)))
plt.subplot(121)
plt.plot(x_range, REWARD_BUFFER)
plt.xlabel('episode')
plt.ylabel('return')
plt.subplot(122)
plt.plot(x_range, REWARD_BUFFER_MEAN)
plt.xlabel('episode')
plt.ylabel('mean_return')
plt.show()
