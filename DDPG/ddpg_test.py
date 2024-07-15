#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : ddpg_test.py
# Time       ：2024/7/12 14:46
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import os

import gym
import numpy as np
import pygame
import torch

from DDPG.ddpg_agent import Actor


def process_frame(frame):
    frame = np.transpose(frame, (1, 0, 2))
    frame = pygame.surfarray.make_surface(frame)
    return pygame.transform.scale(frame, (width, height))


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Initialize environment
env = gym.make(id='Pendulum-v1', render_mode="rgb_array")
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# load parameters
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + "ddpg_actor_20240712144238.pth"

actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path))

# Initialize pygame
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Test game
NUM_EPISODE = 30
NUM_STEP = 200
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0

    for step_i in range(NUM_STEP):
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state, reward, done, truncation, info = env.step(action)
        state = next_state
        episode_reward += reward
        print(f"Step {step_i}: {action}")

        frame = env.render()
        frame = process_frame(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(60)  # FPS

    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")

pygame.quit()
env.close()
