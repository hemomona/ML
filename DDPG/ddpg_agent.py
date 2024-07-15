#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : ddpg_agent.py
# Time       ：2024/7/12 12:49
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import random

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from torch import optim

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# Hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 100000  # 1e5 is float, can not be used to initialize deque
BATCH_SIZE = 64
TAU = 5e-3


class Actor(nn.Module):  # == Policy Net (bs, state_dim) -> (bs, action_dim)
    def __init__(self, state_dim, action_dim, hidden_dim=64, action_bound=2):
        super(Actor, self).__init__()
        self.action_bound = action_bound  # the boundary of action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # tanh returns (-1, 1)
        x = x * self.action_bound  # scale to (-2, 2), which is action uniform
        return x


class Critic(nn.Module):  # QValue Net (bs, state_dim + action_dim) -> (bs, 1)
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Q(s, a) outputs 1 value

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_memo(self, state, action, reward, next_state, done):
        # expand_dims will add 1 at the position of axis (e.g. 0 for the following) in its dims,
        # which means dimension would plus 1
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # tuple: 64 -> array (64, 3)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound=2):
        # Randomly initialize critic network Q(s, a|θ^Q) and actor µ(s|θ^µ) with weights θ^Q and θ^µ.
        # Initialize target network Q' and µ' with weights θ^Q' ← θ^Q, θµ' ← θ^µ
        self.actor = Actor(state_dim, action_dim, action_bound=action_bound).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bound=action_bound).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # '_IncompatibleKeys' object has no attribute 'to'
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)  # 'Adam' object has no attribute 'to'

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Initialize replay buffer R
        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # expand dims from 1D to 2D
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)  # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # update critic
        next_actions = self.actor_target(next_states)  # next action
        next_Q = self.critic_target(next_states, next_actions.detach())  # next value of next action
        target_Q = rewards + GAMMA * next_Q * (1 - dones)  # target value of current action
        current_Q = self.critic(states, actions)  # predicted value of current action
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_Qs = self.actor(states)  # predicted value of actions in current state
        actor_current_Q = self.critic(states, actor_Qs)  # predicted value of the action in current state
        actor_loss = - torch.mean(actor_current_Q)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks of critic and actor
        for target_param, param, in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param, in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
