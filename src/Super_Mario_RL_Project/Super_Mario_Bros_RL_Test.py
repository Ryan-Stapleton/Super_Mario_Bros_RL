import retro
import gym
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random

env = retro.make(game="SuperMarioBros-Nes")
env = env.unwrapped
env.reset()

# class Network(torch.nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.input_shape = env.observation_space.shape
#         self.action_space = env.action_space.n

#         # build an MLP with 2 hidden layers
#         self.layers = torch.nn.Sequential(
#             torch.nn.Linear(*self.input_shape, 128),   # input layer
#             torch.nn.ReLU(),     # this is called an activation function
#             torch.nn.Linear(128, 128),    # hidden layer
#             torch.nn.ReLU(),     # this is called an activation function
#             torch.nn.Linear(128, self.action_space)    # output layer
#             )

#     def forward(self, x):
#         return self.layers(x)

# # Instantiate the network
# q_table = Network(env)

# # Load the state dictionary
# q_table.load_state_dict(torch.load("src/Super_Mario_RL_Project/q_table.pth"))
# q_table.eval()

# # Number of test episodes
# EPISODES = 10 
# success_count = 0

for i in range(5000):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)
    if done:
        break

# for episode in range(EPISODES):
#     state, _ = env.reset()
#     state = torch.tensor(state, dtype=torch.float32)
#     while True:
#         with torch.no_grad():
#             q_values = q_table(state)
#         action = torch.argmax(q_values).item() # Select action with highest predicted q-value
#         state, reward, terminated, truncated, info = env.step(action)
#         state = torch.tensor(state, dtype=torch.float32)
    
#         if reward >= 40:  # Success condition
#             # print(f"The Goal Has Been Reached In Test Episode {episode + 1}")
#             success_count += 1
#             break
    
#         done = terminated or truncated
#         if done:
#             break
    
# print(f"Average Test Policy Success Rate | Success Rate: ", (success_count)/EPISODES * 100,"%")

env.close()