import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

env = gym.make("ALE/Tetris-v5")
env.reset()
next_state, reward, done, _, _ = env.step(1)

print(reward)