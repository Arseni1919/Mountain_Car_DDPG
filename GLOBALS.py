# ------------------------------------------- #
# ------------------IMPORTS:----------------- #
# ------------------------------------------- #
import os
import time
import logging
from collections import namedtuple, deque
from termcolor import colored
import random
from math import log
import gym
import pettingzoo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly
import plotly.express as px
import neptune.new as neptune
from neptune.new.types import File
from dotenv import load_dotenv

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms
from torch.distributions import Normal
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
load_dotenv()

# ------------------------------------------- #
# ------------------FOR ENV:----------------- #
# ------------------------------------------- #
SINGLE_AGENT_ENV_NAME = "MountainCarContinuous-v0"
# SINGLE_AGENT_ENV_NAME = "CartPole-v1"
# SINGLE_AGENT_ENV_NAME = 'LunarLanderContinuous-v2'
# SINGLE_AGENT_ENV_NAME = "BipedalWalker-v3"
from pettingzoo.mpe import simple_spread_v2
MAX_CYCLES = 25
# MAX_CYCLES = 75
# NUMBER_OF_AGENTS = 3
NUMBER_OF_AGENTS = 1
# ENV = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)
# ENV = simple_spread_v2.parallel_env(N=NUMBER_OF_AGENTS, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)

NUMBER_OF_GAMES = 10
SAVE_RESULTS = True
# SAVE_RESULTS = False
SAVE_PATH = 'data'

NEPTUNE = True
# NEPTUNE = False
PLOT_LIVE = True
# PLOT_LIVE = False
RENDER_WHILE_TRAINING = False

# ------------------------------------------- #
# ------------------FOR ALG:----------------- #
# ------------------------------------------- #

# MAX_LENGTH_OF_A_GAME = 10000
# ENTROPY_BETA = 0.001
# REWARD_STEPS = 4
# CLIP_GRAD = 0.1

BATCH_SIZE = 64  # size of the batches
REPLAY_BUFFER_SIZE = BATCH_SIZE * 157
WARMUP = BATCH_SIZE * 3
N_STEPS = REPLAY_BUFFER_SIZE + 10000
# N_EPISODES = 120
N_EPISODES = 70
LR_CRITIC = 1e-4  # learning rate
LR_ACTOR = 1e-4  # learning rate
GAMMA = 0.99  # discount factor
EPSILON = 0.00
SIGMA = 0.4
POLYAK = 0.99
TAU = 0.001
VAL_EVERY = 2000
TRAIN_EVERY = 100
HIDDEN_SIZE = 64
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'new_state'])