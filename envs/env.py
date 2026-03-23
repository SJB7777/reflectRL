import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FittingEnv(gym.Env):
    def __init__(self):
        super().__init__()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # return observation, info

    def step(self, action):
        
        # return observation, reward, terminated, truncated, info

    def render(self):
        pass

