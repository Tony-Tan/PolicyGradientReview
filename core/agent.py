import numpy as np
from collections import deque
import random
from core.exceptions import *


class Agent:
    def __init__(self, memory_size: int):
        self.memory = Memory(memory_size)
        pass

    def react(self, obs: np.ndarray, testing: bool = False) -> np.ndarray:
        """
        :param obs:
        :param testing:
        :return:
        """
        raise NotImplement

    def observe(self, obs: np.ndarray, action: np.ndarray, reward: int, terminated: bool,
                truncated: bool, info: dict, save_obs: bool = True):
        """
        :param obs:
        :param action:
        :param reward:
        :param terminated:
        :param truncated:
        :param info:
        :param save_obs:
        :return:
        """
        if save_obs:
            self.memory.append([obs, action, reward, terminated, truncated, info])

    def learn(self):
        raise NotImplement


class Memory:
    def __init__(self, max_memory_size):
        self.max_memory_size = max_memory_size
        self.memory_buffer = deque(maxlen=max_memory_size)
        self.current_memory_size = 0

    def append(self, new_data):
        self.memory_buffer.append(new_data)
        self.current_memory_size += 1

    def __getitem__(self, item):
        return self.memory_buffer[item]

    def __len__(self):
        return self.current_memory_size if self.current_memory_size < \
                                           self.max_memory_size else self.max_memory_size

    def sample(self, size: int):
        return random.sample(self.memory_buffer, size)
