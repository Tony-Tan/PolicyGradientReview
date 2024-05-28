import numpy as np
from abc_rl.experience_replay import *
from abc_rl.experience_replay import ExperienceReplay
import torch.multiprocessing as mp


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, capacity: int):
        super(UniformExperienceReplay,self).__init__(capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(np.arange(self.__len__()), batch_size,  replace=False)
        return self.get_items(idx)


# class UniformExperienceReplayMP(SharedExperience):
#     """
#     Shared memory implementation of the Uniform Experience Replay buffer.
#     """
#
#     def __init__(self, capacity: int, obs_shape: np.shape, action_shape: np.shape):
#         super(UniformExperienceReplayMP, self).__init__(capacity, obs_shape, action_shape)
#
#
#     def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
#               next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
#         super(UniformExperienceReplayMP, self).store(observation, action, reward, next_observation, done, truncated)
#
#     def sample(self, batch_size: int = 0 ):
#         return self.get_items()

class UniformExperienceReplayMP(ShareMemory):
    def __init__(self, capacity: int, manager: mp.Manager):
        super(UniformExperienceReplayMP, self).__init__(capacity, manager)

    def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
        super(UniformExperienceReplayMP, self).store(observation, action, reward, next_observation, done, truncated)

    def sample(self, batch_size: int = 0):
        with self.lock:
            transitions = self.get_all_items()
            obs = []
            action = []
            reward = []
            next_obs = []
            done = []
            truncated = []
            for item in transitions:
                obs.append(item[0])
                action.append(item[1])
                reward.append(item[2])
                next_obs.append(item[3])
                done.append(item[4])
                truncated.append(item[5])
            obs = np.array(obs,dtype=np.float32)
            action = np.array(action,dtype=np.float32)
            reward = np.array(reward,dtype=np.float32)
            next_obs = np.array(next_obs,dtype=np.float32)
            done = np.array(done,dtype=np.float32)
            truncated = np.array(truncated,dtype=np.float32)


        return (torch.from_numpy(obs),
                torch.from_numpy(action),
                torch.from_numpy(reward),
                torch.from_numpy(next_obs),
                torch.from_numpy(done),
                torch.from_numpy(truncated))

