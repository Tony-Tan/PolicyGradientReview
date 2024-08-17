from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.multiprocessing as mp


def cvt2tensor(obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
               next_obs: np.ndarray, done: np.ndarray, trunc: np.ndarray):
    obs = torch.from_numpy(obs)
    next_obs = torch.from_numpy(next_obs)
    action = torch.from_numpy(action)
    reward = torch.from_numpy(reward)
    done = torch.from_numpy(done)
    trunc = torch.from_numpy(trunc)
    return obs, action, reward, next_obs, done, trunc


class ExperienceReplay(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # This will keep track of the next position to insert into, for overwriting old records

    def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
        if len(self.buffer) < self.capacity:
            # If there is still space in the buffer, append the data
            self.buffer.append([observation, action, reward, next_observation, done, truncated])
        else:
            # Overwrite the oldest data if the buffer is at capacity
            self.buffer[self.position] = [observation, action, reward, next_observation, done, truncated]

        # Update the position in a circular manner
        self.position = (self.position + 1) % self.capacity

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def get_items(self, idx):
        idx_size = len(idx)
        obs = np.empty([idx_size, *self.buffer[0][0].shape], dtype=np.float32)
        action = np.empty([idx_size, *self.buffer[0][1].shape], dtype=np.float32)
        reward = np.empty([idx_size, *self.buffer[0][2].shape], dtype=np.float32)
        next_obs = np.empty([idx_size, *self.buffer[0][0].shape], dtype=np.float32)
        done = np.empty([idx_size, 1], dtype=np.float32)
        truncated = np.empty([idx_size, 1], dtype=np.float32)
        for i, idx_i in enumerate(idx):
            o, a, r, n, d, t = self.buffer[idx_i]
            # the observation is store from 0 to idx_size in obs_next_obs
            obs[i] = o
            # the next observation is store from idx_size to -1 in obs_next_obs
            next_obs[i] = n
            action[i] = a
            reward[i] = r
            # the truncated is store from 0 to idx_size in done_truncated
            done[i] = d
            # the truncated is store from idx_size to -1 in done_truncated
            truncated[i] = t
        return cvt2tensor(obs, action, reward, next_obs, done, truncated)

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


# class SharedExperience(ABC):
#     """
#     Shared memory implementation of the Uniform Experience Replay buffer.
#     """
#
#     def __init__(self, capacity: int, obs_shape: tuple, action_shape: tuple):
#         self.capacity = capacity
#         self.obs_shape = obs_shape
#         self.obs_dim = int(np.prod(obs_shape))
#         self.action_shape = action_shape
#         self.action_dim = int(np.prod(action_shape))
#         self.state_buffer = mp.Array('f', capacity * self.obs_dim)
#         self.next_state_buffer = mp.Array('f', capacity * self.obs_dim)
#         self.action_buffer = mp.Array('f', capacity * self.action_dim)
#         self.reward_buffer = mp.Array('f', capacity)
#         self.done_buffer = mp.Array('f', capacity)
#         self.truncated_buffer = mp.Array('f', capacity)
#         self.ptr = mp.Value('i', 0)
#         self.size = mp.Value('i', 0)
#         self.lock = mp.Lock()
#
#     @staticmethod
#     def _store_batch_in_shared_array(shared_array, data, idx, dim):
#         flat_data = data.flatten()
#         shared_array[idx * dim: (idx + 1) * dim] = flat_data
#
#     def store(self, observation, action, reward, next_observation, done, truncated):
#         with self.lock:
#             index = self.ptr.value % self.capacity
#             self._store_batch_in_shared_array(self.state_buffer, observation.flatten(), index, self.obs_dim)
#             self._store_batch_in_shared_array(self.next_state_buffer, next_observation.flatten(), index, self.obs_dim)
#             # self._store_batch_in_shared_array(self.action_buffer, action, index, self.action_dim)
#             self.action_buffer[index] = action
#             self.reward_buffer[index] = reward
#             self.done_buffer[index] = done
#             self.truncated_buffer[index] = truncated
#             self.ptr.value += 1
#             self.size.value = min(self.size.value + 1, self.capacity)
#
#     def clear(self):
#         with self.lock:
#             self.ptr.value = 0
#             self.size.value = 0
#
#     def __len__(self):
#         return self.size.value
#
#     def get_items(self):
#         with self.lock:
#             states = np.frombuffer(self.state_buffer.get_obj(), dtype=np.float32).reshape(self.capacity,
#                                                                                           *self.obs_shape)
#             next_states = np.frombuffer(self.next_state_buffer.get_obj(), dtype=np.float32).reshape(self.capacity,
#                                                                                                     *self.obs_shape)
#             # actions = np.frombuffer(self.action_buffer.get_obj(), dtype=np.float32).reshape(self.capacity, *self.action_shape)
#             actions = np.frombuffer(self.action_buffer.get_obj(), dtype=np.float32)
#             rewards = np.frombuffer(self.reward_buffer.get_obj(), dtype=np.float32)
#             dones = np.frombuffer(self.done_buffer.get_obj(), dtype=np.float32)
#             truncated = np.frombuffer(self.truncated_buffer.get_obj(), dtype=np.float32)
#
#         return (
#             torch.as_tensor(states),
#             torch.as_tensor(actions),
#             torch.as_tensor(rewards),
#             torch.as_tensor(next_states),
#             torch.as_tensor(dones),
#             torch.as_tensor(truncated)
#         )
#
#     @abstractmethod
#     def sample(self, *args, **kwargs):
#         pass


class ShareMemory(ABC):
    def __init__(self, capacity: int, manager: mp.Manager):
        self.capacity = capacity
        self.buffer = manager.list()
        self.lock = manager.Lock()  # 添加一个锁

    def store(self, observation, action, reward, next_observation, done, truncated, rank: int = None):
        if not rank:
            with self.lock:
                self.buffer.append([observation, action, reward, next_observation, done, truncated])
                if len(self.buffer) > self.capacity:
                    self.buffer.pop(0)
        else:
            self.buffer[rank] = [observation, action, reward, next_observation, done, truncated]

    def init_memory_with_None(self):
        for i in range(self.capacity):
            self.buffer.append(None)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def get_all_items(self):
        return self.buffer

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
