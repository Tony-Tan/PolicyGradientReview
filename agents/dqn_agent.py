# ## DQN Agent for Atari Games
#
# the result of DQN agent on several atari games is shown below:
#
#
# ![](https://raw.githubusercontent.com/Tony-Tan/Reinforcement-Learning-Data/dev/figures/DQN%202015%20rewards.png)
# ![](https://raw.githubusercontent.com/Tony-Tan/Reinforcement-Learning-Data/dev/figures/DQN%202015%20q%20values.png)
#
# you can find the original data in the: [Reinforcement-Learning-Data]
# (https://github.com/Tony-Tan/Reinforcement-Learning-Data/tree/dev/exps/dqn)
#
# paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
import numpy as np
import torch.optim
import torch.nn.functional as F
from collections import deque
from abc_rl.agent import Agent
from models.dqn_networks import DQNAtari
from abc_rl.policy import *
from experience_replay.uniform_experience_replay import *
from abc_rl.perception_mapping import *
from abc_rl.reward_shaping import *
from exploration.epsilon_greedy import *


# Define the image normalization function
# the input of neural network is the uint8 matrix, so we need to normalize the image to [-0.5,0.5]
def image_normalization(image_uint8):
    """
    Normalize the image to [0，1]

    :param image_uint8: Input image tensor
    :return: Normalized image tensor
    """
    return image_uint8 / 255.0


# Define the DQN reward shaping class
# the reward of each step is clipped between $-1$ and $1$
class DQNAtariReward(RewardShaping):
    """
    Class for reward shaping in DQN for Atari games.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, reward):
        """
        Preprocess the reward to clip it to -1 and 1.
        1. positive reward is clipped to 1
        2. negative reward is clipped to -1
        3. leave 0 reward unchanged
        :param reward: Input reward
        :return: Clipped reward
        """
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0


class DQNPerceptionMapping(PerceptionMapping):
    def __init__(self, phi_channel: int, screen_size: int):
        super().__init__()
        self.phi = deque(maxlen=phi_channel)
        self.phi_channel = phi_channel
        self.screen_size = screen_size

    # Append the observation to the phi deque
    def __phi_append(self, obs: np.ndarray):
        """
        Append the observation to the phi deque.

        :param obs: Input observation
        """
        self.phi.append(obs)

    # Reset the phi to zero and reset the last_frame_pre_process
    def reset(self):
        """
        Reset the phi to zero and reset the last_frame_pre_process.
        """
        self.phi.clear()
        for i in range(self.phi_channel):
            self.phi.append(np.zeros([self.screen_size, self.screen_size]))

    # Preprocess the state to a certain size and load it to phi.
    def __call__(self, state: np.ndarray, step_i: int) -> np.ndarray:
        """

        :param state: Input state
        :param step_i: Step index
        :return: Processed state
        """
        if step_i == 0:
            self.reset()
        self.__phi_append(state)
        obs = np.array(self.phi, dtype=np.uint8)
        return obs


#  Class for value function in DQN for Atari games.
class DQNValueFunction(ValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, model_save_path: str, device: torch.device, logger: Logger, ):
        super(DQNValueFunction, self).__init__()
        self.logger = logger
        # Define the value neural network and the target value neural network
        self.value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn.eval()
        self.synchronize_value_nn()
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.update_step = 0
        self.model_save_path = model_save_path

    # synchronize the target value neural network with the value neural network
    def synchronize_value_nn(self):
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def max_state_value(self, obs_tensor):
        with torch.no_grad():
            obs_tensor = image_normalization(obs_tensor)
            outputs = self.target_value_nn(obs_tensor)
        msv, _ = torch.max(outputs, dim=1, keepdim=True)
        return msv

    # Update the value function with the given samples
    def update(self, samples: tuple, weight=None):

        """
        :param samples: Input samples
        :param weight: Importance weight for prioritized experience replay
        """
        obs_tensor = samples[0].to(self.device, non_blocking=True)
        action_tensor = samples[1].to(self.device, non_blocking=True)
        reward_tensor = samples[2].to(self.device, non_blocking=True)
        next_obs_tensor = samples[3].to(self.device, non_blocking=True)
        termination_tensor = samples[4].to(self.device, non_blocking=True)
        truncated_tensor = samples[5].to(self.device, non_blocking=True)

        # calculate the $q$ value of next state
        max_next_state_value = self.max_state_value(next_obs_tensor)
        reward_tensor.resize_as_(max_next_state_value)

        truncated_tensor.resize_as_(max_next_state_value)
        termination_tensor.resize_as_(max_next_state_value)
        # $y = r_{j} + \gamma  * max_{a'}Q(\phi_{j+1},a';\theta)$ for non-terminal state
        #
        # $y = r_{j}$ for terminal state
        # if truncated_tensor.any() == 1:
        #     print(truncated_tensor)
        q_value = reward_tensor + self.gamma * max_next_state_value * (1 - truncated_tensor) * (1 - termination_tensor)
        action_tensor.resize_as_(reward_tensor)
        q_value.resize_as_(reward_tensor)
        actions = action_tensor.long()
        self.optimizer.zero_grad()
        self.value_nn.train()
        # normalize the input image
        obs_tensor = image_normalization(obs_tensor)
        outputs = self.value_nn(obs_tensor)

        obs_action_value = outputs.gather(1, actions)
        # in [prioritized experience replay]() algorithm, weight is used to adjust the importance of the samples
        diff = obs_action_value - q_value
        if weight is not None:
            weight = torch.as_tensor(weight, device=self.device, dtype=torch.float32).resize_as_(diff)
            diff_clipped = torch.clip(diff, -1, 1) * weight
        else:
            diff_clipped = torch.clip(diff, -1, 1)
        loss = F.mse_loss(diff_clipped, torch.zeros_like(diff_clipped))
        loss.backward()
        self.optimizer.step()
        self.update_step += 1
        # return the clipped difference and the q value
        return loss.detach().cpu().numpy().astype(np.float32)

    # Calculate the value of the given phi tensor.
    def __call__(self, phi_tensor: torch.Tensor) -> np.ndarray:
        """

        :param phi_tensor: Input phi tensor
        :return: Value of the phi tensor
        """
        with torch.no_grad():
            if phi_tensor.dim() == 3:
                obs_input = phi_tensor.unsqueeze(0)
            else:
                obs_input = phi_tensor
            self.value_nn.eval()
            obs_input = image_normalization(obs_input)
            state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
            return state_action_values

    def save(self, model_label: str):
        torch.save(self.value_nn.state_dict(), os.path.join(self.model_save_path, f'{model_label}.pth'))

    def load(self, model_label: str):
        self.value_nn.load_state_dict(torch.load(model_label))



class DQNExperienceReplay(UniformExperienceReplay):
    def __init__(self, capacity: int, phi_channel: int):
        super(DQNExperienceReplay, self).__init__(capacity)
        self.phi_channel = phi_channel

    def get_items(self, idx):
        idx_size = len(idx)
        obs_shape = self.buffer[0][0].shape
        obs = np.zeros((idx_size, self.phi_channel, *obs_shape), dtype=np.float32)
        next_obs = np.zeros((idx_size, self.phi_channel, *obs_shape), dtype=np.float32)
        action = np.zeros(idx_size, dtype=np.float32)
        reward = np.zeros(idx_size, dtype=np.float32)
        done = np.zeros(idx_size, dtype=np.float32)
        truncated = np.zeros(idx_size, dtype=np.float32)

        for i, idx_i in enumerate(idx):
            buffer_slice = self.buffer[idx_i - self.phi_channel + 1:idx_i + 1]
            obs[i] = np.array([buf[0] for buf in buffer_slice], dtype=np.float32)
            next_obs[i] = np.array([buf[3] for buf in buffer_slice], dtype=np.float32)

            _, a, r, _, d, t = buffer_slice[-1]
            action[i] = a
            reward[i] = r
            done[i] = d
            truncated[i] = t

        obs = torch.from_numpy(obs)
        next_obs = torch.from_numpy(next_obs)
        action = torch.from_numpy(action)
        reward = torch.from_numpy(reward)
        done = torch.from_numpy(done)
        truncated = torch.from_numpy(truncated)

        return obs, action, reward, next_obs, done, truncated

    def sample(self, batch_size: int):
        idx = np.random.choice(np.arange(self.phi_channel, self.__len__()), batch_size, replace=False)
        return self.get_items(idx)


# DQN agent for Atari games
class DQNAgent(Agent):
    """
    Class for DQN agent for Atari games.
    """

    def __init__(self, screen_size: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, replay_start_size: int,
                 learning_rate: float, step_c: int, gamma: float, training_episodes: int, phi_channel: int,
                 epsilon_max: float, epsilon_min: float, exploration_steps: int, device: torch.device,
                 exp_path: str, exp_name: str, logger: Logger):
        super(DQNAgent, self).__init__(logger)
        self.action_dim = action_space.n
        self.value_function = DQNValueFunction(phi_channel, self.action_dim, learning_rate, gamma,
                                               os.path.join(exp_path, exp_name), device, logger)
        self.exploration_method = DecayingEpsilonGreedy(epsilon_max, epsilon_min, exploration_steps)
        self.memory = DQNExperienceReplay(replay_buffer_size, phi_channel)
        self.perception_mapping = DQNPerceptionMapping(phi_channel, screen_size)
        self.reward_shaping = DQNAtariReward()
        self.device = device
        self.mini_batch_size = mini_batch_size
        self.replay_start_size = replay_start_size
        self.training_episodes = training_episodes
        self.update_step = 0
        self.step_c = step_c

    # Select an action based on the given observation and exploration method.
    def select_action(self, obs: np.ndarray, exploration_method: Exploration = None) -> tuple:
        """

        :param obs: Input observation
        :param exploration_method: Exploration method
        :return: Selected action
        """
        value_array = None
        # if isinstance(exploration_method, RandomAction):
        #     return exploration_method(self.action_dim), value_array
        # else:
        phi_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        value_array = self.value_function(phi_tensor)[0]
        if exploration_method is None:
            return self.exploration_method(value_array), value_array
        else:
            return exploration_method(value_array), value_array

    # Store the agent environment interaction in the memory.
    def store(self, obs, action, reward, next_obs, done, truncated):
        """
        Store the given parameters in the memory.

        :param obs: Observation
        :param action: Action
        :param reward: Reward
        :param next_obs: Next observation
        :param done: Done flag
        :param truncated: Truncated flag
        """
        self.memory.store(obs, np.array(action), np.array(reward), next_obs, np.array(done), np.array(truncated))

    # Perform a training step if the memory size is larger than the update sample size.
    def train_one_step(self):

        """
        Perform a training step if the memory size is larger than the update sample size.
        """

        if len(self.memory) > self.replay_start_size:
            samples = self.memory.sample(self.mini_batch_size)
            loss = self.value_function.update(samples)
            self.update_step += 1
            # synchronize the target value neural network with the value neural network every step_c steps
            if self.update_step % self.step_c == 0:
                self.value_function.synchronize_value_nn()
                if self.logger:
                    self.logger.tb_scalar('loss', np.mean(loss), self.update_step)

    def save_model(self, model_label: str = 'last'):
        self.value_function.save(model_label)

    def load_model(self, model_path:str):
        self.value_function.load(model_path)