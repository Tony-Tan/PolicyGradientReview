import gc
from agents.dqn_agent import *


class ProportionalPrioritization(DQNExperienceReplay):
    def __init__(self, capacity: int, phi_channel: int, alpha: float = 0.6, beta: float = 0.4):
        super(ProportionalPrioritization, self).__init__(capacity, phi_channel)
        self.p = np.zeros(capacity)
        self.alpha = alpha
        self.beta = beta

    def store(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray,
              next_observation: np.ndarray, done: np.ndarray, truncated: np.ndarray):
        super(ProportionalPrioritization, self).store(observation, action, reward, next_observation, done, truncated)
        p_position = self.position - 1
        self.p[p_position] = np.max(self.p) if self.__len__() > 1 else 1.0

    def sample(self, batch_size: int ):
        n = self.__len__()
        # normalize the p as a probability distribution
        p_alpha = self.p[:n]**self.alpha
        p = p_alpha / np.sum(p_alpha)
        # select the index of the samples
        idx = np.random.choice(np.arange(n), batch_size, p=p, replace=False)
        w = (n * p) ** -self.beta
        w = w[idx] / np.max(w)
        return self.get_items(idx), w, idx


class DQNPPAgent(DQNAgent):
    def __init__(self, screen_size: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, replay_start_size: int,
                 learning_rate: float, step_c: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, exp_path: str, exp_name: str, logger: Logger,
                 alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize the DQN agent with proportional prioritization.

        :param input_frame_width: The width of the input frame
        :param input_frame_height: The height of the input frame
        :param action_space: The action space of the environment
        :param mini_batch_size: The mini batch size
        :param replay_buffer_size: The size of the replay buffer
        :param replay_start_size: The start size of the replay buffer
        :param learning_rate: The learning rate
        :param step_c: The number of steps to update the target network
        :param agent_saving_period: The period of saving the agent
        :param gamma: The discount factor
        :param training_steps: The number of training steps
        :param phi_channel: The number of channels in the input frame
        :param epsilon_max: The maximum epsilon value
        :param epsilon_min: The minimum epsilon value
        :param exploration_steps: The number of exploration steps
        :param device: The device for training
        :param logger: The logger for recording training information
        :param alpha: The alpha value for proportional prioritization
        :param beta: The beta value for proportional prioritization
        """
        super(DQNPPAgent, self).__init__(screen_size, action_space, mini_batch_size,
                                         replay_buffer_size, replay_start_size, learning_rate, step_c,
                                         gamma, training_episodes, phi_channel, epsilon_max,
                                         epsilon_min, exploration_steps, device, exp_path, exp_name, logger)
        del self.memory
        gc.collect()
        self.memory = ProportionalPrioritization(replay_buffer_size,phi_channel, alpha, beta)

    def train_one_step(self):
        """
        Perform a training step if the memory size is larger than the update sample size.
        """
        if len(self.memory) > self.replay_start_size:
            samples, w, idx = self.memory.sample(self.mini_batch_size)
            loss = self.value_function.update(samples, w)
            self.memory.p[idx] = loss.reshape(1, -1) + np.float32(1e-5)
            self.update_step += 1
            self.loss_log.append(np.mean(loss))
            if self.update_step % self.step_c == 0:
                self.value_function.synchronize_value_nn()
                if self.logger:
                    self.logger.tb_scalar('loss', np.mean(np.array(self.loss_log)), self.update_step)
                    self.loss_log = []
