from agents.dqn_agent import *
import gc


class DoubleDQNValueFunction(DQNValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float,   model_save_path: str, device: torch.device, logger: Logger):
        super(DoubleDQNValueFunction, self).__init__(input_channel, action_dim, learning_rate,
                                                     gamma,  model_save_path, device, logger)

    def max_state_value(self, obs_tensor):
        with torch.no_grad():
            obs_tensor = image_normalization(obs_tensor)
            outputs_tnn = self.target_value_nn(obs_tensor)
            outputs_nn = self.value_nn(obs_tensor)
        _, greedy_actions = torch.max(outputs_nn, dim=1, keepdim=True)
        msv = outputs_tnn.gather(1, greedy_actions)
        return msv


class DoubleDQNAgent(DQNAgent):
    def __init__(self, screen_size: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, replay_start_size: int,
                 learning_rate: float, step_c: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, exp_path: str, exp_name: str, logger: Logger):
        super(DoubleDQNAgent, self).__init__(screen_size, action_space, mini_batch_size,
                                             replay_buffer_size, replay_start_size, learning_rate, step_c,
                                             gamma, training_episodes, phi_channel, epsilon_max,
                                             epsilon_min, exploration_steps, device, exp_path, exp_name, logger)
        # delete the value function to save memory
        del self.value_function
        gc.collect()
        # create the value function for double dqn agent
        self.value_function = DoubleDQNValueFunction(phi_channel, action_space.n, learning_rate,
                                                     gamma,  os.path.join(exp_path, exp_name), device, logger)
