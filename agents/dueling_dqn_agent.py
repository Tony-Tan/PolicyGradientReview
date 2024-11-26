from agents.dqn_agent import *
from agents.double_dqn_agent import *
from models.dqn_networks import DuelingDQNAtari


class DuelingDQNValueFunction(DQNValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, model_saving_path: str, device: torch.device, logger: Logger):
        super(DuelingDQNValueFunction, self).__init__(input_channel, action_dim, learning_rate,
                                                      gamma, model_saving_path, device, logger)
        self.value_nn = DuelingDQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn = DuelingDQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn.eval()
        self.synchronize_value_nn()
        self.optimizer = torch.optim.RMSprop(self.value_nn.parameters(),
                                             lr=learning_rate,
                                             alpha=0.95,  # squared gradient momentum
                                             momentum=0,  # gradient momentum
                                             eps=0.01)  # minimum squared gradient


class DuelingDQNAgent(DQNAgent):
    def __init__(self, screen_size: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, replay_start_size: int,
                 learning_rate: float, step_c: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, exp_path: str, exp_name: str, logger: Logger):
        super(DuelingDQNAgent, self).__init__(screen_size, action_space, mini_batch_size,
                                              replay_buffer_size, replay_start_size, learning_rate, step_c,
                                              gamma, training_episodes, phi_channel, epsilon_max,
                                              epsilon_min, exploration_steps, device, exp_path, exp_name, logger)
        self.value_function = DuelingDQNValueFunction(phi_channel, action_space.n, learning_rate,
                                                      gamma, os.path.join(exp_path, exp_name), device, logger)


class DuelingDDQNValueFunction(DoubleDQNValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, model_saving_path: str, device: torch.device, logger: Logger):
        super(DuelingDDQNValueFunction, self).__init__(input_channel, action_dim, learning_rate,
                                                       gamma, model_saving_path, device, logger)
        self.value_nn = DuelingDQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn = DuelingDQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn.eval()
        self.synchronize_value_nn()
        self.optimizer = torch.optim.RMSprop(self.value_nn.parameters(),
                                             lr=learning_rate,
                                             alpha=0.95,  # squared gradient momentum
                                             momentum=0,  # gradient momentum
                                             eps=0.01)  # minimum squared gradient

class DuelingDDQNAgent(DoubleDQNAgent):
    def __init__(self, screen_size: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, replay_start_size: int,
                 learning_rate: float, step_c: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, exp_path: str, exp_name: str, logger: Logger):
        super(DuelingDDQNAgent, self).__init__(screen_size, action_space, mini_batch_size,
                                               replay_buffer_size, replay_start_size, learning_rate, step_c,
                                               gamma, training_episodes, phi_channel, epsilon_max,
                                               epsilon_min, exploration_steps, device, exp_path, exp_name, logger)
        self.value_function = DuelingDQNValueFunction(phi_channel, action_space.n, learning_rate,
                                                      gamma, os.path.join(exp_path, exp_name), device, logger)
