import copy
import random

import cv2

from agents.dqn_agent import *
from abc_rl.experience_replay import *
from abc_rl.exploration import *
from utils.configurator import *
from environments.env_atari import *


class DQNPlayGround:
    def __init__(self, agent: DQNAgent, env: AtariEnv, cfg: Configurator, logger: Logger):
        self.agent = agent
        self.env = env
        self.cfg = cfg
        self.logger = logger
        self.metric = {'last reward': 0., 'last steps': 0., 'max reward': 0., 'max steps': 0.}
        self.held_out_obs = []

    def __del__(self):
        self.logger.tb_hparams(self.cfg.config, self.metric)

    def held_out_states_gen(self):
        """
        Generate held out states for testing the agent.
        """
        step_i = 0
        state, _ = self.env.reset()
        while len(self.held_out_obs) < self.cfg['held_out_states_num']:
            action = self.env.action_space.sample()  # 随机动作
            next_state, r, d, t, _ = self.env.step(action)
            obs = self.agent.perception_mapping(state, step_i)
            if np.random.rand() < 0.01:  # 1%的概率保留当前状态
                self.held_out_obs.append(obs)
            state = next_state
            step_i += 1
            if d:
                state, _ = self.env.reset()
                step_i = 0
        self.held_out_obs = np.array(self.held_out_obs, dtype=np.float32)

    def validate_q(self):
        """
        Validate the q values of the agent.
        """
        max_q_array = None
        with torch.no_grad():
            max_q_array_ = np.max(self.agent.value_function(
                torch.as_tensor(self.held_out_obs, dtype=torch.float32).to(self.agent.device)), axis=1)
        return np.mean(np.array(max_q_array_))

    def train(self):
        # training
        self.held_out_states_gen()
        epoch_i = 0
        training_steps = 0
        # record
        best_reward = -np.inf
        while training_steps < self.cfg['training_steps']:
            step_i = reward_cumulated = 0
            state, info = self.env.reset()
            run_test = False
            game_over = False
            while not game_over:
                # perception mapping
                obs = self.agent.perception_mapping(state, step_i)

                if len(self.agent.memory) > self.cfg['replay_start_size']:
                    action, _ = self.agent.select_action(obs)
                else:
                    # random action
                    action = self.env.action_space.sample()
                # environment step
                next_state, reward_raw, done, truncated, info = self.env.step(action)
                # reward shaping
                reward = self.agent.reward_shaping(reward_raw)
                # store the transition
                self.agent.store(state, action, reward, next_state, done, truncated)
                # train the agent 1 step
                if (len(self.agent.memory) > self.cfg['replay_start_size'] and
                        training_steps % self.cfg['model_update_freq'] == 0):
                    self.agent.train_one_step()
                # update the state
                state = next_state
                # update the reward cumulated in the episode
                reward_cumulated += reward_raw
                # debug
                if (len(self.agent.memory) > self.cfg['replay_start_size'] and
                        training_steps % self.cfg['batch_num_per_epoch'] == 0):
                    # test the agent when the training steps reach the batch_num_per_epoch
                    run_test = True
                    epoch_i += 1

                # update the training step counter of the entire training process
                training_steps += 1
                # update the step counter of the current episode
                step_i += 1
                if done and info['lives'] == 0:
                    game_over = True
            # log the training reward
            self.logger.tb_scalar('training reward', reward_cumulated, training_steps)
            if run_test:
                # test the agent
                self.logger.msg(f'{epoch_i} test start:')
                avg_reward, avg_steps = self.test(self.cfg['agent_test_episodes'])
                avg_q = self.validate_q()
                # log the test reward
                self.logger.tb_scalar('avg_reward', avg_reward, epoch_i)
                self.logger.msg(f'{epoch_i} avg_reward: ' + str(avg_reward))
                self.metric['last reward'] = avg_reward
                if avg_reward > self.metric['max reward']:
                    self.metric['max reward'] = avg_reward
                # log the test steps
                self.logger.tb_scalar('avg_steps', avg_steps, epoch_i)
                self.logger.msg(f'{epoch_i} avg_steps: ' + str(avg_steps))
                self.metric['last steps'] = avg_steps
                if avg_steps > self.metric['max steps']:
                    self.metric['max steps'] = avg_steps
                # log the epsilon
                self.logger.tb_scalar('epsilon', self.agent.exploration_method.epsilon, epoch_i)
                self.logger.msg(f'{epoch_i} epsilon: ' + str(self.agent.exploration_method.epsilon))
                # log the q
                self.logger.tb_scalar('q', avg_q, epoch_i)
                self.logger.msg(f'{epoch_i} q: ' + str(avg_q))
                # log the maximum reward

                if self.cfg['save_model']:
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        self.agent.save_model(model_label='best')
                    self.agent.save_model(model_label='last')

    def test(self, test_episode_num: int):
        """
        Test the DQN agent for a given number of episodes.
        :param test_episode_num: The number of episodes for testing
        :return: The average reward and average steps per episode
        """
        env = AtariEnv(self.cfg['env_name'], frame_skip=self.cfg['skip_k_frame'], screen_size=self.cfg['screen_size'],
                       remove_flickering=True)
        exploration_method = EpsilonGreedy(self.cfg['epsilon_for_test'])
        reward_cum = 0
        step_cum = 0
        for i in range(test_episode_num):
            state, info = env.reset()
            done = truncated = False
            step_i = 0
            game_over = False
            while not game_over:
                obs = self.agent.perception_mapping(state, step_i)
                action, _ = self.agent.select_action(obs, exploration_method)
                next_state, reward, done, truncated, info = env.step(action)
                reward_cum += reward
                state = next_state
                step_i += 1
                if done and info['lives'] == 0:
                    game_over = True
            step_cum += step_i
        return reward_cum / test_episode_num, step_cum / test_episode_num
