import time
from agents.async_dqn_agent import *
import argparse
import random
from agents.dqn_agent import *
from environments.env_atari import AtariEnv
from exploration.epsilon_greedy import *
from utils.configurator import Configurator
from tools.dqn_play_ground import DQNPlayGround
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)
global async_dqn_agent

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='PyTorch dqn training arguments')
parser.add_argument('--env_name', default='ALE/Breakout-v5', type=str,
                    help='openai gym environment (default: ALE/Atlantis-v5)')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--log_path', default='../exps/async_dqn/', type=str,
                    help='log save path，default: ../exps/async_dqn/')

# Load hyperparameters from yaml file
cfg = Configurator(parser, '../configs/async_dqn.yaml')


def test(agent, test_episode_num: int):
    """
    Test the DQN agent for a given number of episodes.
    :param test_episode_num: The number of episodes for testing
    :return: The average reward and average steps per episode
    """
    env = AtariEnv(cfg['env_name'],  frame_skip=cfg['skip_k_frame'])
    exploration_method = EpsilonGreedy(cfg['epsilon_for_test'])
    reward_cum = 0
    step_cum = 0
    for i in range(test_episode_num):
        state, _ = env.reset()
        done = truncated = False
        step_i = 0
        while (not done) and (not truncated):
            obs = agent.perception_mapping(state, step_i)
            action = agent.select_action(obs, exploration_method)
            next_state, reward, done, truncated, inf = env.step(action)
            reward_cum += reward
            state = next_state
            step_i += 1
        step_cum += step_i
    return reward_cum / cfg['agent_test_episodes'], step_cum / cfg['agent_test_episodes']


def train_processor(rank: int, agent: AsyncDQNAgent, env: AtariEnv,
                    training_steps_each_worker: int,
                    no_op: int, batch_per_epoch: int, seed: int):
    # training
    np.random.seed(seed)
    random.seed(seed)
    env.env.unwrapped.seed(seed)
    training_steps = 0
    episode = 0
    epoch_i = 0
    while training_steps < training_steps_each_worker:
        run_test = False
        state, _ = env.reset()
        done = False
        truncated = False
        step_i = 0
        reward_cumulated = 0
        obs = agent.perception_mapping(state, step_i)
        while (not done) and (not truncated):
            #
            if step_i >= no_op:
                action = agent.select_action(obs)
            else:
                action = agent.select_action(obs, RandomAction())
            next_state, reward_raw, done, truncated, inf = env.step(action)

            reward = agent.reward_shaping(reward_raw)
            next_obs = agent.perception_mapping(next_state, step_i)
            agent.store(obs, action, reward, next_obs, done, truncated, rank)
            agent.train_one_step()
            obs = next_obs
            reward_cumulated += reward
            training_steps += 1
            step_i += 1
            if rank == 0 and training_steps % batch_per_epoch == 0:
                run_test = True
                epoch_i += 1
            # print(f'pid {rank} - {training_steps} training steps')
        # if rank == 0:
        agent.logger.msg(f'pid {rank} - {step_i}/{training_steps} training reward: ' + str(reward_cumulated))
        agent.logger.tb_scalar(f'training reward/pid_{rank} ', reward_cumulated, training_steps)
        if run_test:
            agent.logger.msg(f'{epoch_i} test start:')
            avg_reward, avg_steps = test(agent, cfg['agent_test_episodes'])
            agent.logger.tb_scalar('avg_reward', avg_reward, epoch_i)
            agent.logger.tb_scalar('avg_steps', avg_steps, epoch_i)
            agent.logger.tb_scalar('epsilon', agent.exploration_method.epsilon, epoch_i)
            agent.logger.msg(f'{epoch_i} avg_reward: ' + str(avg_reward))
            agent.logger.msg(f'{epoch_i} avg_steps: ' + str(avg_steps))
            agent.logger.msg(f'{epoch_i} epsilon: ' + str(agent.exploration_method.epsilon))

        episode += 1


class AsyncDQNPlayGround:
    def __init__(self, agent: AsyncDQNAgent, env: list, cfg:Configurator):
        self.env_list = env
        self.agent = agent
        self.cfg = cfg
        self.worker_num = cfg['worker_num']
        self.training_steps_each_worker = int(self.cfg['training_steps'] / self.worker_num)

    def train(self):
        processes = []

        for rank in range(self.worker_num):
            seed = random.randint(0, 10000)
            p = mp.Process(target=train_processor, args=(rank, self.agent, self.env_list[rank],
                                                         self.training_steps_each_worker,
                                                         self.cfg['no_op'],
                                                         cfg['batch_num_per_epoch'],
                                                         seed))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    logger = Logger(cfg['env_name'], cfg['log_path'])
    logger.msg('\nparameters:' + str(cfg))
    manager = mp.Manager()
    envs = []
    for _ in range(cfg['worker_num']):
        env = AtariEnv(cfg['env_name'], frame_skip=cfg['skip_k_frame'])
        envs.append(env)
    async_dqn_agent = AsyncDQNAgent(cfg['input_frame_width'], cfg['input_frame_height'],
                                    envs[0].action_space, cfg['mini_batch_size'], cfg['replay_buffer_size'],
                                    cfg['learning_rate'], cfg['step_c'], cfg['agent_saving_period'], cfg['gamma'],
                                    cfg['training_steps'], cfg['phi_channel'], cfg['epsilon_max'],
                                    cfg['epsilon_min'], cfg['exploration_steps'], cfg['device'], manager, logger)

    dqn_pg = AsyncDQNPlayGround(async_dqn_agent, envs, cfg)
    dqn_pg.train()
