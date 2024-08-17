import argparse
from tools.dqn_play_ground import DQNPlayGround
from agents.double_dqn_agent import *
from environments.env_atari import AtariEnv
from exploration.epsilon_greedy import *
import copy
import cv2
from tqdm import tqdm
from multiprocessing import Process, Queue, set_start_method
from utils.configurator import Configurator

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='PyTorch Double DQN training arguments')
parser.add_argument('--env_name', default='ALE/Asterix-v5', type=str,
                    help='openai gym environment (default: ALE/Pong-v5)')
parser.add_argument('--device', default='cuda:1', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--log_path', default='../exps/double_dqn/', type=str,
                    help='log save path，default: ./log/')

# Load hyperparameters from yaml file and combine with command line arguments
cfg = Configurator(parser, '../configs/double_dqn.yaml')


def main():
    logger = Logger(cfg['env_name'], cfg['log_path'])
    logger.msg('\nparameters:' + str(cfg))
    env = AtariEnv(cfg['env_name'], frame_skip=cfg['skip_k_frame'])
    double_dqn_agent = DoubleDQNAgent(cfg['screen_size'], env.action_space,
                                      cfg['mini_batch_size'], cfg['replay_buffer_size'], cfg['replay_start_size'],
                                      cfg['learning_rate'], cfg['step_c'], cfg['agent_saving_period'], cfg['gamma'],
                                      cfg['training_steps'], cfg['phi_channel'], cfg['epsilon_max'], cfg['epsilon_min'],
                                      cfg['exploration_steps'], cfg['device'], logger)
    dqn_pg = DQNPlayGround(double_dqn_agent, env, cfg, logger)
    dqn_pg.train()


if __name__ == '__main__':
    main()
