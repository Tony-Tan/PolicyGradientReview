import argparse
from agents.dqn_pp_agent import *
from environments.env_wrapper import AtariEnv
from exploration.epsilon_greedy import *
from utils.hyperparameters import Hyperparameters
from tools.dqn_play_ground import DQNPlayGround

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='PyTorch dqn_pp training arguments')
parser.add_argument('--env_name', default='ALE/Seaquest-v5', type=str,
                    help='openai gym environment (default: ALE/Pong-v5)')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--log_path', default='../exps/dqn_pp/', type=str,
                    help='log save path，default: /exps/dqn_pp/')

# Load hyperparameters from yaml file
cfg = Hyperparameters(parser, '../configs/dqn_pp.yaml')


def main():
    logger = Logger(cfg['env_name'], cfg['log_path'])
    logger.msg('\nparameters:' + str(cfg))
    env = AtariEnv(cfg['env_name'], frame_skip=cfg['skip_k_frame'], logger=logger, screen_size=cfg['screen_size'])

    dqn_agent = DQNPPAgent(cfg['screen_size'], env.action_space, cfg['mini_batch_size'],
                           cfg['replay_buffer_size'], cfg['replay_start_size'], cfg['learning_rate'], cfg['step_c'],
                           cfg['agent_saving_period'], cfg['gamma'], cfg['training_steps'], cfg['phi_channel'],
                           cfg['epsilon_max'], cfg['epsilon_min'], cfg['exploration_steps'], cfg['device'], logger)
    dqn_pg = DQNPlayGround(dqn_agent, env, cfg, logger)
    dqn_pg.train()


if __name__ == '__main__':
    main()
