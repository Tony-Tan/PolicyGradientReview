import argparse
from agents.dqn_pp_agent import *
from environments.env_atari import AtariEnv
from exploration.epsilon_greedy import *
from utils.configurator import Configurator

from tools.dqn_play_ground import DQNPlayGround

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='PyTorch dqn training arguments')
parser.add_argument('--env_name', default='ALE/SpaceInvaders-v5', type=str,
                    help='openai gym environment (default: ALE/Atlantis-v5)')
parser.add_argument('--n_times', default=1, type=int,
                    help='how many times to run the experiment (default: 1)')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--save_model', default=True, type=bool,
                    help='save model or not, default: True')
parser.add_argument('--exp_path', default='../exps/dqn_pp/', type=str,
                    help='exp save path，default: ../exps/dqn_pp/')


def main():
    # Load hyperparameters from yaml file
    cfg = Configurator(parser, '../configs/dqn_pp.yaml')
    logger = Logger(cfg['exp_path'], cfg['exp_name'])
    logger.msg('\nparameters:' + str(cfg))

    # set seed
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gym.utils.seeding.np_random(seed)

    # 保证在CUDA下的可重复性
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env = AtariEnv(cfg['env_name'], frame_skip=cfg['skip_k_frame'], logger=logger, screen_size=cfg['screen_size'],
                   remove_flickering=True, seed=cfg['seed'])
    dqn_pp_agent = DQNPPAgent(cfg['screen_size'], env.action_space, cfg['mini_batch_size'],
                              cfg['replay_buffer_size'], cfg['replay_start_size'], cfg['learning_rate'],
                              cfg['step_c'], cfg['gamma'], cfg['training_steps'], cfg['phi_channel'],
                              cfg['epsilon_max'], cfg['epsilon_min'], cfg['exploration_steps'],
                              cfg['device'], cfg['exp_path'], cfg['exp_name'], logger)
    dqn_pg = DQNPlayGround(dqn_pp_agent, env, cfg, logger)
    dqn_pg.train()


if __name__ == '__main__':
    for i in range(parser.parse_args().n_times):
        main()
