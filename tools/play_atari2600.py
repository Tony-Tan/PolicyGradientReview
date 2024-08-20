import random
import cv2
import argparse
from agents.dqn_agent import *
from abc_rl.experience_replay import *
from abc_rl.exploration import *
from utils.configurator import *

parser = argparse.ArgumentParser(description='DQN Play Atari 2600')
parser.add_argument('--env_name', default='ALE/SpaceInvaders-v5', type=str,
                    help='openai gym environment (default: ALE/Atlantis-v5)')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--model_path',
                    default='../exps/dqn/ALE-SpaceInvaders-v5_2024-08-19-15-17-54/best.pth', type=str,
                    help='exp save path，default: ../exps/dqn/')
parser.add_argument('--epsilon_for_test', default='0.05', type=float,
                    help='epsilon greedy for testing:')
cfg = Configurator(parser, '../configs/dqn.yaml')
args = parser.parse_args()

env = AtariEnv(cfg['env_name'], frame_skip=cfg['skip_k_frame'], logger=None, screen_size=cfg['screen_size'],
               remove_flickering=True)
dqn_agent = DQNAgent(cfg['screen_size'], env.action_space, cfg['mini_batch_size'],
                     cfg['replay_buffer_size'], cfg['replay_start_size'], cfg['learning_rate'], cfg['step_c'],
                     cfg['gamma'], cfg['training_steps'], cfg['phi_channel'],
                     cfg['epsilon_max'], cfg['epsilon_min'], cfg['exploration_steps'], cfg['device'],
                     cfg['exp_path'], cfg['exp_name'], None)
dqn_agent.load_model(args.model_path)
exploration_method = EpsilonGreedy(args.epsilon_for_test)
reward_cum = 0
step_cum = 0
while True:
    state, info = env.reset()
    done = truncated = False
    step_i = 0
    while not (done or truncated):
        state_show = cv2.cvtColor(env.render(), cv2.cv2.COLOR_BGR2RGB)
        cv2.imshow(args.env_name, cv2.resize(state_show, [state_show.shape[1]*5, state_show.shape[0]*5]))
        cv2.imshow('state', cv2.resize(state, [state.shape[1] * 5, state.shape[0] * 5], interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(10)
        obs = dqn_agent.perception_mapping(state, step_i)
        action, _ = dqn_agent.select_action(obs, exploration_method)
        next_state, reward, done, truncated, inf = env.step(action)
        if reward != 0:
            print(reward)
        reward_cum += reward
        state = next_state
        step_i += 1
    step_cum += step_i
