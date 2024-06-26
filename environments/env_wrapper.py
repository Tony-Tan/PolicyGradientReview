# main class of env
import cv2
import gymnasium as gym
from gymnasium import envs
from gymnasium import Wrapper
from gymnasium.wrappers import AtariPreprocessing
from utils.commons import *

custom_env_list = []


class EnvError(Exception):
    def __init__(self, error_inf):
        self.error_inf = error_inf

    def __str__(self):
        return 'Environment error: ' + self.error_inf


class AtariEnv:
    # Class for perception mapping in DQN for Atari games.
    # Preprocess the observation by taking the maximum value for each pixel colour value over the frame being encoded
    # and the previous frame. This is necessary to remove flickering that is present in games where some objects
    # appear only in even frames while other objects appear only in odd frames, an artefact caused by the limited
    # number of sprites Atari 2600 can display at once. Second, we then extract the Y channel, also known as
    # luminance, from the RGB frame and rescale it to $84\times 84$.
    # from paper: Playing Atari with Deep Reinforcement Learning
    def __init__(self, env_id: str, **kwargs):
        if env_id in gym.envs.registry.keys():
            if 'ALE' in env_id:
                self.env_id = env_id
                self.env = gym.make(env_id, repeat_action_probability=0.0, frameskip=1, render_mode=None)
                self.screen_size = kwargs['screen_size'] if 'screen_size' in kwargs.keys() else None
                self.logger = kwargs['logger'] if 'logger' in kwargs.keys() else None
                self.frame_skip = kwargs['frame_skip'] if 'frame_skip' in kwargs.keys() else 1
                self.gray_state_Y = kwargs['gray_state_Y'] if 'gray_state_Y' in kwargs.keys() else True
                self.scale_state = kwargs['scale_state'] if 'scale_state' in kwargs.keys() else False
                self.remove_flickering = kwargs['remove_flickering'] if 'remove_flickering' in kwargs.keys() else True
                self.last_frame = None
                self.lives_counter = 0
                self.env_type = 'Atari'
                self.action_space = self.env.action_space
                self.state_space = self.env.observation_space
                if self.logger:
                    self.logger.msg(f'env id: {env_id} |repeat_action_probability: 0 ')
                    self.logger.msg(f'frame_skip: {self.frame_skip} |screen_size: {self.screen_size} |'
                                    f'grayscale_obs:{True} |scale_obs:{False}')
        else:
            raise EnvError('atari game not exist in openai gymnasium')

        # self.logger('EnvWrapper init: {env_id} from {env_type} had be built'.
        #             format(env_id=self.env_id, env_type=self.env_type))

    def reset(self):
        """Implement the `reset` method that initializes the environment to its initial state"""
        state, info = self.env.reset()
        if 'lives' in info.keys():
            self.lives_counter = info['lives']

        if self.gray_state_Y:
            state = cv2.cvtColor(state, cv2.COLOR_BGR2YUV)[:,:,0]
        if self.screen_size:
            state = cv2.resize(state, [self.screen_size, self.screen_size])
        if self.remove_flickering:
            self.last_frame = state
        if self.scale_state:
            state = state/255.
        return state, info

    def step(self, action):
        reward_cum = 0
        state = self.last_frame
        done = trunc = info = None
        for i in range(self.frame_skip):
            if self.remove_flickering:
                self.last_frame = state
            state, reward, done, trunc, info = self.env.step(action)
            if 'lives' in info.keys() and info['lives'] < self.lives_counter:
                self.lives_counter = info['lives']
                reward = -1
            reward_cum += reward
            if done or trunc:
                break
        # cv2.imshow('frame', cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(30)
        if self.gray_state_Y:
            state = cv2.cvtColor(state, cv2.COLOR_BGR2YUV)[:, :, 0]
        if self.screen_size:
            state = cv2.resize(state, [self.screen_size, self.screen_size])
        if self.remove_flickering:
            if self.gray_state_Y:
                self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
            if self.screen_size:
                self.last_frame = cv2.resize(self.last_frame, [self.screen_size, self.screen_size])
            state = np.maximum(state, self.last_frame)
        if self.scale_state:
            state = state / 255.
        return state, reward_cum, done, trunc, info

    def render(self):
        """
        Include a `render` method for visualizing the environment's current state.
        """
        pass


if __name__ == '__main__':
    for key_i in envs.registry.keys():
        print(key_i)
