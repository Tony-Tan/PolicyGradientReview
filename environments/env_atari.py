# main class of env
import cv2
import gymnasium as gym
from gymnasium import envs
from gymnasium import Wrapper
from gymnasium.wrappers import AtariPreprocessing
from utils.commons import *
from gymnasium.spaces import Discrete
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
                try:
                    self.env = gym.make(env_id, repeat_action_probability=0.0, frameskip=1, render_mode=None,
                                        obs_type="grayscale")
                    # self.env = DQNAtariActionWrapper(self.env)
                except gym.error.Error as e:
                    raise EnvError(f"Failed to create environment: {str(e)}")
                self.screen_size = kwargs['screen_size'] if 'screen_size' in kwargs.keys() else None
                self.logger = kwargs['logger'] if 'logger' in kwargs.keys() else None
                self.frame_skip = kwargs['frame_skip'] if 'frame_skip' in kwargs.keys() else 1
                self.gray_state_Y = kwargs['gray_state_Y'] if 'gray_state_Y' in kwargs.keys() else True
                self.scale_state = kwargs['scale_state'] if 'scale_state' in kwargs.keys() else False
                self.remove_flickering = kwargs['remove_flickering'] if 'remove_flickering' in kwargs.keys() else True
                self.last_frame = None
                self.render_frame = None
                self.env_type = 'Atari'
                self.action_space = self.env.action_space
                self.state_space = self.env.observation_space
                if self.logger:
                    self.logger.msg(f'env id: {env_id} |repeat_action_probability: 0 ')
                    self.logger.msg(f'screen_size: {self.screen_size} | grayscale_obs:{self.gray_state_Y} \n'
                                    f'scale_obs:{self.scale_state} | action space size: {self.action_space.n}\n'
                                    f'remove flickering: {self.remove_flickering} | frame skip: {self.frame_skip}\n'
                                    f'state space shape: {self.state_space.shape} ')
        else:
            raise EnvError('atari game not exist in openai gymnasium')

    def __process_frame(self, frame):
        # if self.gray_state_Y:
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:, :, 0]
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.screen_size:
            frame = cv2.resize(frame, [self.screen_size, self.screen_size], interpolation=cv2.INTER_AREA)
        if self.scale_state:
            frame = frame / 255.
        return frame

    def reset(self):
        """Implement the `reset` method that initializes the environment to its initial state"""
        state, info = self.env.reset()
        self.last_frame = state
        self.render_frame = state
        state = self.__process_frame(state)

        return state, info

    def step(self, action):
        state, reward, done, trunc, info = None, 0, False, False, None
        state_removed_flickering = None
        for _ in range(self.frame_skip):
            state, reward_, done, trunc, info = self.env.step(action)
            reward += reward_
            state_removed_flickering = np.maximum(self.last_frame, state)
            self.render_frame = state_removed_flickering
            self.last_frame = state
            if done or trunc:
                reward = -1
                break

        state_processed = self.__process_frame(state_removed_flickering)
        return state_processed, reward, done, trunc, info

    def render(self):
        """
        Include a `render` method for visualizing the environment's current state.
        """
        return self.render_frame




if __name__ == '__main__':
    for key_i in envs.registry.keys():
        print(key_i)
