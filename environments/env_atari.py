import cv2
import gymnasium as gym
from gymnasium import envs
from utils.commons import *
from gymnasium.spaces import Discrete
from collections import deque
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from gymnasium.wrappers import AtariPreprocessing

custom_env_list = []

class EnvError(Exception):
    def __init__(self, error_inf):
        self.error_inf = error_inf

    def __str__(self):
        return 'Environment error: ' + self.error_inf


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.fire_action = 1

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, trunc, info = self.env.step(self.fire_action)
        if done:
            self.env.reset(**kwargs)
        return obs, info


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
                self.logger = kwargs['logger'] if 'logger' in kwargs.keys() else None
                self.frame_skip = kwargs['frame_skip'] if 'frame_skip' in kwargs.keys() else 1
                self.no_op_max = kwargs['no_op_max'] if 'no_op_max' in kwargs.keys() else 30
                self.seed = kwargs['seed'] if 'seed' in kwargs.keys() else 0
                self.last_frame = None
                self.render_frame = None
                self.env_type = 'Atari'
                self.life_counter = 0
                self.env_id = env_id
                try:
                    self.env = gym.make(env_id, repeat_action_probability=0.0, frameskip=1, full_action_space=False)
                except gym.error.Error as e:
                    raise EnvError(f"Failed to create environment: {str(e)}")

                self.env = AtariPreprocessing(self.env, noop_max=self.no_op_max, frame_skip=1,
                                              screen_size=84, terminal_on_life_loss=False, grayscale_obs=True,
                                              scale_obs=False)

                self.env = FireResetEnv(self.env)
                self.action_space = self.env.action_space
                self.state_space = self.env.observation_space
                self._obs_buffer = deque(maxlen=4)  # Store the last two observations to compute the max

        else:
            raise EnvError('Atari game not exist in openai gymnasium')

    def reset(self):
        self._obs_buffer.clear()
        obs, info = self.env.reset(seed=self.seed)
        # if 'lives' in info.keys():
        #     self.life_counter = info['lives']
        self._obs_buffer.append(obs)
        self.render_frame = obs
        return obs, info

    def step(self, action):
        next_obs, reward, done, trunc, info = None, None, None, None, None
        reward_sum = 0
        for _ in range(self.frame_skip):
            next_obs, reward, done, trunc, info = self.env.step(action)
            reward_sum += reward
            self._obs_buffer.append(next_obs)
            # if self.life_counter > info['lives']:
            #     done = True  # 标记子回合结束
            #     reward_sum = -1  # 设置负奖励
            #     self.life_counter = info['lives']
            if done or trunc:
                break
        next_obs = np.max(np.stack(self._obs_buffer), axis=0)
        self.render_frame = next_obs
        return next_obs, reward_sum, done, trunc, info

    def render(self):
        """
        Include a `render` method for visualizing the environment's current state.
        """
        return self.render_frame


if __name__ == '__main__':
    for key_i in envs.registry.keys():
        print(key_i)
