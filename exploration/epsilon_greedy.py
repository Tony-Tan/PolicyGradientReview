from abc_rl.exploration import *
import numpy as np

class EpsilonGreedy(Exploration):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        super().__init__()

    def __call__(self, values: np.ndarray):
        optimal_action = np.random.choice(
            np.flatnonzero(values == np.max(values)))
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(values))
        else:
            return optimal_action



class DecayingEpsilonGreedy(Exploration):
    def __init__(self, max_epsilon: float, min_epsilon: float, total_step: int):
        super().__init__()
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = (max_epsilon - min_epsilon) / total_step
        self.decaying_counter = 0

    def __call__(self, values: np.ndarray):
        self.decaying_counter += 1
        self.epsilon = max(self.max_epsilon - self.decaying_counter * self.decay_rate, self.min_epsilon)
        optimal_action = np.random.choice(np.flatnonzero(values == np.max(values)))
        # if a float random number less than epsilon, then explore
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(values))
        else:
            return optimal_action



        # if random.randint(0, 1_000_000) < self.epsilon * 1_000_000:
        #     return random.randint(0, len(values) - 1)
        # else:
        #     return optimal_action
