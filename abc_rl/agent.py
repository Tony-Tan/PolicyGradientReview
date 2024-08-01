from abc import ABC, abstractmethod
from environments.env_wrapper import *


class Agent(ABC):
    def __init__(self, logger:Logger):
        self.logger = logger
        pass

    # def __save_folder_create(self):
    #     if not os.path.exists(self.save_path):
    #         os.makedirs(self.save_path)
    @abstractmethod
    def select_action(self,  **kwargs) -> np.ndarray:
        ...

    @abstractmethod
    def store(self, **kwargs):
        ...

    @abstractmethod
    def train_one_step(self, **kwargs):
        ...

    @abstractmethod
    def save_model(self, **kwargs):
        ...

    @abstractmethod
    def load_model(self, **kwargs):
        ...

