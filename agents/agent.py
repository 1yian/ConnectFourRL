import random
import torch
import torch.nn as nn
from env.env import ConnectFour


class TorchAgent(nn.Module()):

    def __init__(self):
        super().__init__()

    def select_move(self, game: ConnectFour, eval=True):
        """

        :param state: A copy of a ConnectFour instance, representing the current game state.
        :param eval: Whether to pick moves greedily for evaluation, or based on some sort of exploration scheme.
        :return: The agent's desired move as an integer.
        """
        return random.randint(0, ConnectFour.NUM_COLS - 1)

    @staticmethod
    def get_default_params():
        """
        Get the default parameters of this agent.
        :return: A dict with keys being the parameter type and the values being the desired setting of the corresponding parameter.
        """
        return {
            'seed': 42
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
