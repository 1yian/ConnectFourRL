import random

import torch
import torch.nn as nn
from hyperopt import hp

from env.game import ConnectFourState


class TorchAgent(nn.Module()):

    def __init__(self):
        super().__init__()

    def select_move(self, game: ConnectFourState, eval=True):
        """

        :param state: A copy of a ConnectFour instance, representing the current game state.
        :param eval: Whether to pick moves greedily for evaluation, or based on some sort of exploration scheme.
        :return: The agent's desired move as an integer.
        """
        return random.randint(0, ConnectFourState.NUM_COLS - 1)

    @staticmethod
    def get_default_params():
        """
        Get the default parameters of this agent.
        :return: A dict with keys being the parameter type and the values being the desired setting of the corresponding parameter.
        """
        return {
            'seed': 42
        }

    @staticmethod
    def get_hyperparam_space():
        """
        Get the hyperparam tuning space of the agent.
        :return: A dict with hp spaces for the tunable parameters defined in get_default_params
        """
        return {
            'seed': hp.uniform(-100, 100)
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
