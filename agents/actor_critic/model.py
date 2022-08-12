from typing import Any, List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from agents.utils.utils import tensorize_game_state, hot_encode_valid_moves
from agents.utils.residual_block import ResidualBlock
from agents.utils.dataset import ConnectFourDataset
from env.game import ConnectFourState

import agents.agent


class ActorCritic(agents.agent.TorchAgent):
    def __init__(self, params):
        super().__init__()
        activation = nn.LeakyReLU()

        self.params = params

        conv_filters = self.params['num_filters_per_layer']
        self.input_layer = nn.Conv2d(4, conv_filters, 1, 1)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(filters=conv_filters, activation=activation),
            ResidualBlock(filters=conv_filters, activation=activation),
            ResidualBlock(filters=conv_filters, activation=activation),
            ResidualBlock(filters=conv_filters, activation=activation)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(conv_filters, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Flatten(start_dim=1),
            activation,
            nn.Linear(42, 256),
            activation,
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(conv_filters, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.Flatten(start_dim=1),
            activation,
            nn.Linear(84, ConnectFourState.NUM_COLS),
        )

    def forward(self, games: List[ConnectFourState]) -> dict[str, Any]:
        """
        Takes in a batch of processed game states, runs it forward through the network and returns policy and value
        predictions with grad attached.
        :param games: List of ConnectFour instances representing a batch.
        :return: dict containing:
            policy: torch.Tensor- batch of log probability distributions on moves
            value: torch.Tensor - batch of predicted value of the states
            greedy_move: torch.IntegerTensor - batch of predicted greedy moves the agent
                                                would make in this state (deterministic)
            exploration_move: torch.IntegerTensor - batch of predicted exploration
                                                    moves the agent would make in this state (stochastic)
        """

        states = torch.Tensor([agents.utils.utils.tensorize_game_state(game) for game in games]).float().to(self.device)
        x = self.input_layer(states)
        x = self.residual_blocks(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        mask = torch.zeros_like(logits)
        valid_moves = hot_encode_valid_moves([game.get_valid_moves() for game in games])
        mask[torch.from_numpy(valid_moves).to(self.device)] = -1e99

        masked_logits = logits + mask

        return {
            'policy': masked_logits,
            'value': value,
            'greedy_move': torch.argmax(masked_logits.detach().cpu(), 0).item(),
            'exploration_move': None,  # TODO
        }

    def sample_exploration_move(self, logits):
        noise = torch.Tensor(np.random.dirichlet([self.params['dirichlet_alpha']] * ConnectFourState.NUM_COLS))
        # TODO

    def select_move(self, game: ConnectFourState, eval=True):
        with torch.no_grad():
            ret = self.forward([game])
        return ret['greedy_move' if eval else 'exploration_move'].detach().cpu().item()

    def train(self):
        env = ConnectFourState()
        dataset = ConnectFourDataset()
        optim = torch.optim.AdamW(self.parameters(), lr=self.params['learning_rate'],
                                  weight_decay=self.params['weight_decay'])

        for epoch in range(self.params['num_params']):
            for episode_idx in range(self.params['episode_per_epoch']):
                record = self._selfplay_rollout(env, greedy=False)
                dataset.add_record(record, discount_factor=self.params['discount_factor'])
            # TODO: Training here.
            dataset.clear()

    def _selfplay_rollout(self, env: ConnectFourState, greedy=True) -> List[Dict]:
        done = env.reset()
        record = []

        while not done:
            prediction = self.forward([env])[0]
            move = prediction['greedy_move' if greedy else 'exploration_move'].detach().cpu().item()
            record.append(
                {
                    'prediction': prediction,
                    'greedy': greedy,
                    'player': env.turn,
                    'state': env.copy()
                }
            )
            done = env.move(move)

        return record

    @staticmethod
    def train_model(params=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = params if params is not None else ActorCritic.get_default_params()

        agent = ActorCritic(params).to(device)
        agent.train()

        # TODO: evaluate agent, return score (ie winrate vs baseline)
        return 0

    @staticmethod
    def get_default_params():
        return {
            'num_filters_per_layer': 256,
            'learning_rate': 1e-3,
            'learning_rate_decay': 0.981,
            'discount_factor': 1,
            'entropy_scale': 1e-5,
            'dirichlet_alpha': 0.25,
            'weight_decay': 1e-5,

            'num_epochs': 300,
            'episodes_per_epoch': 1,

        }

    @staticmethod
    def get_hyperparam_space():
        # TODO
        return {

        }
