import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import connectfour.a2c.config as config


class ActorCriticAgent(nn.Module):

    def __init__(self, writer=False, output=True):
        super(ActorCriticAgent, self).__init__()

        # Parameters for how we want to show training progress
        self.output = output
        self.writer = SummaryWriter() if writer else None
        self.iter = 0

        # Pick GPU if supported, else CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Here we define the key parts of the neural network.
        activation = nn.ReLU()
        conv_filters = 256

        # Take a convolutional head
        self.conv1 = nn.Conv2d(2, conv_filters, 1, 1)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(filters=conv_filters, activation=activation),
            ResidualBlock(filters=conv_filters, activation=activation),
            ResidualBlock(filters=conv_filters, activation=activation),
            ResidualBlock(filters=conv_filters, activation=activation),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(conv_filters, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.Flatten(start_dim=1),
            activation,
            nn.Linear(84, config.ACTION_DIM),
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

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.L2_PENALTY,
                                          eps=config.EPS)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=config.LEARNING_RATE_DECAY)

    def forward(self, x, valid_moves):
        x = x.to(self.device)

        x = self.conv1(x)
        x = self.residual_blocks(x)

        policy = self.policy_head(x)
        policy = self._mask_invalid_actions(policy, valid_moves)
        value = self.value_head(x)
        return policy, value

    def _mask_invalid_actions(self, logits, valid_moves):
        valid_moves = torch.log(valid_moves)

        min_mask = torch.ones(*valid_moves.size(), dtype=torch.float) * torch.finfo(torch.float).min
        inf_mask = torch.max(valid_moves, min_mask).to(self.device)
        # Mask the logits of invalid actions with a number very close to the minimum float number.
        return logits + inf_mask

    def select_actions(self, states, greedy):
        """
        Non-grad selection purely for inference/exploration.
        Takes in a vector of states and outputs the agents selected action.

        :param states: a vector of numpy arrays representing the raw state of connect four.
        :param greedy: whether or not to pick our action using a greedy or stochastic policy.

        :return: actions: a resultant vector of actions from the agent.
                 training_data: a dict of information containing:
                     action_probs: the policy probabilities of the selected actions for the given states: π(actions | states).
                     processed_states: a vector of the processed_states for input into the neural network
                     valid_moves: a vector of the one hot encoded valid moves
        """

        valid_moves = self._get_valid_moves(states)
        processed_states = torch.stack([self._preprocess_state(state) for state in states])

        noisy = (config.DIRECHLET_ALPHA > 0) and not greedy
        probs = self.get_policy_dist(processed_states, valid_moves, noisy=noisy)

        dist = torch.distributions.categorical.Categorical(probs=probs)
        actions = dist.sample().numpy() if not greedy else np.argmax(probs, 1)
        action_prob = probs[range(len(actions)), actions]

        training_data = {'processed_states': processed_states, 'valid_moves': valid_moves, 'action_probs': action_prob}

        return actions, training_data

    def get_policy_dist(self, processed_states, valid_moves, noisy):
        with torch.no_grad():
            logits, value = self(processed_states, valid_moves)
            logits = logits.cpu()
            probs = nn.functional.softmax(logits, 1)

        if noisy:
            probs = self._get_noise(probs, valid_moves)
        return probs


    @staticmethod
    def _get_valid_moves(states):
        valid_moves_vector = []
        for state in states:
            top_row = state[0]
            valid_moves = [0 if element != 0 else 1 for element in top_row]
            valid_moves_vector.append(valid_moves)
        return torch.Tensor(valid_moves_vector)

    @staticmethod
    def _get_noise(probs: torch.Tensor, valid_moves: torch.Tensor) -> torch.Tensor:
        valid_moves = valid_moves.numpy()
        for moves, prob in zip(valid_moves, probs):
            num_moves = int(np.sum(moves))
            noise = torch.Tensor(np.random.dirichlet([config.DIRECHLET_ALPHA] * num_moves))
            count = 0

            for i in range(len(prob)):
                if moves[i] == 1:
                    prob[i] += noise[count]
                    count += 1
            prob /= torch.sum(prob)
        return probs

    @staticmethod
    def _preprocess_state(state: np.ndarray) -> torch.Tensor:
        top_channel = (state == 1)
        bottom_channel = (state == -1)
        result = np.stack([top_channel, bottom_channel]).astype(int)
        result = torch.from_numpy(result).float()
        return result

    def train_on_loader(self, loader: DataLoader):
        for batch in loader:
            self.optimizer.zero_grad()
            states, actions, rewards, probs, valid_moves = batch
            rewards = rewards.to(self.device) / 4

            policy_logits, value = self(states, valid_moves)

            policy_log_probs = nn.functional.log_softmax(policy_logits, dim=1)

            advantage = rewards - value.detach()

            log_probs = policy_log_probs[range(len(actions)), actions]

            current_probs = torch.exp(log_probs).clone().detach()
            importance_sample_ratio = current_probs / probs.to(self.device)

            policy_loss = (importance_sample_ratio.detach() * (-advantage * log_probs)).mean()

            entropy_loss = (config.ENTROPY_SCALE * policy_log_probs * torch.exp(policy_log_probs)).sum(dim=1).mean()

            value = value.squeeze(1)
            value_loss = torch.nn.functional.mse_loss(value, rewards).mean()

            total_loss = policy_loss + value_loss + entropy_loss
            total_loss.backward()
            self.optimizer.step()

            if self.writer:
                self.writer.add_scalar('Train/entropy_loss', -entropy_loss, self.iter)
                self.writer.add_scalar('Train/policy_loss', policy_loss, self.iter)
                self.writer.add_scalar('Train/value_loss', value_loss, self.iter)
                self.iter += 1

            if self.output:
                print("Policy loss: {:2f}\tValue loss: {:2f}\tEntropy loss: {:2f}".format(policy_loss, value_loss,
                                                                                          -entropy_loss))

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': self.iter
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter = checkpoint['iter']



class ResidualBlock(nn.Module):
    def __init__(self, filters=256, kernel_size=3, activation=nn.functional.leaky_relu):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.padding = (kernel_size - 1) // 2

        self.activation = activation

        self.conv1 = nn.Conv2d(self.filters, self.filters, self.kernel_size, 1, padding=self.padding)
        self.conv1_bn = nn.BatchNorm2d(self.filters)

        self.conv2 = nn.Conv2d(self.filters, self.filters, self.kernel_size, 1, padding=self.padding)
        self.conv2_bn = nn.BatchNorm2d(self.filters)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = x + input
        x = self.activation(x)
        return x
