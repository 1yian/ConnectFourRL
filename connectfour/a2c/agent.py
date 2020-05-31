import numpy as np
import torch
import torch.nn as nn

import connectfour.a2c.config as config


class ActorCriticAgent(nn.Module):

    def __init__(self, writer=None, output=True):
        super(ActorCriticAgent, self).__init__()

        self.iter = 0
        self.output = output
        self.writer = writer

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy_net = GenericNetwork(config.ACTION_DIM)
        self.value_net = GenericNetwork(1)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.L2_PENALTY,
                                          eps=config.EPS)

    def forward(self, x):
        x = x.to(self.device)
        policy = self.policy_net(x)

        value = self.value_net(x)
        value = torch.tanh(value)

        return policy, value

    def select_actions(self, states, valid_moves, greedy):
        states = torch.stack([self.preprocess_state(state) for state in states])
        with torch.no_grad():
            logits, _ = self(states)
        probs = nn.functional.softmax(logits, 1).cpu().detach()
        for prob, moves in zip(probs, valid_moves):
            for i in range(len(prob)):
                prob[i] *= moves[i]

        if greedy:
            return np.argmax(probs, 1)

        noise = self.get_noise(probs)
        probs += torch.Tensor(noise)
        dist = torch.distributions.categorical.Categorical(probs=probs)
        action = dist.sample()
        return states, np.array(action)

    @staticmethod
    def get_noise(probs):
        noises = []
        for i in range(len(probs)):
            noise = np.random.dirichlet([config.DIRECHLET_ALPHA] * len(probs[i]))
            noises.append(noise)
        return noises

    @staticmethod
    def preprocess_state(state):
        new_state = torch.zeros(2, *config.AGENT_INPUT)
        for i in range(len(state)):
            row = state[i]
            for j in range(len(row)):
                element = state[i][j]
                if element != 0:
                    channel = 0 if element == 1 else 1
                    new_state[channel][i + config.ROW_OFFSET][j + config.COL_OFFSET] = 1
        return new_state

    def train_on_loader(self, loader):

        for batch in loader:

            self.optimizer.zero_grad()
            states, actions, rewards = batch
            rewards = rewards.to(self.device) / 7
            policy_logits, value = self(states)
            policy_log_probs = nn.functional.log_softmax(policy_logits, dim=1)

            advantage = rewards - value.detach()

            policy_loss = - (advantage * policy_log_probs[range(len(actions)), actions]).mean()

            entropy_loss = (config.ENTROPY_SCALE * policy_log_probs * torch.exp(policy_log_probs)).sum(dim=1).mean()

            value = value.squeeze(1)
            value_loss = torch.nn.functional.smooth_l1_loss(value, rewards)

            total_loss = (policy_loss + entropy_loss + value_loss).mean()
            total_loss.backward()
            self.optimizer.step()

            if self.writer:
                self.writer.add_scalar('Train/actor loss', policy_loss, self.iter)
                self.writer.add_scalar('Train/critic loss', value_loss, self.iter)
                self.writer.add_scalar('Train/entropy', -entropy_loss, self.iter)
                self.iter += 1

            if self.output:
                print("Actor loss: {:2f}\tCritic loss: {:2f}\tEntropy: {:2f}".format(policy_loss, value_loss,
                                                                                     -entropy_loss))

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter': self.iter
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter = checkpoint['iter']


class GenericNetwork(nn.Module):
    def __init__(self, output_dim):
        super(GenericNetwork, self).__init__()
        self.activation = nn.LeakyReLU()

        self.conv_1 = nn.Conv2d(2, 64, 2, 1)
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 128, 2, 1)
        self.bn_2 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(4608, 256)

        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activation(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activation(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
