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

        self.conv1 = nn.Conv2d(2, 64, 3, 1)
        self.res1 = ResidualBlock(filters=64)
        self.res2 = ResidualBlock(filters=64)

        self.res3 = ResidualBlock(filters=64)
        self.policy_conv = nn.Conv2d(64, 2, 1, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_head = nn.Linear(40, config.ACTION_DIM)

        self.val_conv = nn.Conv2d(64, 2, 1, 1)
        self.val_bn = nn.BatchNorm2d(2)
        self.val_fc = nn.Linear(40, 1)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.L2_PENALTY,
                                          eps=config.EPS)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = nn.Flatten(start_dim=1)(p)
        p = nn.functional.leaky_relu(p)
        p = self.policy_head(p)

        v = self.val_conv(x)
        v = self.val_bn(v)
        v = nn.Flatten(start_dim=1)(v)
        v = nn.functional.leaky_relu(v)
        v = self.val_fc(v)
        v = torch.tanh(v)

        return p, v

    def select_actions(self, states, valid_moves, greedy):
        states = torch.stack([self.preprocess_state(state) for state in states])
        with torch.no_grad():
            logits, _ = self(states)
            logits = logits.cpu().detach()
            unmasked_probs = nn.functional.softmax(logits, 1).clone()
            for l, moves in zip(logits, valid_moves):
                for i in range(len(l)):
                    if moves[i] == 0:
                        l[i] = -1e999
            probs = nn.functional.softmax(logits, 1)

        if greedy:
            actions = np.argmax(probs, 1)
            return actions, probs[range(len(actions)), actions], states

        dist = torch.distributions.categorical.Categorical(probs=probs)
        actions = dist.sample()
        return np.array(actions), unmasked_probs[range(len(actions)), actions], states

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
            states, actions, rewards, probs, valid_moves = batch
            rewards = rewards.to(self.device) / 4

            policy_logits, value = self(states)

            policy_log_probs = nn.functional.log_softmax(policy_logits, dim=1)

            advantage = rewards - value.detach()

            log_probs = policy_log_probs[range(len(actions)), actions]

            masked_probs = torch.exp(log_probs).clone().detach()
            importance_sample_ratio = masked_probs / probs.to(self.device)

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
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter = checkpoint['iter']


class GenericNetwork(nn.Module):
    def __init__(self, output_dim):
        super(GenericNetwork, self).__init__()
        self.activation = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(2, 256, 4, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 64, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(384, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


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
