import torch
from torch.utils.data.dataset import Dataset

import connectfour.a2c.config as config


class BellmanDiscount:

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, episode):
        return self.discount(episode)

    def discount(self, episode):
        running_return = 0
        for exp in reversed(episode):
            running_return = exp['reward'] + self.gamma * running_return
            exp['reward'] = running_return
        return episode


class ExperienceReplayBuffer(Dataset):

    def __init__(self, transform=BellmanDiscount(config.GAMMA)):
        self.experiences = []
        self.transform = transform

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        return self.experiences[idx]

    def add(self, episode):
        if self.transform:
            self.transform(episode)
        self.experiences += episode

    def clear(self):
        self.experiences.clear()


def collate_experiences(experiences):
    batch_states = []
    batch_actions = []
    batch_rewards = []
    for exp in experiences:
        batch_states.append(exp['state'])
        batch_actions.append(exp['action'])
        batch_rewards.append(exp['reward'])
    return torch.stack(batch_states), torch.LongTensor(batch_actions), torch.Tensor(batch_rewards)