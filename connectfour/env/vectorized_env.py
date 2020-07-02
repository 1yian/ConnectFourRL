import time

import torch.multiprocessing as mp


class VectorizedEnv:
    def __init__(self, game_class, num_envs):
        self.envs = [game_class() for _ in range(num_envs)]
        self.num_workers = min(mp.cpu_count(), num_envs)


    def step(self, actions):
        states, rewards, dones = [], [], []
        for action, env in zip(actions, self.envs):
            state, reward, done = env.step(action)
            states.append(state)
            rewards.append(reward)
            if done:
                env.reset()
            dones.append(done)
        return states, rewards, dones

    def reset(self):
        return [env.reset() for env in self.envs]

    def render(self):
        return [env.render() for env in self.envs]
