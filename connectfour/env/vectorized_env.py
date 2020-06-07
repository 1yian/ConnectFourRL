import time

import torch.multiprocessing as mp


class VectorizedEnv:
    def __init__(self, game_class, num_envs):
        self.envs = [game_class() for _ in range(num_envs)]
        self.num_workers = min(mp.cpu_count(), num_envs)

    def step_async(self, actions):
        with mp.Pool(self.num_workers) as p:
            t = time.time()
            result = p.starmap(_step_env, zip(self.envs, actions))
            print(time.time() - t)
            result = zip(*result)
        print(result)
        states, rewards, dones = result
        return list(states), list(rewards), list(dones)

    def step(self, actions):
        states, rewards, dones = [], [], []
        for action, env in zip(actions, self.envs):
            state, reward, done = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
        return states, rewards, dones

    def reset(self):
        return [env.reset() for env in self.envs]

    def render(self):
        return [env.render() for env in self.envs]


def _reset_env(env):
    return env.reset()


def _step_env(env, action):
    return env.step(action)
