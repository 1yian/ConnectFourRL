class ParallelEnv:
    def __init__(self, game_class, parallel_envs):
        self.envs = [game_class() for _ in range(parallel_envs)]

    def step(self, actions):
        rewards = []
        states = []
        dones = []
        for action, env in zip(actions, self.envs):
            state, reward, done = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        return states, rewards, dones

    def reset(self):
        return [env.reset() for env in self.envs]
