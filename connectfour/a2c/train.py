import random

import numpy as np

from connectfour.env.parallel_env import ParallelEnv


class TrainingSession:
    def __init__(self, game, agent, buffer):
        self.game = game
        self.agent = agent
        self.buffer = buffer

    def eval_vs_random(self, num_envs, num_episodes):
        score = {'agent': 0, 'random': 0}
        env = ParallelEnv(game_class=self.game, parallel_envs=num_envs)
        agent_is_playing = True
        states = env.reset()
        episode_count = 0
        while episode_count < num_episodes * num_envs:

            if agent_is_playing:
                valid_moves = [self.get_valid_moves(state) for state in states]
                _, actions = self.agent.select_actions(states, valid_moves, greedy=True)
            else:
                valid_moves = [self.get_valid_move_indices(state) for state in states]
                actions = np.array([random.choice(moves) for moves in valid_moves])

            new_states, rewards, dones = env.step(actions)
            for env_index in range(num_envs):
                if dones[env_index]:
                    if rewards[env_index] != 0:
                        if agent_is_playing:
                            score['agent'] += 1
                        else:
                            score['random'] += 1
                    else:
                        score['agent'] += 0.5
                        score['random'] += 0.5
                    episode_count += 1
            agent_is_playing = not agent_is_playing
            states = new_states
        return score

    def self_play_episodes(self, num_envs, num_episodes):
        p1_episode = [[] for _ in range(num_envs)]
        p2_episode = [[] for _ in range(num_envs)]

        env = ParallelEnv(game_class=self.game, parallel_envs=num_envs)

        current_player_episodes, opposing_player_episodes = p1_episode, p2_episode

        states = env.reset()
        episode_count = 0
        while episode_count < num_episodes * num_envs:
            valid_moves = [self.get_valid_moves(state) for state in states]
            processed_states, actions = self.agent.select_actions(states, valid_moves, greedy=False)
            new_states, rewards, dones = env.step(actions)
            experiences = [{'state': state, 'reward': reward, 'action': action, 'done': done}
                           for state, action, reward, done in zip(processed_states, actions, rewards, dones)]

            for env_index in range(num_envs):
                current_episode = current_player_episodes[env_index]
                other_episode = opposing_player_episodes[env_index]

                current_episode.append(experiences[env_index])

                reward = experiences[env_index]['reward']
                if reward != 0:
                    self.buffer.add(current_episode)
                    other_episode[-1]['reward'] = - reward
                    self.buffer.add(other_episode)

                    current_player_episodes[env_index] = []
                    opposing_player_episodes[env_index] = []

                    episode_count += 1

            states = new_states
            current_player_episodes, opposing_player_episodes = opposing_player_episodes, current_player_episodes

    def get_valid_moves(self, state):
        top_row = state[0]
        ohe_valid_moves = [0] * len(top_row)
        for i in range(len(top_row)):
            if top_row[i] == 0:
                ohe_valid_moves[i] = 1
        return ohe_valid_moves

    def get_valid_move_indices(self, state):
        top_row = state[0]
        move_indices = []
        for i in range(len(top_row)):
            if top_row[i] == 0:
                move_indices.append(i)
        return move_indices
