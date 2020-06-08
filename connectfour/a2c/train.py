import random
import numpy as np
from connectfour.env.vectorized_env import VectorizedEnv


class TrainingSession:
    def __init__(self, game, agent, buffer):
        self.game = game
        self.agent = agent
        self.buffer = buffer

    def eval_vs_random(self, num_envs, episodes_per_env):
        max_episodes = episodes_per_env * num_envs
        env = VectorizedEnv(game_class=self.game, num_envs=num_envs)

        score = {'Agent': 0, 'Random': 0}
        agent_is_playing = True

        states = env.reset()
        episode_count = 0
        while episode_count < max_episodes:

            if agent_is_playing:
                actions, *_ = self.agent.select_actions(states, greedy=True)
            else:
                actions = [self._get_rand_move(state) for state in states]

            new_states, rewards, dones = env.step(actions)
            for env_index in range(num_envs):
                if dones[env_index]:
                    if rewards[env_index] != 0:
                        winner = 'Agent' if agent_is_playing else 'Random'
                        score[winner] += 1
                    else:
                        score['Agent'] += 0.5
                        score['Random'] += 0.5
                    episode_count += 1

            agent_is_playing = not agent_is_playing
            states = new_states
        return score

    def self_play_episodes(self, num_envs, episodes_per_env):
        env = VectorizedEnv(game_class=self.game, num_envs=num_envs)
        states = env.reset()

        p1_episodes, p2_episodes = [[] for _ in range(num_envs)], [[] for _ in range(num_envs)]
        current_episodes, opponent_episodes = p1_episodes, p2_episodes

        episode_count = 0
        total_episodes = (episodes_per_env * num_envs)
        while episode_count < total_episodes:
            actions, data = self.agent.select_actions(states, greedy=False)
            states, action_probs, valid_moves = data['processed_states'], data['action_probs'], data['valid_moves']

            new_states, rewards, dones = env.step(actions)

            exp_data = zip(states, actions, rewards, dones, action_probs, valid_moves)
            experiences = self._unpack_experience_data(exp_data)

            for env_index in range(num_envs):
                current_episodes[env_index].append(experiences[env_index])

                if experiences[env_index]['done']:
                    # Mirror the terminal reward since connect four is symmetrical
                    reward = experiences[env_index]['reward']
                    opponent_episodes[env_index][-1]['reward'] = - reward

                    self.buffer.add(current_episodes[env_index])
                    self.buffer.add(opponent_episodes[env_index])

                    current_episodes[env_index] = []
                    opponent_episodes[env_index] = []

                    episode_count += 1

            states = new_states
            current_episodes, opponent_episodes = opponent_episodes, current_episodes

    @staticmethod
    def _unpack_experience_data(raw_exp_data):
        experiences = []
        for raw_exp in raw_exp_data:
            state, action, reward, done, action_prob, valid_moves = raw_exp
            exp_dict = {
                'state': state,
                'action': action,
                'reward': reward,
                'done': done,
                'action_prob': action_prob,
                'valid_moves': valid_moves,
            }
            experiences.append(exp_dict)
        return experiences

    def _get_rand_move(self, state):
        valid_moves = self._get_valid_move_indices(state)
        return random.choice(valid_moves)

    @staticmethod
    def _get_valid_move_indices(state):
        top_row = state[0]
        move_indices = []
        for i in range(len(top_row)):
            if top_row[i] == 0:
                move_indices.append(i)
        return move_indices
