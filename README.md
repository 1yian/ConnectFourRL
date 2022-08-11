## ConnectFourRL
### Codebase for a reinforcement learning based approach to creating a Connect Four AI


## Project Status

The project is currently in development. Right now I have implemented a variant of off-policy actor-critic, and pUCT Monte-Carlo Tree Search (MCTS) to use the actor-critic predictions as the basis of a lookahead search. This use of MCTS is used in a similar way to AlphaZero. My future work involves:

  - Refactoring the codebase
  - Implementing a variant of the AlphaZero algorithm
  - Implementing DQN, and various algorithms
  - Writing a survey paper on various RL approaches in this environment

To test my work so far, clone down this repository. You will need an environment with `torch`, `numpy`, and `tensorboard` installed.

Try `python3 test.py`.

A graphical interface for the game should pop up, where you can then play against my agent to see how well it performs against you in ConnectFour.

