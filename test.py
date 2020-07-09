from torch.utils.tensorboard import SummaryWriter

from connectfour.a2c.agent import ActorCriticAgent
from connectfour.a2c.replay_buffer import ExperienceReplayBuffer
from connectfour.a2c.train import TrainingSession
from connectfour.env.env import ConnectFour
from connectfour.mcts.mcts import MCTSNode, RandomEvaluator, RandomPredictor, A2C


if __name__ == '__main__':

    agent = ActorCriticAgent(output=False, writer=False)
    agent.load_checkpoint('./checkpoints/a2c_240.pt')
    env = ConnectFour()
    agent_is_playing=False
    node = None
    state = env.reset()
    while True:
        print(env.render())
        if agent_is_playing:

            #action = agent.select_actions([state], greedy=True)[0]
            action = int(input("Which column do you want to move? (0-6): "))
        else:
            node = MCTSNode(env, A2C(), A2C())
            node, action = node.mcts(2000, True)
        state, reward, done = env.step(action)

        agent_is_playing = not agent_is_playing
        if done:
            break
