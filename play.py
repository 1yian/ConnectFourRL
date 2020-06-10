from connectfour.a2c.agent import ActorCriticAgent
from connectfour.a2c.train import TrainingSession

from connectfour.env.env import ConnectFour

if __name__ == '__main__':
    agent = ActorCriticAgent(output=True, writer=False)
    agent.load_checkpoint('checkpoints/a2c_120.pt')
    training_sess = TrainingSession(ConnectFour, agent, None)
    training_sess.eval_vs_person(agent, True, greedy=True)
