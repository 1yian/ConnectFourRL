from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from connectfour.a2c.agent import ActorCriticAgent
from connectfour.a2c.replay_buffer import ExperienceReplayBuffer, collate_experiences
from connectfour.a2c.train import TrainingSession
from connectfour.env.env import ConnectFour

if __name__ == '__main__':

    writer = SummaryWriter()
    agent = ActorCriticAgent(output=True, writer=True)
    buffer = ExperienceReplayBuffer()
    training_sess = TrainingSession(ConnectFour, agent, buffer)
    agent.load_checkpoint('checkpoints/a2c_135.pt')
    import cProfile
    pr = cProfile.Profile()
    pr.run('training_sess.self_play_episodes(24, 64)')
    pr.print_stats(sort='time')