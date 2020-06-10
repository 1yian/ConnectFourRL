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
    agent.load_checkpoint('checkpoints/a2c_75.pt')

    for i in range(76, 2560):
        training_sess.self_play_episodes(64, 512)
        loader = DataLoader(buffer, batch_size=4096, shuffle=True, num_workers=8, collate_fn=collate_experiences,
                            pin_memory=True)
        agent.train_on_loader(loader)
        buffer.clear()
        if i % 15 == 0:
            agent.save_checkpoint("./checkpoints/a2c_{}.pt".format(i))
        scores = training_sess.eval_vs_random(8, 50)
        print(scores)
        if agent.writer:
            agent.writer.add_scalar('Score vs random', scores['Agent'], i)
        agent.scheduler.step()
