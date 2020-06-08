from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from connectfour.a2c.agent import ActorCriticAgent
from connectfour.a2c.replay_buffer import ExperienceReplayBuffer, collate_experiences
from connectfour.a2c.train import TrainingSession
from connectfour.env.env import ConnectFour

if __name__ == '__main__':
    writer = SummaryWriter()
    agent = ActorCriticAgent(output=True, writer=False)
    buffer = ExperienceReplayBuffer()
    training_sess = TrainingSession(ConnectFour, agent, buffer)

    for i in range(2048):
        training_sess.self_play_episodes(4, 90)
        loader = DataLoader(buffer, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_experiences,
                            pin_memory=True)
        agent.train_on_loader(loader)
        buffer.clear()
        if i % 25 == 0:
            agent.save_checkpoint("./checkpoints/a2c_{}.pt".format(i))
        scores = training_sess.eval_vs_random(4, 25)
        print(scores)
        if agent.writer:
            agent.writer.add_scalar('Score vs random', scores['Agent'], i)
        agent.scheduler.step()
