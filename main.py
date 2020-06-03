from torch.utils.data.dataloader import DataLoader

from connectfour.a2c.agent import ActorCriticAgent
from connectfour.a2c.replay_buffer import ExperienceReplayBuffer, collate_experiences
from connectfour.a2c.train import TrainingSession
from connectfour.env.env import ConnectFour

agent = ActorCriticAgent(output=True)
buffer = ExperienceReplayBuffer()
training_sess = TrainingSession(ConnectFour, agent, buffer)

for i in range(2048):
    training_sess.self_play_episodes(4, 1024)
    loader = DataLoader(buffer, batch_size=4096, shuffle=True, num_workers=8, collate_fn=collate_experiences,
                        pin_memory=True)
    agent.train_on_loader(loader)
    buffer.clear()
    agent.save_checkpoint("./a2c_{}.pt".format(i))
    print(training_sess.eval_vs_random(4, 25))
