from torch.utils.data.dataloader import DataLoader

from connectfour.a2c.agent import ActorCriticAgent
from connectfour.a2c.replay_buffer import ExperienceReplayBuffer, collate_experiences
from connectfour.a2c.train import TrainingSession
from connectfour.env.env import ConnectFour

agent = ActorCriticAgent(output=True)
buffer = ExperienceReplayBuffer()
training_sess = TrainingSession(ConnectFour, agent, buffer)

for i in range(128):
    training_sess.self_play_episodes(8, 128)
    loader = DataLoader(buffer, batch_size=256, shuffle=True, num_workers=4, collate_fn=collate_experiences,
                        pin_memory=True)
    agent.train_on_loader(loader)
    buffer.clear()
