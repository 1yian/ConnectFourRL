from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from connectfour.a2c.agent import ActorCriticAgent
from connectfour.a2c.replay_buffer import ExperienceReplayBuffer, collate_experiences
from connectfour.a2c.train import TrainingSession
from connectfour.env.env import ConnectFour

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the training script')
    add_arg = parser.add_argument
    add_arg('--batch_size', help='Batch size when training on experience replay buffer', type=int, default=1024)
    add_arg('--iterations', help='The number of iterations to run the training algorithm', type=int, default=256)
    add_arg('--episodes', help='The number of episodes per iteration to self-play on', type=int, default=256)
    add_arg('--tensorboard', help='Whether to use tensorboard or not for logging', action="store_true", default=False)
    add_arg('--print', help="Whether to print the output of the training algorithm", action="store_true", default=False)
    add_arg('--checkpoint', help="The path to a checkpoint of a model to start training from", type=str, default='')
    add_arg('--save_every', help="How often the algorithm saves a checkpoint of the models", type=int, default=15)

    args = parser.parse_args()

    writer = SummaryWriter() if args.tensorboard else None

    agent = ActorCriticAgent(output=args.print, writer=writer)
    buffer = ExperienceReplayBuffer()

    training_sess = TrainingSession(ConnectFour, agent, buffer)
    if args.checkpoint:
        agent.load_checkpoint(args.checkpoint)
    parallel_envs = 12
    for iter in range(args.iterations):

        training_sess.self_play_episodes(parallel_envs, args.episodes // parallel_envs)
        loader = DataLoader(buffer, batch_size=args.batch_size, shuffle=True, num_workers=8,
                            collate_fn=collate_experiences)
        agent.train_on_loader(loader)
        buffer.clear()

        scores = training_sess.eval_vs_random(8, 50)
        if args.print:
            print(scores)
        if agent.writer:
            agent.writer.add_scalar('Score vs random', scores['Agent'], iter)

        if iter % 15 == 0:
            agent.save_checkpoint("./checkpoints/a2c_{}.pt".format(iter))
        agent.scheduler.step()
