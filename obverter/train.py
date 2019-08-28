import argparse
import os
import random

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from egg import core
import neptune
from neptunecontrib.api.utils import get_filepaths

from compositionality import data
from visual_compositionality import visual_data
from visual_compositionality.pretrain import Vision
from obverter.agent import Agent, AgentWrapper
from obverter.callbacks import CompositionalityMetricObverter, NeptuneMonitor, EarlyStopperAccuracy



def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=5,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_attributes', type=int, default=2,
                        help='Number of attributes (default: 2')
    parser.add_argument('--seed', type=int, default=171,
                        help="Random seed")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--visual', action='store_true', default=False,
                        help="Use visual input instead of one-hot vectors")

    # Agent architecture
    parser.add_argument('--sender_hidden', type=int, default=200,
                        help='Size of the hidden layer of Sender (default: 200)')
    parser.add_argument('--receiver_hidden', type=int, default=200,
                        help='Size of the hidden layer of Receiver (default: 200)')
    parser.add_argument('--sender_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 50)')
    parser.add_argument('--receiver_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 50)')
    parser.add_argument('--rnn_cell', type=str, default='rnn')
    parser.add_argument('--sender_lr', type=float, default=0.0002,
                        help="Learning rate for Sender's parameters (default: 1e-3)")
    parser.add_argument('--receiver_lr', type=float, default=0.0002,
                        help="Learning rate for Receiver's parameters (default: 1e-3)")

    args = core.init(parser)
    print(args)
    return args


def entangled_loss(targets, receiver_output_1, receiver_output_2):
    acc_1 = (receiver_output_1.argmax(dim=1) == targets[:, 0]).detach().float()
    acc_2 = (receiver_output_2.argmax(dim=1) == targets[:, 1]).detach().float()
    loss_1 = F.cross_entropy(receiver_output_1, targets[:, 0], reduction="none")
    loss_2 = F.cross_entropy(receiver_output_2, targets[:, 1], reduction="none")
    acc = (acc_1 * acc_2).mean(dim=0)
    loss = loss_1 + loss_2
    return loss, {f'accuracy': acc.item(),
                  f'first_accuracy': acc_1.mean(dim=0).item(),
                  f'second_accuracy': acc_2.mean(dim=0).item()}


class ObverterGame(nn.Module):

    def __init__(self, agents, max_len, vocab_size, loss):
        super(ObverterGame, self).__init__()
        self.agents = agents
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.loss = loss

    def forward(self, sender_input, target):
        sender, receiver = random.sample(self.agents, k=2)
        with torch.no_grad():
            message = sender.decode(sender_input)
        output_1, output_2 = receiver(message)
        loss, logs = self.loss(target, output_1, output_2)
        return loss.mean(), logs


if __name__ == "__main__":
    opts = get_params()
    opts.on_slurm = os.environ.get('SLURM_JOB_NAME', False)
    core.util._set_seed(opts.seed)
    if opts.visual:
        full_dataset, train, test = visual_data.prepare_datasets(opts.n_features, opts.n_attributes)
    else:
        full_dataset, train, test = data.prepare_datasets(opts.n_features, opts.n_attributes)
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=opts.batch_size, drop_last=False, shuffle=False)

    agents = [AgentWrapper(
            agent=Agent(opts.receiver_hidden, opts.n_features),
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.rnn_cell,
            obverter_loss=entangled_loss,
            vision_module=Vision.from_pretrained('visual_compositionality/vision_model.pth') if opts.visual else None
    ) for _ in range(2)]
    game = ObverterGame(agents=agents, max_len=2, vocab_size=opts.vocab_size, loss=entangled_loss)
    optimizer = torch.optim.Adam([{'params': agent.parameters(), 'lr': opts.lr} for agent in agents])
    neptune.init('tomekkorbak/obverter2')
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths(), tags=['blunt_bishop']) as experiment:
        trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                               validation_data=test_loader,
                               callbacks=[
                                   CompositionalityMetricObverter(full_dataset, agents[0], opts, opts.vocab_size, prefix='1_'),
                                   CompositionalityMetricObverter(full_dataset, agents[1], opts, opts.vocab_size, prefix='2_'),
                                   NeptuneMonitor(),
                                   core.ConsoleLogger(print_train_loss=not opts.on_slurm),
                                   EarlyStopperAccuracy(threshold=0.98, field_name='accuracy', delay=2)
                               ])
        trainer.train(n_epochs=opts.n_epochs)
