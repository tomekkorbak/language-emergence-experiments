import argparse
import os
import random
from itertools import product

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from egg import core
from egg.core.rnn import RnnEncoder
import neptune
from neptunecontrib.api.utils import get_filepaths

from compositionality.data import prepare_datasets
from obverter.callbacks import CompositionalityMetricObverter, NeptuneMonitor, EarlyStopperAccuracy


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=5,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_attributes', type=int, default=2,
                        help='Number of attributes (default: 2')
    parser.add_argument('--seed', type=int, default=171,
                        help="Random seed")
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help="")
    parser.add_argument('--config', type=str, default=None)

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
    parser.add_argument('--sender_entropy_coeff', type=float, default=0.1)
    parser.add_argument('--receiver_entropy_coeff', type=float, default=0)
    parser.add_argument('--length_cost', type=float, default=0.02)

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


class Agent(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features * 2)
        self.fc2_1 = nn.Linear(n_features * 2, n_features)
        self.fc2_2 = nn.Linear(n_features * 2, n_features)

    def forward(self, message, _):
        hidden = torch.nn.functional.leaky_relu(self.fc1(message))
        return self.fc2_1(hidden).squeeze(dim=0), self.fc2_2(hidden).squeeze(dim=0)


class AgentWrapper(nn.Module):

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1, obverter_loss=None):
        super(AgentWrapper, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)
        self.vocab_size = vocab_size
        self.obverter_loss = obverter_loss
        self.max_len = 2

    def forward(self, message):
        encoded = self.encoder(message)
        return self.agent(encoded, None)

    def decode(self, sender_input):
        message_dimensions = [range(self.vocab_size)] * self.max_len
        all_possible_messages = torch.LongTensor(list(product(*message_dimensions)))
        prediction_1, prediction_2 = self(all_possible_messages)
        messages_to_send = []
        for input in sender_input:
            proxy_target = input.argmax(dim=1).unsqueeze(dim=0).repeat(all_possible_messages.size(0), 1)
            proxy_loss, _ = self.obverter_loss(proxy_target, prediction_1, prediction_2)
            best_message = all_possible_messages[proxy_loss.argmin()]
            messages_to_send.append(best_message)
        return torch.stack(messages_to_send, dim=0)


class ObverterGame(nn.Module):

    def __init__(self, agents, max_len, vocab_size, loss):
        super(ObverterGame, self).__init__()
        self.agents = agents
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.loss = loss

    def forward(self, sender_input, target):
        sender, receiver = random.sample(self.agents, k=2)
        message = sender.decode(sender_input)
        output_1, output_2 = receiver(message)
        loss, logs = self.loss(target, output_1, output_2)
        return loss.mean(), logs


if __name__ == "__main__":
    opts = get_params()
    opts.on_slurm = os.environ.get('SLURM_JOB_NAME', False)

    full_dataset, train, test = prepare_datasets(5, 2)
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=opts.batch_size, drop_last=False, shuffle=False)

    agents = [AgentWrapper(
            agent=Agent(opts.receiver_hidden, opts.n_features),
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.rnn_cell,
            obverter_loss=entangled_loss
    ) for _ in range(2)]
    game = ObverterGame(agents=agents, max_len=2, vocab_size=opts.vocab_size, loss=entangled_loss)
    optimizer = torch.optim.Adam([{'params': agent.parameters(), 'lr': opts.lr} for agent in agents])
    neptune.init('tomekkorbak/obverter2')
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths(), tags=['awesome_anscombe']) as experiment:
        trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                               validation_data=test_loader,
                               callbacks=[
                                   CompositionalityMetricObverter(full_dataset, agents[0], opts, opts.vocab_size, test.indices, prefix='1_'),
                                   CompositionalityMetricObverter(full_dataset, agents[1], opts, opts.vocab_size, test.indices, prefix='2_'),
                                   NeptuneMonitor(),
                                   core.ConsoleLogger(print_train_loss=not opts.on_slurm),
                                   EarlyStopperAccuracy(threshold=0.99, field_name='accuracy', delay=5)
                               ])
        trainer.train(n_epochs=opts.n_epochs)
