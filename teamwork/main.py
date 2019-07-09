import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from egg import core
from egg.zoo.simple_autoenc.features import OneHotLoader
import neptune
from neptunecontrib.monitoring.utils import send_figure
import seaborn as sns
import matplotlib.pyplot as plt

from teamwork.wrappers import ReinforceMultiAgentWrapper


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--alphabet_size', type=int, default=5,
                        help='Alphabet size (default: 8)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1)')
    parser.add_argument('--receiver_population_size', type=int, default=2,
                        help='Receiver population size (default: 3)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=0.1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--executive_sender_entropy_coeff', type=float, default=1e-2,
                        help='The entropy regularisation coefficient for Executive sender (default: 1e-2)')
    parser.add_argument('--receivers_entropy_coeff', type=float, default=0.1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument('--sender_lr', type=float, default=1e-2,
                        help="Learning rate for Sender's parameters (default: 1e-2)")
    parser.add_argument('--executive_sender_lr', type=float, default=1e-2,
                        help="Learning rate for Executive sender's parameters (default: 1e-2)")
    parser.add_argument('--receiver_lr', type=float, default=1e-2,
                        help="Learning rate for Receiver's parameters (default: 1e-2)")
    parser.add_argument('--seed', type=int, default=17,
                        help="Random seed")
    parser.add_argument('--use_reinforce', type=bool, default=False,
                        help="Whether to use Reinforce or Gumbel-Softmax for optimizing sender and receiver."
                             "Executive receiver will be always optimized using Reinforce")
    args = core.init(parser)
    print(args)
    return args


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        return self.fc1(x)


class ExecutiveSender(nn.Module):
    def __init__(self, n_choices, n_features):
        super(ExecutiveSender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_choices)

    def forward(self, x):
        return self.fc1(x)


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Receiver, self).__init__()
        self.fc1 = core.RelaxedEmbedding(n_features, n_hidden)

    def forward(self, x, _input):
        return self.fc1(x)


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output == sender_input.argmax(dim=1)).detach().float().mean(dim=0)
    return -acc, {'acc': acc.item()}


def loss_diff(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}


class MultiAgentGame(nn.Module):

    def __init__(self, senders, executive_sender, receivers, loss,
                 sender_entropy_coeff=0,
                 receiver_entropy_coeff=0,
                 executive_sender_entropy_coeff=0):
        super(MultiAgentGame, self).__init__()
        self.senders = senders
        self.executive_sender = executive_sender
        self.receivers = receivers
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.executive_sender_entropy_coeff = executive_sender_entropy_coeff
        self.sender_entropy_coeff = sender_entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, sender_input, labels, receiver_input=None):
        id, executive_sender_log_prob, executive_sender_entropy = self.executive_sender(sender_input)
        receiver = self.receivers[id.item()]
        sender = self.senders[id.item()]

        if opts.use_reinforce:
            message, sender_log_prob, sender_entropy = sender(sender_input)
            receiver_output, receiver_log_prob, receiver_entropy = receiver(message, receiver_input)
        else:
            message = sender(sender_input)
            receiver_output = receiver(message, receiver_input)
            receiver_log_prob, sender_log_prob = 0, 0

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        advantage = (loss.detach() - self.mean_baseline)
        sender_loss = advantage * (sender_log_prob + receiver_log_prob)
        exec_sender_loss = advantage * (executive_sender_log_prob + receiver_log_prob)
        receiver_loss = advantage * receiver_log_prob
        policy_loss = (sender_loss + receiver_loss + exec_sender_loss).mean()

        if opts.use_reinforce:
            entropy_loss = -(
                sender_entropy.mean() * self.sender_entropy_coeff +
                receiver_entropy.mean() * self.receiver_entropy_coeff +
                executive_sender_entropy.mean() * self.executive_sender_entropy_coeff
            )
        else:
            entropy_loss = -executive_sender_entropy.mean() * self.executive_sender_entropy_coeff

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        rest_info['baseline'] = self.mean_baseline
        rest_info['loss'] = loss.mean().item()
        if opts.use_reinforce:
            rest_info['sender_entropy'] = sender_entropy.mean()
            rest_info['receiver_entropy'] = receiver_entropy.mean()
        rest_info['executive_sender_entropy'] = executive_sender_entropy.mean()
        return full_loss, rest_info


class NeptuneMonitor:

    def __init__(self, experiment, game):
        self.experiment = experiment
        self.game = game

    def log(self, mode, epoch, loss, rest):
        self.experiment.send_metric(f'{mode}_loss', loss)
        for metric, value in rest.items():
            self.experiment.send_metric(f'{mode}_{metric}', value)

        self.save_codebook(
            weight_list=[F.softmax(sender.agent.fc1.weight.detach(), dim=0).numpy()
                         for sender in self.game.senders],
            epoch=epoch,
            label='Sender softmax'
        )
        self.save_codebook(
            weight_list=[F.softmax(self.game.executive_sender.agent.fc1.weight.detach(), dim=0).numpy()],
            epoch=epoch,
            label='Executive sender softmax'
        )
        self.save_codebook(
            weight_list=[F.softmax(receiver.agent.fc1.weight.detach(), dim=1).numpy()
                         for receiver in self.game.receivers],
            epoch=epoch,
            label='Receiver softmax'
        )

        self.save_codebook(
            weight_list=[sender.agent.fc1.weight.detach().numpy()
                         for sender in self.game.senders],
            epoch=epoch,
            label='Sender'
        )
        self.save_codebook(
            weight_list=[self.game.executive_sender.agent.fc1.weight.detach().numpy()],
            epoch=epoch,
            label='Executive sender'
        )
        self.save_codebook(
            weight_list=[receiver.agent.fc1.weight.detach().numpy()
                         for receiver in self.game.receivers],
            epoch=epoch,
            label='Receiver'
        )

    def save_codebook(self, weight_list, epoch, label):
        figure, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 5))
        figure.suptitle(f'Epoch {epoch}')
        for i, (matrix, ax) in enumerate(zip(weight_list, axes)):
            g = sns.heatmap(matrix, annot=True, fmt='.2f', ax=ax)
            g.set_title(f'{label} {i}')
        send_figure(figure)
        plt.close()


class CustomTrainer(core.Trainer):

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            train_loss, train_rest = self.train_epoch()
            self.monitor.log('train', self.epoch, train_loss, train_rest)
            if epoch % self.validation_freq == 0:
                validation_loss, rest = self.eval()
                self.monitor.log('validation', epoch, validation_loss, rest)
                print(f'validation: epoch {epoch}, loss {validation_loss},  {rest}', flush=True)

    def train_epoch(self):

        def _add_dicts(a, b):
            result = dict(a)
            for k, v in b.items():
                result[k] = result.get(k, 0) + v
            return result

        def _div_dict(d, n):
            result = dict(d)
            for k in result:
                result[k] /= n
            return result

        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for i, batch in enumerate(self.train_data):
            if i % 10 == 0:
                self.optimizer.zero_grad()
            optimized_loss, rest = self.game(*batch)
            mean_rest = _add_dicts(mean_rest, rest)
            optimized_loss.backward()
            if i % 10 == 0:
                self.optimizer.step()
            n_batches += 1
            mean_loss += optimized_loss

        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss, mean_rest


if __name__ == "__main__":
    opts = get_params()
    train_loader = OneHotLoader(n_features=opts.n_features, batch_size=opts.batch_size,
                                batches_per_epoch=opts.batches_per_epoch)
    test_loader = OneHotLoader(n_features=opts.n_features, batch_size=opts.batch_size,
                               batches_per_epoch=opts.batches_per_epoch, seed=opts.seed)
    if opts.use_reinforce:
        senders = [core.ReinforceWrapper(Sender(opts.alphabet_size, opts.n_features))
                   for _ in range(opts.receiver_population_size)]
        receivers = [core.ReinforceWrapper(Receiver(opts.n_features, opts.alphabet_size))
                     for _ in range(opts.receiver_population_size)]
    else:
        senders = [core.GumbelSoftmaxWrapper(Sender(opts.alphabet_size, opts.n_features))
                   for _ in range(opts.receiver_population_size)]
        receivers = [core.GumbelSoftmaxWrapper(Receiver(opts.n_features, opts.alphabet_size))
                     for _ in range(opts.receiver_population_size)]
    executive_sender = core.ReinforceWrapper(ExecutiveSender(opts.receiver_population_size, opts.n_features))

    game = MultiAgentGame(senders, executive_sender, receivers,
                          loss if opts.use_reinforce else loss_diff,
                          opts.sender_entropy_coeff,
                          opts.executive_sender_entropy_coeff,
                          opts.receivers_entropy_coeff)
    sender_params = [{'params': sender.parameters(), 'lr': opts.sender_lr} for sender in senders]
    executive_sender_params = [{'params': executive_sender.parameters(), 'lr': opts.executive_sender_lr}]
    receivers_params = [{'params': receiver.parameters(), 'lr': opts.receiver_lr} for receiver in receivers]
    optimizer = torch.optim.Adam(sender_params + receivers_params + executive_sender_params)

    neptune.init('tomekkorbak/teamwork')
    with neptune.create_experiment(params=vars(opts), upload_source_files=[__file__]) as experiment:
        trainer = CustomTrainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader)
        trainer.monitor = NeptuneMonitor(experiment=experiment, game=game)
        trainer.train(n_epochs=opts.n_epochs)
