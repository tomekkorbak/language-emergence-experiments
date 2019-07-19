import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from egg import core
import neptune
from neptunecontrib.api.utils import get_filepaths

from teamwork.wrappers import GumbelSoftmaxMultiAgentEnsemble, GSSequentialTeamworkGame
from teamwork.callbacks import NeptuneMonitor, Dump
from teamwork.agents import Sender, ExecutiveSender, Receiver
from teamwork.data import TupleDataset


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_attributes', type=int, default=2,
                        help='Number of attributes (default: 2')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1)')
    parser.add_argument('--population_size', type=int, default=2,
                        help='Population size (default: 3)')

    parser.add_argument('--sender_hidden', type=int, default=200,
                        help='Size of the hidden layer of Sender (default: 200)')
    parser.add_argument('--receiver_hidden', type=int, default=200,
                        help='Size of the hidden layer of Receiver (default: 200)')
    parser.add_argument('--sender_embedding', type=int, default=5,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 5)')
    parser.add_argument('--receiver_embedding', type=int, default=5,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 5)')
    parser.add_argument('--executive_sender_entropy_coeff', type=float, default=1e-2,
                        help='The entropy regularisation coefficient for Executive sender (default: 1e-2)')

    parser.add_argument('--sender_lr', type=float, default=0.001,
                        help="Learning rate for Sender's parameters (default: 1e-2)")
    parser.add_argument('--executive_sender_lr', type=float, default=0.001,
                        help="Learning rate for Executive sender's parameters (default: 1e-2)")
    parser.add_argument('--receiver_lr', type=float, default=0.001,
                        help="Learning rate for Receiver's parameters (default: 1e-2)")
    parser.add_argument('--seed', type=int, default=117,
                        help="Random seed")
    parser.add_argument('--use_reinforce', type=bool, default=False,
                        help="Whether to use Reinforce or Gumbel-Softmax for optimizing sender and receiver."
                             "Executive receiver will be always optimized using Reinforce")
    parser.add_argument('--config', type=str, default=None)

    args = core.init(parser)
    print(args)
    return args


def loss_diff(target, receiver_output, idx):
    acc = (receiver_output.argmax(dim=1) == target).detach().float().mean(dim=0)
    loss = F.cross_entropy(receiver_output, target, reduction="none")
    return loss, {f'accuracy_{idx}': acc.item()}


if __name__ == "__main__":
    opts = get_params()
    perceptual_dimensions = [opts.n_features for _ in range(opts.n_attributes)]
    train, dev = TupleDataset.create_train_and_dev(perceptual_dimensions=perceptual_dimensions)
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(dev, batch_size=opts.batch_size, drop_last=True, shuffle=False)

    senders = [
        core.RnnSenderGS(
            agent=Sender(opts.sender_hidden, opts.n_features, opts.n_attributes),
            vocab_size=opts.vocab_size,
            emb_dim=opts.sender_embedding,
            n_hidden=opts.sender_hidden,
            max_len=opts.max_len,
            temperature=1)
        for _ in range(opts.population_size)]
    sender_ensemble = GumbelSoftmaxMultiAgentEnsemble(agents=senders)
    receivers_1 = [
        core.RnnReceiverGS(
            agent=Receiver(opts.receiver_hidden, 100, 2),
            vocab_size=opts.vocab_size,
            emb_dim=opts.receiver_embedding,
            n_hidden=opts.receiver_hidden)
        for _ in range(opts.population_size)]
    receiver_ensemble_1 = GumbelSoftmaxMultiAgentEnsemble(agents=receivers_1)
    receivers_2 = [
        core.RnnReceiverGS(
            agent=Receiver(opts.receiver_hidden, 100, 2),
            vocab_size=opts.vocab_size,
            emb_dim=opts.receiver_embedding,
            n_hidden=opts.receiver_hidden)
        for _ in range(opts.population_size)]
    receiver_ensemble_2 = GumbelSoftmaxMultiAgentEnsemble(agents=receivers_2)
    executive_sender = core.ReinforceWrapper(ExecutiveSender(opts.population_size, opts.n_features, opts.n_attributes))

    game = GSSequentialTeamworkGame(sender_ensemble, receiver_ensemble_1, receiver_ensemble_2, executive_sender, loss_diff,
                                    opts.executive_sender_entropy_coeff)
    sender_params = [{'params': sender_ensemble.parameters(), 'lr': opts.sender_lr}]
    executive_sender_params = [{'params': executive_sender.parameters(), 'lr': opts.executive_sender_lr}]
    receivers_params = [{'params': receiver_ensemble_1.parameters(), 'lr': opts.receiver_lr},
                        {'params': receiver_ensemble_2.parameters(), 'lr': opts.receiver_lr}]
    optimizer = torch.optim.Adam(sender_params + receivers_params + executive_sender_params)

    neptune.init('tomekkorbak/compositionality')
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths()) as experiment:
        trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader,
                               callbacks=[Dump(experiment, test_loader),
                                          NeptuneMonitor(experiment=experiment),
                                          core.ConsoleLogger(print_train_loss=True)])
        trainer.train(n_epochs=opts.n_epochs)
