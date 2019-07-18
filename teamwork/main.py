import argparse

import torch
import torch.nn.functional as F
from egg import core
from egg.zoo.simple_autoenc.features import OneHotLoader
import neptune
from neptunecontrib.api.utils import get_filepaths

from teamwork.wrappers import GumbelSoftmaxMultiAgentEnsemble,  GSSequentialTeamworkGame
from teamwork.callbacks import NeptuneMonitor
from teamwork.agents import Sender, ExecutiveSender, Receiver


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
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


def loss_diff(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float().mean(dim=0)
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc.item()}


if __name__ == "__main__":
    opts = get_params()
    train_loader = OneHotLoader(n_features=opts.n_features, batch_size=opts.batch_size,
                                batches_per_epoch=opts.batches_per_epoch)
    test_loader = OneHotLoader(n_features=opts.n_features, batch_size=opts.batch_size,
                               batches_per_epoch=opts.batches_per_epoch, seed=opts.seed)
    senders = [
        core.RnnSenderGS(
            agent=Sender(opts.sender_hidden, opts.n_features),
            vocab_size=opts.vocab_size,
            emb_dim=opts.sender_embedding,
            n_hidden=opts.sender_hidden,
            max_len=opts.max_len,
            temperature=1)
        for _ in range(opts.population_size)]
    sender_ensemble = GumbelSoftmaxMultiAgentEnsemble(agents=senders)
    receivers = [
        core.RnnReceiverGS(
            agent=Receiver(opts.n_features, opts.receiver_hidden),
            vocab_size=opts.vocab_size,
            emb_dim=opts.receiver_embedding,
            n_hidden=opts.receiver_hidden)
        for _ in range(opts.population_size)]
    receiver_ensemble = GumbelSoftmaxMultiAgentEnsemble(agents=receivers)
    executive_sender = core.ReinforceWrapper(ExecutiveSender(opts.population_size, opts.n_features))

    game = GSSequentialTeamworkGame(sender_ensemble, receiver_ensemble, executive_sender, loss_diff,
                                    opts.executive_sender_entropy_coeff)
    sender_params = [{'params': sender_ensemble.parameters(), 'lr': opts.sender_lr}]
    executive_sender_params = [{'params': executive_sender.parameters(), 'lr': opts.executive_sender_lr}]
    receivers_params = [{'params': receiver_ensemble.parameters(), 'lr': opts.receiver_lr}]
    optimizer = torch.optim.Adam(sender_params + receivers_params + executive_sender_params)

    neptune.init('tomekkorbak/teamwork')
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths(), tags=['grid5']) as experiment:
        trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader,
                               callbacks=[NeptuneMonitor(experiment=experiment), core.ConsoleLogger()])
        trainer.train(n_epochs=opts.n_epochs)
