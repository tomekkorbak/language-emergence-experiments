import argparse
import os

import torch
from torch.utils.data import DataLoader
from egg import core
import neptune
from neptunecontrib.api.utils import get_filepaths

from compositionality2.agents import Sender, Receiver
from compositionality2.data import prepare_datasets
from compositionality2.games import loss
from compositionality2.callbacks import CompositionalityMetric, NeptuneMonitor


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=5,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')

    parser.add_argument('--sender_hidden', type=int, default=200,
                        help='Size of the hidden layer of Sender (default: 200)')
    parser.add_argument('--receiver_hidden', type=int, default=200,
                        help='Size of the hidden layer of Receiver (default: 200)')
    parser.add_argument('--sender_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')
    parser.add_argument('--rnn_cell', type=str, default='rnn')
    parser.add_argument('--sender_lr', type=float, default=0.0005,
                        help="Learning rate for Sender's parameters (default: 1e-3)")
    parser.add_argument('--receiver_lr', type=float, default=0.0005,
                        help="Learning rate for Receiver's parameters (default: 1e-3)")
    parser.add_argument('--sender_entropy_coeff', type=float, default=0.1)
    parser.add_argument('--receiver_entropy_coeff', type=float, default=0)
    parser.add_argument('--length_cost', type=float, default=0.05)

    parser.add_argument('--seed', type=int, default=171,
                        help="Random seed")
    parser.add_argument('--pretrain', type=bool, default=False,
                        help="")
    parser.add_argument('--config', type=str, default=None)

    args = core.init(parser)
    print(args)
    return args


if __name__ == "__main__":
    opts = get_params()
    train, test, full_dataset, test_targets = prepare_datasets()
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=opts.batch_size, drop_last=False, shuffle=False)
    sender = core.RnnSenderReinforce(
            agent=Sender(opts.n_features * 2, opts.sender_hidden),
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            force_eos=True,
            cell=opts.rnn_cell)
    receiver = core.RnnReceiverDeterministic(
            agent=Receiver(opts.n_features * 2, opts.receiver_hidden),
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.rnn_cell)

    neptune.init('tomekkorbak/template-transfer2')
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths(), tags=['buffled_berkeley']) as experiment:
        print(os.environ)
        compositional_game = core.SenderReceiverRnnReinforce(
            sender, receiver, loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=opts.receiver_entropy_coeff,
            length_cost=opts.length_cost)
        # compositional_game = CompositionalGameReinforce(sender, receiver, loss)
        sender_params = [{'params': sender.parameters(), 'lr': opts.sender_lr}]
        receiver_params = [{'params': receiver.parameters(), 'lr': opts.receiver_lr}]
        optimizer = torch.optim.Adam(sender_params + receiver_params)
        trainer = core.Trainer(game=compositional_game, optimizer=optimizer, train_data=train_loader,
                               validation_data=test_loader,
                               callbacks=[
                                   CompositionalityMetric(full_dataset, opts, test_targets=test_targets),
                                   NeptuneMonitor(train_freq=100, test_freq=1),
                                   core.ConsoleLogger(print_train_loss=not os.environ.get('SLURM_JOB_NAME')),
                               ])
        trainer.train(n_epochs=opts.n_epochs)
