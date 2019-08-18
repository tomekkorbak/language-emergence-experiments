import argparse
import os

import torch
from torch.utils.data import DataLoader
from egg import core
import neptune
from neptunecontrib.api.utils import get_filepaths

from compositionality.games import PretrainingmGameReinforce, PretrainingmGameGS, CompositionalGameGS, CompositionalGameReinforce, InputNoiseInjector
from compositionality.wrappers import RnnReceiverDeterministic
from compositionality.callbacks import NeptuneMonitor, EarlyStopperAccuracy, CompositionalityMetricReinforce, CompositionalityMetricGS
from compositionality.agents import Sender, Receiver
from compositionality.data import prepare_datasets


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=5,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_attributes', type=int, default=2,
                        help='Number of attributes (default: 2')
    parser.add_argument('--sender_hidden', type=int, default=200,
                        help='Size of the hidden layer of Sender (default: 200)')
    parser.add_argument('--receiver_hidden', type=int, default=200,
                        help='Size of the hidden layer of Receiver (default: 200)')
    parser.add_argument('--sender_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')
    parser.add_argument('--rnn_cell', type=str, default='rnn')
    parser.add_argument('--sender_lr', type=float, default=0.001,
                        help="Learning rate for Sender's parameters (default: 1e-3)")
    parser.add_argument('--receiver_lr', type=float, default=0.001,
                        help="Learning rate for Receiver's parameters (default: 1e-3)")
    parser.add_argument('--seed', type=int, default=171,
                        help="Random seed")
    parser.add_argument('--pretrain', action='store_true', default=False, help="")
    parser.add_argument('--reinforce', action='store_true', default=False, help="")

    parser.add_argument('--noise_strategy', type=str, default='full_permutation',
                        help="")
    parser.add_argument('--config', type=str, default=None)

    args = core.init(parser)
    print(args)
    return args


if __name__ == "__main__":
    opts = get_params()
    opts.on_slurm = os.environ.get('SLURM_JOB_NAME', False)
    full_dataset, train, test = prepare_datasets(opts.n_features, opts.n_attributes)
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=10, drop_last=False, shuffle=False)

    if opts.reinforce:
        pretrained_senders = [
            core.RnnSenderReinforce(
                agent=Sender(opts.sender_hidden, opts.n_features, opts.n_attributes),
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_embedding,
                hidden_size=opts.sender_hidden,
                max_len=1,
                cell=opts.rnn_cell,
                force_eos=False
            )
            for i in range(2)]
        sender_3 = core.RnnSenderReinforce(
            agent=Sender(opts.sender_hidden, opts.n_features, opts.n_attributes),
            vocab_size=opts.vocab_size * 2,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            force_eos=False,
            cell=opts.rnn_cell)
        receiver = RnnReceiverDeterministic(
            agent=Receiver(opts.receiver_hidden, opts.n_features, opts.n_attributes),
            vocab_size=opts.vocab_size * 2,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.rnn_cell)
    else:
        pretrained_senders = [
            core.RnnSenderGS(
                agent=Sender(opts.sender_hidden, opts.n_features, opts.n_attributes),
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_embedding,
                hidden_size=opts.sender_hidden,
                max_len=1,
                temperature=3.,
                trainable_temperature=True,
                cell=opts.rnn_cell,
                force_eos=False
            )
            for i in range(2)]
        sender_3 = core.RnnSenderGS(
                agent=Sender(opts.sender_hidden, opts.n_features, opts.n_attributes),
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_embedding,
                hidden_size=opts.sender_hidden,
                max_len=opts.max_len,
                temperature=3.,
                trainable_temperature=True,
                force_eos=False,
                cell=opts.rnn_cell)
        receiver = core.RnnReceiverGS(
                agent=Receiver(opts.receiver_hidden, opts.n_features, opts.n_attributes),
                vocab_size=opts.vocab_size,
                embed_dim=opts.receiver_embedding,
                hidden_size=opts.receiver_hidden,
                cell=opts.rnn_cell)

    neptune.init('tomekkorbak/template-transfer')
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths(), tags=['awesome_anscombe']) as experiment:
        CompositionalityMetric = CompositionalityMetricReinforce if opts.reinforce else CompositionalityMetricGS

        # Pretraining game
        if opts.pretrain:
            pretraining_game = PretrainingmGameReinforce(pretrained_senders, receiver, InputNoiseInjector(opts.noise_strategy)) if opts.reinforce else PretrainingmGameGS(pretrained_senders, receiver, InputNoiseInjector(opts.noise_strategy))
            sender_params = [{'params': sender.parameters(), 'lr': opts.sender_lr} for sender in pretrained_senders]
            receiver_params = [{'params': receiver.parameters(), 'lr': opts.receiver_lr}]
            optimizer = torch.optim.Adam(sender_params + receiver_params)
            trainer = core.Trainer(
                game=pretraining_game, optimizer=optimizer, train_data=train_loader,
                validation_data=test_loader,
                callbacks=[
                    CompositionalityMetric(full_dataset, pretrained_senders[0], opts, 10, test.indices, prefix='1_'),
                    CompositionalityMetric(full_dataset, pretrained_senders[1], opts, 10, test.indices, prefix='2_'),
                    NeptuneMonitor(prefix='pretrain'),
                    core.ConsoleLogger(print_train_loss=not opts.on_slurm),
                    EarlyStopperAccuracy(threshold=0.95, field_name='accuracy'),
                ])
            trainer.train(n_epochs=500_000)
            pretraining_game.train(False)

        # Compositional game
        assert sender_3.training
        receiver.train(not opts.pretrain)
        compositional_game = CompositionalGameReinforce(sender_3, receiver) if opts.reinforce else CompositionalGameGS(sender_3, receiver)
        sender_params = [{'params': sender_3.parameters(), 'lr': opts.sender_lr}]
        receiver_params = [{'params': receiver.parameters(), 'lr': opts.receiver_lr}]
        optimizer = torch.optim.Adam(sender_params + receiver_params)
        trainer = core.Trainer(game=compositional_game, optimizer=optimizer, train_data=train_loader,
                               validation_data=test_loader,
                               callbacks=[
                                   CompositionalityMetric(full_dataset, sender_3, opts, 20, test.indices, prefix='comp'),
                                   NeptuneMonitor(prefix='comp'),
                                   core.ConsoleLogger(print_train_loss=not opts.on_slurm),
                               ])
        trainer.train(n_epochs=opts.n_epochs)
