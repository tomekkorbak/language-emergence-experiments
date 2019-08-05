import argparse

import torch
from torch.utils.data import DataLoader
from egg import core
import neptune
from neptunecontrib.api.utils import get_filepaths

from compositionality.games import PretrainingmGame, CompositionalGame, loss_diff, RnnReceiverGS
from compositionality.callbacks import NeptuneMonitor, CompositionalityMetric, CompositionalityMetricPretraining2, CompositionalityMetricPretraining3, EarlyStopperAccuracy
from compositionality.agents import Sender, Receiver
from compositionality.data import prepare_datasets


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_attributes', type=int, default=2,
                        help='Number of attributes (default: 2')
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
    parser.add_argument('--sender_lr', type=float, default=0.0001,
                        help="Learning rate for Sender's parameters (default: 1e-3)")
    parser.add_argument('--receiver_lr', type=float, default=0.0001,
                        help="Learning rate for Receiver's parameters (default: 1e-3)")
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
    full_dataset, train, test = prepare_datasets(opts.n_features, opts.n_attributes)
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=10, drop_last=False, shuffle=False)

    sender_1, sender_2 = [
        core.RnnSenderGS(
            agent=Sender(opts.sender_hidden, opts.n_features, opts.n_attributes),
            vocab_size=opts.vocab_size,
            emb_dim=opts.sender_embedding,
            n_hidden=opts.sender_hidden,
            max_len=1,
            temperature=3.,
            trainable_temperature=True,
            cell=opts.rnn_cell)
        for i in range(2)]
    sender_3 = core.RnnSenderGS(
            agent=Sender(opts.sender_hidden, opts.n_features, opts.n_attributes),
            vocab_size=opts.vocab_size,
            emb_dim=opts.sender_embedding,
            n_hidden=opts.sender_hidden,
            max_len=2,
            temperature=3.,
            trainable_temperature=True,
            cell=opts.rnn_cell)
    receiver = RnnReceiverGS(
            agent=Receiver(opts.receiver_hidden, opts.n_features, opts.n_attributes),
            vocab_size=opts.vocab_size,
            emb_dim=opts.receiver_embedding,
            n_hidden=opts.receiver_hidden,
            cell=opts.rnn_cell)

    neptune.init('tomekkorbak/template-transfer')
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths(), tags=[]) as experiment:

        # Pretraining game
        if opts.pretrain:
            pretrained_senders = [sender_1, sender_2]
            pretraining_game = PretrainingmGame(pretrained_senders, receiver, loss_diff)
            sender_params = [{'params': sender.parameters(), 'lr': opts.sender_lr} for sender in pretrained_senders]
            receiver_params = [{'params': receiver.parameters(), 'lr': opts.receiver_lr}]
            optimizer = torch.optim.Adam(sender_params + receiver_params)
            trainer = core.Trainer(
                game=pretraining_game, optimizer=optimizer, train_data=train_loader,
                validation_data=test_loader,
                callbacks=[
                    CompositionalityMetricPretraining2(full_dataset, opts, test.indices, prefix='1_'),
                    CompositionalityMetricPretraining3(full_dataset, opts, test.indices, prefix='2_'),
                    NeptuneMonitor(),
                    core.ConsoleLogger(print_train_loss=True),
                    EarlyStopperAccuracy(threshold=0.95, field_name='pretraining_accuracy'),
                ])
            trainer.train(n_epochs=25_000)
            pretraining_game.train(False)

        # Compositional game
        assert sender_3.training
        assert not receiver.training
        compositional_game = CompositionalGame(sender_3, receiver, loss=loss_diff)
        sender_params = [{'params': sender_3.parameters(), 'lr': opts.sender_lr}]
        receiver_params = [{'params': receiver.parameters(), 'lr': opts.receiver_lr * 0.01}]
        optimizer = torch.optim.Adam(sender_params + receiver_params)
        trainer = core.Trainer(game=compositional_game, optimizer=optimizer, train_data=train_loader,
                               validation_data=test_loader,
                               callbacks=[
                                   CompositionalityMetric(full_dataset, opts, test.indices),
                                   NeptuneMonitor(),
                                   core.ConsoleLogger(print_train_loss=True),
                                   # core.TemperatureUpdater(agent=sender_3, update_frequency=1000),
                               ])
        trainer.train(n_epochs=opts.n_epochs)
