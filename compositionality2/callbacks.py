import collections
from typing import List

from neptunecontrib.monitoring.utils import send_figure
import neptune
import seaborn as sns
import matplotlib.pyplot as plt
from egg.core import Callback, EarlyStopperAccuracy
import torch
from tabulate import tabulate

from compositionality2.metrics import compute_concept_symbol_matrix, compute_context_independence, compute_representation_similarity


def prune(message: List[int]):
    prunned_message = []
    for symbol in message:
        if symbol != 0:
            prunned_message.append(symbol)
        else:
            break
    return prunned_message


class NeptuneMonitor(Callback):

    def __init__(self, train_freq=1, test_freq=1):
        self.epoch_counter = 0
        self.train_freq = train_freq
        self.test_freq = test_freq

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        if self.epoch_counter % self.train_freq == 0:
            neptune.send_metric(f'test_loss', loss)
            for metric, value in rest.items():
                neptune.send_metric(f'train_{metric}', value)

    def on_test_end(self, loss, rest):
        if self.epoch_counter % self.test_freq == 0:
            neptune.send_metric(f'test_loss', loss)
            for metric, value in rest.items():
                neptune.send_metric(f'test_{metric}', value)


class CompositionalityMetric(Callback):

    def __init__(self, dataset, opts, test_targets, prefix=''):
        self.dataset = dataset
        self.epoch_counter = 0
        self.opts = opts
        self.prefix = prefix
        self.test_targets = test_targets

    def on_epoch_end(self, *args):
        self.epoch_counter += 1
        if self.epoch_counter % 100 == 0:
            self.input_to_message = collections.defaultdict(list)
            train_state = self.trainer.game.training  # persist so we restore it back
            self.trainer.game.train(mode=False)
            self.run_inference()
            self.trainer.game.train(mode=train_state)
            self.concept_symbol_matrix, concepts = compute_concept_symbol_matrix(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * 2,
                vocab_size=self.opts.vocab_size
            )
            self.print_table_input_to_message()
            self.draw_concept_symbol_matrix()

            # Context independence metrics
            context_independence_scores, v_cs = compute_context_independence(
                self.concept_symbol_matrix,
                input_dimensions=[self.opts.n_features] * 2,
            )
            neptune.send_metric(self.prefix + 'context independence', context_independence_scores.mean(axis=0))
            neptune.send_text(self.prefix + 'v_cs', str(v_cs.tolist()))
            neptune.send_text(self.prefix + 'context independence scores', str(context_independence_scores.tolist()))

            # RSA
            correlation_coeff, p_value = compute_representation_similarity(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * 2
            )
            neptune.send_metric(self.prefix + 'RSA', correlation_coeff)
            neptune.send_metric(self.prefix + 'RSA_p_value', p_value)

    def run_inference(self):
        with torch.no_grad():
            inputs, = self.dataset.tensors
            messages, _, _ = self.trainer.game.sender(inputs)
        for i in range(inputs.size(0)):
            input = tuple(inputs[i].argmax(dim=1).tolist())
            message = tuple(prune(messages[i].tolist()))
            neptune.send_text(self.prefix + 'messages', f'{input} -> {message}')
            self.input_to_message[input].append(message)

    def print_table_input_to_message(self):
        table_data = [['x'] + list(range(self.opts.n_features))] + [[i] + [None] * self.opts.n_features for i in range(self.opts.n_features)]
        for (input1, input2), messages in self.input_to_message.items():
            table_data[input1 + 1][input2 + 1] = '  '.join((' '.join((str(s) for s in message)) for message in set(messages)))
        for a, b in self.test_targets:
            table_data[a+1][b+1] = '*' + table_data[a+1][b+1]
        filename = f'{self.prefix}input_to_message_{self.epoch_counter}.txt'
        with open(file=filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='fancy_grid'))
        neptune.send_artifact(filename)

    def draw_concept_symbol_matrix(self):
        figure, ax = plt.subplots(figsize=(20, 5))
        figure.suptitle(f'Concept-symbol matrix {self.epoch_counter}')
        g = sns.heatmap(self.concept_symbol_matrix, annot=True, fmt='.2f', ax=ax)
        g.set_title(f'Concept-symbol matrix {self.epoch_counter}')
        send_figure(figure, channel_name=self.prefix + 'concept_symbol_matrix')
        plt.close()