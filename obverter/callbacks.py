import collections
import random

from neptunecontrib.monitoring.utils import send_figure
import neptune
import seaborn as sns
import matplotlib.pyplot as plt
from egg.core import Callback, EarlyStopperAccuracy
import torch
from tabulate import tabulate

from compositionality.metrics import compute_concept_symbol_matrix, compute_context_independence, compute_representation_similarity


class NeptuneMonitor(Callback):

    def __init__(self, prefix=None):
        self.epoch_counter = 0
        self.prefix = prefix + '_' if prefix else ''

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        if self.epoch_counter % 10 == 0:
            neptune.send_metric(f'{self.prefix}test_loss', self.epoch_counter, loss)
            for metric, value in rest.items():
                neptune.send_metric(f'{self.prefix}train_{metric}', self.epoch_counter, value)

    def on_test_end(self, loss, rest):
        neptune.send_metric(f'{self.prefix}test_loss', self.epoch_counter, loss)
        for metric, value in rest.items():
            neptune.send_metric(f'{self.prefix}test_{metric}', self.epoch_counter, value)


class CompositionalityMetric(Callback):

    def __init__(self, dataset, sender, opts, vocab_size, test_indices, prefix=''):
        self.dataset = dataset
        self.sender = sender
        self.epoch_counter = 0
        self.opts = opts
        self.vocab_size = vocab_size
        self.test_indices = test_indices
        self.prefix = prefix

        self.epoch_counter = 0

    def on_epoch_end(self, *args):
        self.epoch_counter += 1
        if self.epoch_counter % 100 == 0:
            self.input_to_message = collections.defaultdict(list)
            self.message_to_output = collections.defaultdict(list)
            train_state = self.trainer.game.training  # persist so we restore it back
            self.trainer.game.train(mode=False)
            for _ in range(10):
                self.run_inference()
            self.concept_symbol_matrix, concepts = compute_concept_symbol_matrix(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes,
                vocab_size=self.vocab_size
            )
            self.trainer.game.train(mode=train_state)
            self.print_table_input_to_message()
            self.draw_concept_symbol_matrix()

            # Context independence metrics
            context_independence_scores, v_cs = compute_context_independence(
                self.concept_symbol_matrix,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes,
            )
            neptune.send_metric(self.prefix + 'context independence', self.epoch_counter, context_independence_scores.mean(axis=0))
            neptune.send_text(self.prefix + 'v_cs', str(v_cs.tolist()))
            neptune.send_text(self.prefix + 'context independence scores', str(context_independence_scores.tolist()))
            # RSA
            correlation_coeff, p_value = compute_representation_similarity(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes
            )
            neptune.send_metric(self.prefix + 'RSA', self.epoch_counter, correlation_coeff)
            neptune.send_metric(self.prefix + 'RSA_p_value', self.epoch_counter, p_value)

    def run_inference(self):
        raise NotImplementedError()

    def print_table_input_to_message(self):
        table_data = [['x'] + list(range(self.opts.n_features))] + [[i] + [None] * self.opts.n_features for i in range(self.opts.n_features)]
        for (input1, input2), messages in self.input_to_message.items():
            table_data[input1 + 1][input2 + 1] = '  '.join((' '.join((str(s) for s in message)) for message in set(messages)))
        for a, b in zip(range(self.opts.n_features), self.test_indices):
            table_data[a+1][(b % self.opts.n_features) + 1] = '*' + table_data[a+1][(b % self.opts.n_features) +1]
        filename = f'{self.prefix}input_to_message_{self.epoch_counter}.txt'
        with open(file=filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='fancy_grid'))
        neptune.send_artifact(filename)
        with open(file='latex' + filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='latex'))
        neptune.send_artifact('latex' + filename)

    def draw_concept_symbol_matrix(self):
        figure, ax = plt.subplots(figsize=(20, 5))
        figure.suptitle(f'Concept-symbol matrix {self.epoch_counter}')
        g = sns.heatmap(self.concept_symbol_matrix, annot=True, fmt='.2f', ax=ax)
        g.set_title(f'Concept-symbol matrix {self.epoch_counter}')
        send_figure(figure, channel_name=self.prefix + 'concept_symbol_matrix')
        plt.close()


class CompositionalityMetricObverter(CompositionalityMetric):

    def run_inference(self):
        with torch.no_grad():
            inputs, targets = self.dataset.tensors
            messages = self.sender.decode(inputs)
            for i in range(inputs.size(0)):
                input = tuple(inputs[i].argmax(dim=1).tolist())
                message = tuple(messages[i].tolist())
                neptune.send_text(self.prefix + 'messages', f'{input} -> {message}')
                self.input_to_message[input].append(message)


class EarlyStopperAccuracy(EarlyStopperAccuracy):
    """
    Implements early stopping logic that stops training when a threshold on a metric
    is achieved.
    """
    def __init__(self, threshold: float, field_name: str = 'acc', delay=5) -> None:
        """
        :param threshold: early stopping threshold for the validation set accuracy
            (assumes that the loss function returns the accuracy under name `field_name`)
        :param field_name: the name of the metric return by loss function which should be evaluated against stopping
            criterion (default: "acc")
        """
        super(EarlyStopperAccuracy, self).__init__(threshold, field_name)
        self.delay = delay

    def should_stop(self) -> bool:
        if len(self.validation_stats) < self.delay:
            return False
        assert self.trainer.validation_data is not None, 'Validation data must be provided for early stooping to work'
        return all(logs[self.field_name] > self.threshold for _, logs in self.validation_stats[-self.delay:])

    def on_train_end(self):
        if self.should_stop():
            print(f'Stopped early on epoch {self.epoch}')