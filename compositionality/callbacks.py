import collections

from neptunecontrib.monitoring.utils import send_figure
import neptune
import seaborn as sns
import matplotlib.pyplot as plt
from egg.core import Callback
import torch
from tabulate import tabulate

from compositionality.metrics import compute_concept_symbol_matrix, compute_context_independence


class NeptuneMonitor(Callback):

    def __init__(self):
        self.epoch_counter = 0

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        if self.epoch_counter % 10 == 0:
            neptune.send_metric(f'test_loss', loss)
            for metric, value in rest.items():
                neptune.send_metric(f'train_{metric}', value)

    def on_test_end(self, loss, rest):
        neptune.send_metric(f'test_loss', loss)
        for metric, value in rest.items():
            neptune.send_metric(f'test_{metric}', value)


class CompositionalityMetric(Callback):

    def __init__(self, dataset, opts, test_indices):
        self.dataset = dataset
        self.epoch_counter = 0
        self.opts = opts
        self.test_indices = test_indices

    def on_epoch_end(self, *args):
        self.epoch_counter += 1
        if self.epoch_counter % 100 == 0:
            self.input_to_message = collections.defaultdict(list)
            self.message_to_output = collections.defaultdict(list)
            train_state = self.trainer.game.training  # persist so we restore it back
            self.trainer.game.train(mode=False)
            for _ in range(10):
                self.run_inference()
            self.concept_symbol_matrix = compute_concept_symbol_matrix(self.input_to_message, self.opts)
            self.trainer.game.train(mode=train_state)
            self.print_table_input_to_message()
            self.draw_concept_symbol_matrix()

            # Context independence metrics
            context_independence_scores, v_cs = compute_context_independence(self.concept_symbol_matrix, self.opts)
            neptune.send_metric('context independence', context_independence_scores.mean(dim=0))
            neptune.send_text('v_cs', str(v_cs.tolist()))
            neptune.send_text('context independence scores', str(context_independence_scores.tolist()))

    def run_inference(self):
        with torch.no_grad():
            inputs, targets = self.dataset.tensors
            messages = self.trainer.game.sender(inputs)
            first_receiver_output = self.trainer.game.receiver_1(messages)[:, -1, ...]
            second_receiver_output = self.trainer.game.receiver_2(messages)[:, -1, ...]
            for i in range(inputs.size(0)):
                input = tuple(inputs[i].argmax(dim=1).tolist())
                message = tuple(messages[i].argmax(dim=1).tolist())
                output = tuple([first_receiver_output[i].argmax(dim=0).item(), second_receiver_output[i].argmax(dim=0).item()])
                target = tuple(targets[i].tolist())
                neptune.send_text('messages', f'{input} -> {message} -> {output} (expected {target})')
                self.input_to_message[input].append(message)
                self.message_to_output[message].append(output)

    def print_table_input_to_message(self):
        table_data = [['x'] + list(range(self.opts.n_features))] + [[i] + [None] * self.opts.n_features for i in range(self.opts.n_features)]
        for (input1, input2), messages in self.input_to_message.items():
            table_data[input1 + 1][input2 + 1] = '  '.join((' '.join((str(s) for s in message)) for message in set(messages)))
        for a, b in zip(range(self.opts.n_features), self.test_indices):
            table_data[a+1][(b % self.opts.n_features) + 1] = '*' + table_data[a+1][(b % self.opts.n_features) +1]
        filename = f'input_to_message_{self.epoch_counter}'
        with open(file=filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='fancy_grid'))
        neptune.send_artifact(filename)

    def draw_concept_symbol_matrix(self):
        figure, ax = plt.subplots(figsize=(20, 5))
        figure.suptitle(f'Concept-symbol matrix {self.epoch_counter}')
        g = sns.heatmap(self.concept_symbol_matrix.numpy(), annot=True, fmt='.2f', ax=ax)
        g.set_title(f'Concept-symbol matrix {self.epoch_counter}')
        send_figure(figure, channel_name='concept_symbol_matrix')
        plt.close()


class TemperatureUpdater(Callback):

    def __init__(self, agent, decay=0.9, minimum=0.1, update_frequency=1):
        self.agent = agent
        assert hasattr(agent, 'temperature'), 'Agent must have a `temperature` attribute'
        assert not isinstance(agent.temperature, torch.nn.Parameter), \
            'When using TemperatureUpdater `temperature` cannot be trainable'
        self.decay = decay
        self.minimum = minimum
        self.update_frequency = update_frequency
        self.epoch_counter = 0

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        if self.epoch_counter % self.update_frequency == 0:
            self.agent.temperature = max(self.minimum, self.agent.temperature * self.decay)
        neptune.send_metric('temperature', self.agent.temperature)
