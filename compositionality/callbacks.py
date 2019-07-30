import collections

from neptunecontrib.monitoring.utils import send_figure
import neptune
import seaborn as sns
import matplotlib.pyplot as plt
from egg.core import Callback
import torch
from tabulate import tabulate


Rollout = collections.namedtuple('Rollout', field_names=['input', 'message'])


def compute_context_independence(concept_symbol_matrix, opts, exclude_indices=None):
    v_cs = concept_symbol_matrix.argmax(dim=1)
    context_independence_scores = torch.zeros(opts.n_features * opts.n_attributes)
    for concept in range(concept_symbol_matrix.size(0)):
        v_c = v_cs[concept]
        p_vc_c = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[concept, :].sum(dim=0)
        p_c_vc = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[:, v_c].sum(dim=0)
        context_independence_scores[concept] = p_vc_c * p_c_vc
    neptune.send_text('v_cs', str(v_cs.tolist()))
    neptune.send_text('context independence scores', str(context_independence_scores.tolist()))
    return context_independence_scores.mean(dim=0)



def compute_concept_symbol_matrix(input_to_message, opts, epsilon=1e-4):
    concept_to_message = collections.defaultdict(list)
    for (concept1, concept2), messages in input_to_message.items():
            concept_to_message[concept1] += messages
            concept_to_message[opts.n_features + concept2] += messages
    concept_symbol_matrix = torch.FloatTensor(opts.n_features * opts.n_attributes,
                                              opts.vocab_size).fill_(epsilon)
    for concept, messages in concept_to_message.items():
        for message in messages:
            for symbol in message:
                concept_symbol_matrix[concept, symbol] += 1
    return concept_symbol_matrix


class NeptuneMonitor(Callback):

    def __init__(self, experiment):
        self.experiment = experiment
        self.epoch_counter = 0

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        if self.epoch_counter % 10 == 0:
            self.experiment.send_metric(f'test_loss', loss)
            for metric, value in rest.items():
                self.experiment.send_metric(f'train_{metric}', value)

    def on_test_end(self, loss, rest):
        self.experiment.send_metric(f'test_loss', loss)
        for metric, value in rest.items():
            self.experiment.send_metric(f'test_{metric}', value)

    # def save_codebook(self, weight_list, epoch, label):
    #     figure, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 5))
    #     figure.suptitle(f'Epoch {epoch}')
    #     for i, (matrix, ax) in enumerate(zip(weight_list, axes)):
    #         g = sns.heatmap(matrix, annot=True, fmt='.2f', ax=ax)
    #         g.set_title(f'{label} {i}')
    #     send_figure(figure, channel_name=label)
    #     plt.close()


class CompositionalityMetric(Callback):

    def __init__(self, experiment, dataset, opts, test_indices):
        self.experiment = experiment
        self.dataset = dataset
        self.epoch_counter = 0
        self.opts = opts
        self.test_indices = test_indices

    def on_epoch_end(self, *args):
        self.epoch_counter += 1
        if self.epoch_counter % 100 == 0:
            self.counter = collections.Counter()
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
            self.experiment.send_metric('context independence', compute_context_independence(self.concept_symbol_matrix, self.opts))

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
                self.counter[Rollout(input, message)] += 1
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
        self.experiment.log_artifact(filename)



    def draw_concept_symbol_matrix(self):
        figure, ax = plt.subplots(figsize=(20, 5))
        figure.suptitle(f'Concept-symbol matrix {self.epoch_counter}')
        g = sns.heatmap(self.concept_symbol_matrix.numpy(), annot=True, fmt='.2f', ax=ax)
        g.set_title(f'Concept-symbol matrix {self.epoch_counter}')
        send_figure(figure, channel_name='concept_symbol_matrix')
        plt.close()


class TemperatureUpdater(Callback):

    def __init__(self, experiment, agent, decay=0.9, minimum=0.1, update_frequency=1):
        self.experiment = experiment
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
        self.experiment.send_metric('temperature', self.agent.temperature)


if __name__ == "__main__":
    pass