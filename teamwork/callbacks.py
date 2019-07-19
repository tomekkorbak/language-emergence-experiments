from neptunecontrib.monitoring.utils import send_figure
import neptune
import seaborn as sns
import matplotlib.pyplot as plt
from egg.core import Callback
import torch
import torch.nn.functional as F


class NeptuneMonitor(Callback):

    def __init__(self, experiment):
        self.experiment = experiment
        self.epoch_counter = 0

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        self.experiment.send_metric(f'test_loss', loss)
        for metric, value in rest.items():
            self.experiment.send_metric(f'train_{metric}', value)

    def on_test_end(self, loss, rest):
        self.experiment.send_metric(f'test_loss', loss)
        for metric, value in rest.items():
            self.experiment.send_metric(f'test_{metric}', value)

        self.save_codebook(
            weight_list=[F.softmax(self.trainer.game.executive_sender.agent.fc1.weight.detach(), dim=0).numpy()],
            epoch=self.epoch_counter,
            label='Executive sender softmax'
        )

    def save_codebook(self, weight_list, epoch, label):
        figure, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 5))
        figure.suptitle(f'Epoch {epoch}')
        for i, (matrix, ax) in enumerate(zip(weight_list, axes)):
            g = sns.heatmap(matrix, annot=True, fmt='.2f', ax=ax)
            g.set_title(f'{label} {i}')
        send_figure(figure)
        plt.close()


class Dump(Callback):

    def __init__(self, experiment, test_set):
        self.experiment = experiment
        self.test_set = test_set

    def on_epoch_end(self, *args):
        with torch.no_grad():
            mapping = {}
            for (input_1, input_2), targets in self.test_set:
                indices, _, _ = self.trainer.game.executive_sender((input_1, input_2))
                message = self.trainer.game.sender_ensemble((input_1, input_2), indices)
                receiver_output_1 = self.trainer.game.first_receiver_ensemble(message, indices)[:, -1, ...]
                receiver_output_2 = self.trainer.game.second_receiver_ensemble(message, indices)[:, -1, ...]

                for i in range(input_1.size(0)):
                    neptune.send_text(
                        'messages',
                        f'{input_1[i].argmax(dim=0).item()} & {input_2[i].argmax(dim=0).item()} -> '
                        f'{indices[i].item()} -> '
                        f'{message[i].argmax(dim=1).tolist()} -> '
                        f'{receiver_output_1[i].argmax(dim=0).item()} & {receiver_output_2[i].argmax(dim=0).item()} '
                        f'(expected {targets[0][i]} & {targets[1][i]})'
                    )
                    mapping[(input_1[i].argmax(dim=0).item(), input_2[i].argmax(dim=0).item())] = f'{message[i].argmax(dim=1).tolist()}'
                    if len(mapping) == 10:
                        neptune.send_text('mapping', str(mapping))
                        return
