import torch.nn as nn
import torch.nn.functional as F


def loss_diff(targets, receiver_output_1, receiver_output_2, prefix):
    acc_1 = (receiver_output_1.argmax(dim=1) == targets[:, 0]).detach().float()
    acc_2 = (receiver_output_2.argmax(dim=1) == targets[:, 1]).detach().float()
    loss_1 = F.cross_entropy(receiver_output_1, targets[:, 0], reduction="none")
    loss_2 = F.cross_entropy(receiver_output_2, targets[:, 1], reduction="none")
    acc = (acc_1 * acc_2).mean(dim=0)
    loss = loss_1 + loss_2
    return loss, {f'{prefix}_accuracy': acc.item(),
                  f'{prefix}_first_accuracy': acc_1.mean(dim=0).item(),
                  f'{prefix}_second_accuracy': acc_2.mean(dim=0).item()}


class PretrainingmGame(nn.Module):
    def __init__(
            self,
            senders,
            receivers,
            loss,
    ):
        super(PretrainingmGame, self).__init__()
        self.sender_1, self.sender_2 = senders
        self.receiver_1, self.receiver_2 = receivers
        self.loss = loss

    def forward(self, sender_input, target):
        message_1 = self.sender_1(sender_input)
        message_2 = self.sender_2(sender_input)
        first_receiver_output = self.receiver_1(message_1)[:, -1, ...]
        second_receiver_output = self.receiver_2(message_2)[:, -1, ...]
        loss, rest_info = self.loss(target, first_receiver_output, second_receiver_output, prefix='pretraining')
        return loss.mean(), rest_info


class CompositionalGame(nn.Module):
    def __init__(
            self,
            sender,
            receivers,
            loss,
    ):
        super(CompositionalGame, self).__init__()
        self.sender = sender
        self.receiver_1, self.receiver_2 = receivers
        self.loss = loss

    def forward(self, sender_input, target):
        message = self.sender(sender_input)
        first_receiver_output = self.receiver_1(message)[:, -1, ...]
        second_receiver_output = self.receiver_2(message)[:, -1, ...]
        loss, rest_info = self.loss(target, first_receiver_output, second_receiver_output, prefix='comp')
        return loss.mean(), rest_info

