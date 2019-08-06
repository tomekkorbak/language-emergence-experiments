import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg import core


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


def simple_loss(target, output, prefix):
    acc = (output.argmax(dim=1) == target).detach().float().mean(dim=0)
    loss = F.cross_entropy(output, target, reduction="none")
    return loss, {f'{prefix}_accuracy': acc.item()}


class InputNoiseInjector(nn.Module):

    def __init__(self, strategy: str = None):
        super(InputNoiseInjector, self).__init__()
        self.strategy = strategy

    def forward(self, input):
        if self.strategy == 'full_permutation':
            permutation = torch.randperm(input.size()[-1])
            return input[..., permutation]
        return input


class PretrainingmGame(nn.Module):
    def __init__(
            self,
            senders,
            receiver,
            loss,
            noise_injector=InputNoiseInjector(strategy='full_permutation')
    ):
        super(PretrainingmGame, self).__init__()
        self.sender_1, self.sender_2 = senders
        self.receiver = receiver
        self.loss = loss
        self.noise_injector = noise_injector

    def forward(self, sender_input, target):
        if random.choice([True, False]):
            message_1 = self.sender_1(sender_input)
            with torch.no_grad():
                message_2 = self.sender_2(self.noise_injector(sender_input))
            message = torch.cat([message_1, message_2], dim=1)
            first_receiver_output, second_receiver_output = self.receiver(message)
            loss, rest_info = simple_loss(target[:, 0], first_receiver_output[:, -1, ...], prefix='pretraining')
        else:
            with torch.no_grad():
                message_1 = self.sender_1(self.noise_injector(sender_input))
            message_2 = self.sender_2(sender_input)
            message = torch.cat([message_1, message_2], dim=1)
            first_receiver_output, second_receiver_output = self.receiver(message)
            loss, rest_info = simple_loss(target[:, 1], second_receiver_output[:, -1, ...], prefix='pretraining')
        return loss.mean(), rest_info


class PretrainingmGameWithTargetedNoise(nn.Module):
    def __init__(
            self,
            senders,
            receiver,
            loss,
            noise_injector=InputNoiseInjector(strategy='full_permutation')
    ):
        super(PretrainingmGameWithTargetedNoise, self).__init__()
        self.sender_1, self.sender_2 = senders
        self.receiver = receiver
        self.loss = loss
        self.noise_injector = noise_injector

    def forward(self, sender_input, target):
        if random.choice([True, False]):
            message_1 = self.sender_1(sender_input[:, 0])
            with torch.no_grad():
                message_2 = self.sender_2(self.noise_injector(sender_input[:, 1]))
            message = torch.cat([message_1, message_2], dim=1)
            first_receiver_output, second_receiver_output = self.receiver(message)
            loss, rest_info = simple_loss(target[:, 0], first_receiver_output[:, -1, ...], prefix='pretraining')
        else:
            with torch.no_grad():
                message_1 = self.sender_1(self.noise_injector(sender_input[:, 0]))
            message_2 = self.sender_2(sender_input[:, 1])
            message = torch.cat([message_1, message_2], dim=1)
            first_receiver_output, second_receiver_output = self.receiver(message)
            loss, rest_info = simple_loss(target[:, 1], second_receiver_output[:, -1, ...], prefix='pretraining')
        return loss.mean(), rest_info


class CompositionalGame(nn.Module):
    def __init__(
            self,
            sender,
            receiver,
            loss,
    ):
        super(CompositionalGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, target):
        message = self.sender(sender_input)
        first_receiver_output, second_receiver_output = self.receiver(message)
        loss, rest_info = self.loss(target, first_receiver_output[:, -1, ...], second_receiver_output[:, -1, ...], prefix='comp')
        return loss.mean(), rest_info


class RnnReceiverGS(core.RnnReceiverGS):

    def forward(self, message, input=None):
        outputs1, outputs2 = [], []

        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c)) if prev_hidden is not None else \
                    self.cell(e_t)
            else:
                h_t = self.cell(e_t, prev_hidden)
            output1, output2 = self.agent(h_t, input)
            outputs1.append(output1)
            outputs2.append(output2)

            prev_hidden = h_t

        outputs1 = torch.stack(outputs1).permute(1, 0, 2)
        outputs2 = torch.stack(outputs2).permute(1, 0, 2)

        return outputs1, outputs2