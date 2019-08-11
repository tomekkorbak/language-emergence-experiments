from collections import defaultdict

from torch import nn
import torch.nn.functional as F
# from egg.utils import find_lengths


def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    first_label, second_label = labels[:, 0], labels[:, 1]
    labels = first_label * second_label
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {'acc': acc}


class CompositionalGameReinforce(nn.Module):
    def __init__(
            self,
            sender,
            receiver,
            loss,
    ):
        super(CompositionalGameReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, target):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        receiver_output, log_prob_r, entropy_r = self.receiver(message, target)
        loss, rest = self.loss(None, None, None, receiver_output, labels)

        log_prob = log_prob_s.mean(dim=1)
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()
        optimized_loss = policy_loss + loss.mean() - entropy_s.mean() * 0.2

        if self.training:
            self.update_baseline('loss', loss)
        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


