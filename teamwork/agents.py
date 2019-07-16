import torch.nn as nn
from egg import core


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        return self.fc1(x)


class ExecutiveSender(nn.Module):
    def __init__(self, n_choices, n_features):
        super(ExecutiveSender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_choices)

    def forward(self, x):
        return self.fc1(x)


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Receiver, self).__init__()
        self.fc1 = core.RelaxedEmbedding(n_features, n_hidden)

    def forward(self, x, _input):
        return self.fc1(x).squeeze(dim=0)
