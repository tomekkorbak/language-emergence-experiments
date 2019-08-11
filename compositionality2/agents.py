import torch
import torch.nn as nn


class Sender(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features)
        self.fc2 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        x = x.flatten(1, 2)
        hidden = self.fc1(x).tanh()
        return self.fc2(hidden).tanh()


class Receiver(nn.Module):
    def __init__(self, n_features, hidden_layer_size):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size * 2, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 2)

    def forward(self, message, distractor):
        distractor = self.fc1(distractor.flatten(1, 2)).tanh()
        combined = torch.cat((distractor, message), dim=-1)
        hidden = self.fc2(combined).tanh()
        return self.fc3(hidden)
