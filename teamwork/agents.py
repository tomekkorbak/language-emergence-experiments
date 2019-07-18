from typing import Tuple

import torch
import torch.nn as nn


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features, n_attributes):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_attributes*n_features, n_attributes*n_features*2)
        self.fc2 = nn.Linear(n_attributes*n_features*2, n_hidden)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):
        input = torch.cat(input, dim=1)
        hidden = torch.nn.functional.relu(self.fc1(input))
        return self.fc2(hidden)


class ExecutiveSender(nn.Module):
    def __init__(self, n_choices, n_features, n_attributes):
        super(ExecutiveSender, self).__init__()
        self.fc1 = nn.Linear(n_attributes*n_features, n_attributes*n_features*2)
        self.fc2 = nn.Linear(n_attributes*n_features*2, n_choices)

    def forward(self, input):
        input = torch.cat(input, dim=1)
        hidden = torch.nn.functional.relu(self.fc1(input))
        return self.fc2(hidden)


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_outputs):
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _input):
        return self.fc1(x).squeeze(dim=0)


if __name__ == "__main__":
    from teamwork.data import TupleDataset
    from torch.utils import data
    train, dev = TupleDataset.create_train_and_dev(perceptual_dimensions=[10, 10])
    loader = data.DataLoader(train, batch_size=16, drop_last=True, shuffle=True)
    batch = next(iter(loader))
    (idx1, idx2), target = batch
    sender = Sender(n_hidden=200, n_features=10, n_attributes=2)
    emb = sender((idx1, idx2))
    assert emb.shape == torch.Size((32, 200))
