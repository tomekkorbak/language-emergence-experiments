import itertools
import random

import torch
import torch.nn.functional as F
from torch.utils import data


def split(full_dataset, n_features, n_attributes):
    """Split dataset making sure that each symbol is represented with equal frequency"""
    assert n_attributes == 2, 'Only implemented for 2 attrs'
    first_dim_indices, second_dim_indices = list(range(n_features)), list(range(n_features))
    random.shuffle(second_dim_indices)
    test_indices = [a * n_features + b for a, b in zip(first_dim_indices, second_dim_indices)]
    train_indices = [i for i in range(n_features * n_features) if i not in test_indices]
    return data.Subset(full_dataset, train_indices), data.Subset(full_dataset, test_indices)


def prepare_datasets(n_features, n_attributes):
    dimensions = [range(n_features)] * n_attributes
    targets = torch.LongTensor(list(itertools.product(*dimensions)))
    features = F.one_hot(targets, n_features).squeeze().float()
    full_dataset = data.TensorDataset(features, targets)
    train_dataset, test_dataset = split(full_dataset, n_features, n_attributes)
    return full_dataset, train_dataset, test_dataset


if __name__ == "__main__":
    full, train, test = prepare_datasets(5, 2)
    loader = data.DataLoader(train, batch_size=8, drop_last=True, shuffle=True)
    batch = next(iter(loader))
