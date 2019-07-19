import itertools
from typing import List, Tuple
import random

import torch
import torch.nn.functional as F
from torch.utils import data


class TupleDataset(data.Dataset):
    def __init__(
            self,
            tuples: List[Tuple[int, ...]],
            targets: List[int]
    ):
        self.list_of_tuples = tuples
        self.targets = targets

    @classmethod
    def create_train_and_dev(
            cls,
            perceptual_dimensions: List[int],
    ) -> Tuple['TupleDataset', 'TupleDataset']:
        list_of_dim = [range(elem) for elem in perceptual_dimensions]
        tuples = list(itertools.product(*list_of_dim))
        train_targets, test_targets = cls.choose_test_set(tuples)
        train_features, test_features = cls.generate_features(train_targets), cls.generate_features(test_targets)

        train = cls(tuples=train_features, targets=train_targets)
        dev = cls(tuples=test_features, targets=test_targets)
        return train, dev

    @staticmethod
    def choose_test_set(list_of_tuples):
        first_dim_indices = list(range(10))
        second_dim_indices = list(range(10))
        random.shuffle(second_dim_indices)
        indices = [a * 10 + b for a, b in zip(first_dim_indices, second_dim_indices)]
        train = [item for idx, item in enumerate(list_of_tuples) if idx not in indices]
        test = [item for idx, item in enumerate(list_of_tuples) if idx in indices]
        return train, test

    @staticmethod
    def generate_features(list_of_tuples):
        return [[F.one_hot(torch.LongTensor([elem]), 10).squeeze().float() for elem in tuple]
                for tuple in list_of_tuples]

    def __len__(self) -> int:
        return len(self.list_of_tuples) * 100

    def __getitem__(self, idx: int) -> Tuple[List[torch.LongTensor], List[int]]:
        idx = idx % len(self.list_of_tuples)
        return self.list_of_tuples[idx], self.targets[idx]


if __name__ == "__main__":
    train, dev = TupleDataset.create_train_and_dev(perceptual_dimensions=[10, 10])
    loader = data.DataLoader(train, batch_size=16, drop_last=True, shuffle=True)
    batch = next(iter(loader))
    (idx1, idx2), target = batch
    assert idx1.shape == idx2.shape == torch.Size((16, 10))
    assert len(target) == 2
