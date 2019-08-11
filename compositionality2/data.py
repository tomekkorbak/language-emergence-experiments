import itertools
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils import data


def split_targets(dimensions):
    targets = list(itertools.product(range(dimensions[0]), range(dimensions[1])))
    first_dim_indices, second_dim_indices = list(range(dimensions[0])), list(range(dimensions[1]))
    random.shuffle(second_dim_indices)
    test_targets = [targets[a * 5 + b] for a, b in zip(first_dim_indices, second_dim_indices)]
    train_targets = [target for target in targets if target not in test_targets]
    return targets, train_targets, test_targets


def create_dataset(pairs):
    targets, labels, distractors = zip(*pairs)
    return data.TensorDataset(
        F.one_hot(torch.LongTensor(targets), 5).squeeze().float(),
        torch.LongTensor(labels),
        F.one_hot(torch.LongTensor(distractors), 5).squeeze().float())


def get_label(target: Tuple[int, int], distractor: Tuple[int, int]) -> Tuple[int, int]:
    return target[0] == distractor[0], target[1] == distractor[1]


def prepare_datasets(dimensions=[5, 5]):
    assert dimensions == [5, 5], 'Other dimensions not implemented yet'
    all_targets, train_targets, test_targets = split_targets(dimensions)

    train_pairs = []
    for target in train_targets:
        distractors = [target] * 20 + random.choices(all_targets, k=20)
        for distractor in distractors:
            train_pairs.append((target, get_label(target, distractor), distractor))

    test_pairs = []
    for target in test_targets:
        distractors = [target] * 20 + random.choices(all_targets, k=20)
        for distractor in distractors:
            test_pairs.append((target, get_label(target, distractor), distractor))

    train_dataset = create_dataset(train_pairs)
    test_dataset = create_dataset(test_pairs)
    all_targets_dataset = data.TensorDataset(F.one_hot(torch.LongTensor(all_targets), 5).squeeze().float())
    return train_dataset, test_dataset, all_targets_dataset, test_targets


if __name__ == "__main__":
    train, test, full, _ = prepare_datasets()
    print(len(train), len(test), len(full))
