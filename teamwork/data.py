from functools import reduce
import itertools
from typing import List, Tuple

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
            train_set_size=80
    ) -> Tuple['TupleDataset', 'TupleDataset']:
        world_dim = reduce(lambda x, y: x * y, perceptual_dimensions)
        list_of_dim = [range(elem) for elem in perceptual_dimensions]
        all_vectors = list(itertools.product(*list_of_dim))
        all_vectors = [[F.one_hot(torch.LongTensor([elem]), dimension).squeeze().float()
                        for elem, dimension in zip(tuple, perceptual_dimensions)]
                    for tuple in all_vectors]
        train = cls(tuples=all_vectors[:train_set_size]*100, targets=list(range(train_set_size))*100)
        dev = cls(tuples=all_vectors[train_set_size:]*100, targets=list(range(train_set_size, world_dim))*100)
        return train, dev

    def __len__(self) -> int:
        return len(self.list_of_tuples)

    def __getitem__(self, idx: slice) -> Tuple[List[Tuple[int, ...]], List[int]]:
        return self.list_of_tuples[idx], self.targets[idx]


if __name__ == "__main__":
    train, dev = TupleDataset.create_train_and_dev(perceptual_dimensions=[10, 10])
    loader = data.DataLoader(train, batch_size=16, drop_last=True, shuffle=True, )
    batch = next(iter(loader))
    (idx1, idx2), target = batch
    assert idx1.shape == idx2.shape == torch.Size((16, 10))
    assert target.shape == torch.Size((16,))


