import torch
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


colors = ['blue', 'cyan', 'gray', 'green', 'magenta']

object_types = ['box', 'sphere', 'cylinder', 'torus', 'ellipsoid']


class ColoredFiguresDataset(ImageFolder):

    def __getitem__(self, index):
        filename, _ = self.imgs[index]
        color, figure = filename.split('/')[-2].split('-')
        color_idx, figure_idx = colors.index(color), object_types.index(figure)
        label = torch.LongTensor([color_idx, figure_idx])
        return super(ColoredFiguresDataset, self).__getitem__(index)[0], label

# class ColoredFiguresDataset2(torchvision.datasets.ImageFolder):
#
#     def get_distractor(self, sender_input, sender_label):
#         if random.choice([True, False]):
#             return sender_input, sender_label
#         else:
#             random_idx = random.randint(0, len(self)-1)
#             return super(ColoredFiguresDataset2, self).__getitem__(random_idx)
#
#     def __getitem__(self, index):
#         sender_input, sender_label = super(ColoredFiguresDataset2, self).__getitem__(index)
#         distractor_image, distractor_label = self.get_distractor(sender_input, sender_label)
#         return sender_input, int(distractor_label == sender_label), distractor_image


def prepare_datasets(n_features=5, n_attributes=2):
    assert (n_features, n_attributes) == (5, 2)
    train_path = 'visual_compositionality/train'
    test_path = 'visual_compositionality/test'

    train_dataset = ColoredFiguresDataset(root=train_path, transform=ToTensor())
    test_dataset = ColoredFiguresDataset(root=test_path, transform=ToTensor())
    full_dataset = data.ConcatDataset([train_dataset, test_dataset])
    return full_dataset, train_dataset, test_dataset


if __name__ == "__main__":
    pass
