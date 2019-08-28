from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from visual_compositionality.visual_data import prepare_datasets


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 50)
        )
        self.classifier_1 = nn.Linear(50, 5)
        self.classifier_2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.embedd(x)
        return self.classifier_1(x), self.classifier_2(x)

    def embedd(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Vision(nn.Module):
    def __init__(self):
        super(Vision, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*29*29, 25)
        self.classifier_1 = nn.Linear(25, 5)
        self.classifier_2 = nn.Linear(25, 5)

    @classmethod
    def from_pretrained(cls, path: str):
        vision = cls()
        vision.load_state_dict(torch.load(path))
        vision.train(False)
        return vision

    def forward(self, x):
        x = self.embedd(x)
        return self.classifier_1(x), self.classifier_2(x)

    def embedd(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50*29*29)
        x = F.relu(self.fc1(x))
        return x


def entangled_loss(targets, receiver_output_1, receiver_output_2):
    acc_1 = (receiver_output_1.argmax(dim=1) == targets[:, 0]).detach().float()
    acc_2 = (receiver_output_2.argmax(dim=1) == targets[:, 1]).detach().float()
    loss_1 = F.cross_entropy(receiver_output_1, targets[:, 0], reduction="none")
    loss_2 = F.cross_entropy(receiver_output_2, targets[:, 1], reduction="none")
    acc = (acc_1 * acc_2).mean(dim=0)
    loss = loss_1 + loss_2
    return loss, {f'accuracy': acc.item(),
                  f'first_accuracy': acc_1.mean(dim=0).item(),
                  f'second_accuracy': acc_2.mean(dim=0).item()}


if __name__ == "__main__":
    full, _, _ = prepare_datasets(5, 2)
    loader = DataLoader(full, batch_size=32, shuffle=True)
    model = Vision()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(100):
        acc = 0
        for i, (input, target) in enumerate(tqdm(loader)):
            output_1, output_2 = model(input)
            loss, logs = entangled_loss(target, output_1, output_2)
            acc += logs['accuracy']
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        print(acc/i)
        if (acc/i) > 0.99:
            break
    torch.save(model.state_dict(), 'vision_model.pth')