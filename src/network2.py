import torch.utils.data.dataset
import torch.nn as nn

import config
from dataset import Dataset

device = config.device()

#experimento red 2
class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)).to(device)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)).to(device)

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)).to(device)

        self.fc_layers = nn.Sequential(
            nn.Linear(102400, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 5)
        ).to(device)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out