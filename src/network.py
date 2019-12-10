import torch.utils.data.dataset
import torch.nn as nn
import config

device = config.device()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=2, padding=1).to(device)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=1).to(device)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1).to(device)

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=500).to(device)
        self.fc2 = nn.Linear(in_features=500, out_features=50).to(device)
        self.fc3 = nn.Linear(in_features=50, out_features=2).to(device)

    def forward(self, x):
        x = torch.relu(self.conv1(x)).to(device)
        x = nn.functional.max_pool2d(x, 2).to(device)
        x = torch.relu(self.conv2(x)).to(device)
        x = nn.functional.max_pool2d(x, 2).to(device)
        x = torch.relu(self.conv3(x)).to(device)
        x = nn.functional.max_pool2d(x, 2).to(device)
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x)).to(device)
        x = torch.relu(self.fc2(x)).to(device)
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
