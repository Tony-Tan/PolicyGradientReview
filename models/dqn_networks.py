import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch


import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNAtari(nn.Module):
    def __init__(self, input_channel_size: int, output_size: int):
        super(DQNAtari, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.fc_2 = nn.Linear(512, output_size)

        # Apply He initialization to the convolutional layers
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        # Optional: Zero-initialize the biases
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)

        # Apply Xavier initialization to the fully connected layers
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.fc_2.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.fc_2.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # self.bn1(x))
        x = self.conv2(x)
        x = F.relu(x)  # self.bn2(x))
        x = self.conv3(x)
        x = F.relu(x)  # self.bn3(x))
        x = x.view(-1, 3136)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x


# advanced network
class DuelingDQNAtari(DQNAtari):
    def __init__(self, input_channel_size: int, output_size: int):
        super(DuelingDQNAtari, self).__init__(input_channel_size, output_size)
        # for advanced output
        self.fc_3 = nn.Linear(512, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # self.bn1(x))
        x = self.conv2(x)
        x = F.relu(x)  # self.bn2(x))
        x = self.conv3(x)
        x = F.relu(x)  # self.bn3(x))
        x = x.view(-1, 3136)
        x = self.fc(x)
        features = F.relu(x)
        a_values = self.fc_2(features)
        s_values = self.fc_3(features)
        hat_s = a_values.mean(dim=1, keepdim=True)
        return a_values - hat_s + s_values
