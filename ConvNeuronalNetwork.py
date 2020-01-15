import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, input, output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input, out_channels=32, kernel_size=8, stride=4)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.r2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.r3 = nn.ReLU()
        self.linear1 = nn.Linear(64*7*7, 512)
        self.r4 = nn.ReLU()
        self.linear2 = nn.Linear(512, output)

    def forward(self, x):
        x = self.r1(self.conv1(x))
        x = self.r2(self.conv2(x))
        x = self.r3(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = self.r4(self.linear1(x))
        x = self.linear2(x)

        return x.squeeze(0)