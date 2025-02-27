import numpy as np
import torch
import torch.nn as nn

class DeepNetWorkV1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(180, 90),
            nn.GELU(),
            nn.Linear(90, 50),
            nn.GELU(),
            nn.Linear(50,16),
            nn.GELU(),
            nn.Linear(16,2),
        )

    def forward(self, x):
        return self.fc(x)


class DeepNetWork(nn.Module):
    """  神经网络结构
    """

    def __init__(self, ):
        super(DeepNetWork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        """ 前向传播
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.out(x)
