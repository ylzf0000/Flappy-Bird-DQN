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
            nn.Linear(50, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.fc(x)


class TransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha = nn.MultiheadAttention(180, 6, 0.0, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(180, 180 * 4),
            nn.GELU(),
            nn.Linear(180 * 4, 180),
        )
        self.norm = nn.RMSNorm(normalized_shape=180)

    def forward(self, x):
        x_norm = self.norm(x)
        h = x + self.mha(x_norm, x_norm, x_norm)[0]
        out = h + self.fc(self.norm(h))
        return out


class DeepNetWorkV2(nn.Module):
    def __init__(self, layer_num, seq_len):
        super().__init__()
        self.positional_embedding = nn.Embedding(seq_len, 180)
        self.transformers = nn.ModuleList([TransformerBlock() for _ in range(layer_num)])
        self.fc = nn.Sequential(
            nn.Linear(180, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )
        self.norm = nn.RMSNorm(normalized_shape=180)

    def forward(self, x):  # B, len, dim
        b, seq_len, dim = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.positional_embedding(pos)
        for transformer in self.transformers:
            x = transformer(x)
        x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2).contiguous()).squeeze()
        x = self.norm(x)
        x = self.fc(x).squeeze()
        return x


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
