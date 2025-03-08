import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, 0.1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm = nn.RMSNorm(normalized_shape=embed_dim)

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


class DuelingDeepNetwork(nn.Module):
    def __init__(self, layer_num, seq_len, input_dim, embed_dim, num_heads):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.positional_embedding = nn.Embedding(seq_len, embed_dim)
        self.transformers = nn.ModuleList(
            [TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(layer_num)])
        self.advantage_stream = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.norm = nn.RMSNorm(normalized_shape=embed_dim)

    def forward(self, x):  # B, len, dim
        x = self.fc1(x)
        b, seq_len, dim = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.positional_embedding(pos)
        for transformer in self.transformers:
            x = transformer(x)
        x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2).contiguous()).squeeze()
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = values + (advantages - advantages.mean())
        return q_values.squeeze()


class DuelingDeepNetworkSimple(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean())
        return q_values


class DuelingDeepNetworkSimpleV2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        dim = 256
        multiply = 4
        self.feature = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(dim, dim // multiply),
            nn.LayerNorm(dim // multiply),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // multiply, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(dim, dim // multiply),
            nn.LayerNorm(dim // multiply),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // multiply, 2)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean())
        return q_values


class DuelingDeepNetworkSimpleV3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        dim = 512
        multiply = 4

        self.feature1 = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.feature2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(dim, dim // multiply),
            # nn.LayerNorm(dim // multiply),
            nn.GELU(),
            nn.Linear(dim // multiply, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(dim, dim // multiply),
            # nn.LayerNorm(dim // multiply),
            nn.GELU(),
            nn.Linear(dim // multiply, 2)
        )

    def forward(self, x):
        features = self.feature1(x)
        features = features + self.feature2(features)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean())
        return q_values


class DuelingDeepNetworkSimpleV4(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        dim = 512
        multiply = 4
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(dim, dim // multiply),
            # nn.LayerNorm(dim // multiply),
            nn.GELU(),
            nn.Linear(dim // multiply, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(dim, dim // multiply),
            # nn.LayerNorm(dim // multiply),
            nn.GELU(),
            nn.Linear(dim // multiply, 2)
        )

    def forward(self, x):
        features1 = self.fc1(x)
        features2 = features1 + self.fc2(features1)
        features3 = features2 + self.fc3(features2)
        values = self.value_stream(features3)
        advantages = self.advantage_stream(features3)
        q_values = values + (advantages - advantages.mean())
        return q_values


class DQN(nn.Module):
    def __init__(self, input_dim=12, action_dim=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.LayerNorm(128)  # 批量归一化加速收敛
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.1)  # 防止过拟合
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


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
class ActorCritic(nn.Module):
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=256):
        super().__init__()
        # Shared Feature Extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Actor Network (Policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic Network (Value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value