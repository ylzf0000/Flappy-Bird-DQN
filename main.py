# import flappy_bird_gymnasium
# import gymnasium
#
# env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
#
# obs, _ = env.reset()
# while True:
#     # Next action:
#     # (feed the observation to your agent here)
#     action = env.action_space.sample()
#
#     # Processing:
#     obs, reward, terminated, _, info = env.step(action)
#
#     # Checking if the player is still alive
#     if terminated:
#         break
#
# env.close()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import flappy_bird_gym

# 超参数
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
TARGET_UPDATE = 100
MEMORY_CAPACITY = 10000


# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 初始化环境和模型
env = flappy_bird_gym.make("FlappyBird-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_CAPACITY)
epsilon = EPSILON_START

# 训练循环
episode_rewards = []
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # 选择动作（ε-greedy）
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        # 执行动作
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # 存储经验
        memory.push(state, action, reward, next_state, done)

        # 更新状态
        state = next_state

        # 训练网络（当经验足够时）
        if len(memory) >= BATCH_SIZE:
            # 从缓冲区采样
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 转换为张量
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            # 计算Q值
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))
            next_q = target_net(next_states).max(1)[0].detach()
            target_q = rewards + GAMMA * next_q * (1 - dones)

            # 计算损失
            loss = nn.MSELoss()(current_q.squeeze(), target_q)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 衰减ε
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    episode_rewards.append(total_reward)
    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()