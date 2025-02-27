import os, sys, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from RLArguments import get_argparser_for_model_arguments, RLArguments
import flappy_bird_gymnasium
import gymnasium
from model import *


class ReplayMemory:
    """
        在每个timestep下agent与环境交互得到的转移样本 (st,at,rt,st+1) 储存到回放记忆库，
        要训练时就随机拿出一些（minibatch）数据来训练，打乱其中的相关性
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    def size(self):
        return len(self.memory)


class Agent_DQN:
    def __init__(self, args: RLArguments):
        self.current_state = None
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flappy_bird_env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
        self.memory = ReplayMemory(args.replay_memory_size)  # init some parameters
        self.time_step = 0
        # 有epsilon的概率，随机选择一个动作，1-epsilon的概率通过网络输出的Q（max）值选择动作
        self.epsilon = args.epsilon_start
        # 当前值网络, 目标网络
        self.policy_net = DeepNetWorkV1().to(self.device)
        self.target_net = DeepNetWorkV1().to(self.device)
        self.target_net.eval()
        # 加载训练好的模型，在训练的模型基础上继续训练
        self.load_model()
        # 使用均方误差作为损失函数
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)

    def save_model(self):
        if self.args.save_model_checkpoint:
            torch.save(self.policy_net.state_dict(), self.args.save_model_checkpoint)
            print(f"save model param to {self.args.save_model_checkpoint}.")

    def load_model(self):
        if self.args.load_model_checkpoint and os.path.exists(self.args.load_model_checkpoint):
            d = torch.load(self.args.load_model_checkpoint, weights_only=True)
            self.policy_net.load_state_dict(d)
            self.target_net.load_state_dict(d)
            print(f"load model param from {self.args.load_model_checkpoint}.")

    def train_one_batch(self):
        # 从缓冲区采样
        batch = self.memory.sample(self.args.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + self.args.gamma * next_q * (1 - dones)

        # 计算损失
        loss = self.loss_func(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def run(self):
        if self.args.mode == 'train':
            self.train()
        if self.args.mode == 'play':
            self.test()

    def test(self):
        env = self.flappy_bird_env
        state, _ = self.flappy_bird_env.reset()
        total_reward = 0
        terminated = False

        while not terminated:
            action = self.get_action(env, state, greedy=True)

            # 执行动作
            next_state, reward, terminated, truncated, info, done = env.step(action)
            total_reward += reward
            # 更新状态
            state = next_state

    def train(self):
        env = self.flappy_bird_env
        for episode in range(1000):
            state, _ = self.flappy_bird_env.reset()
            total_reward = 0
            terminated = False

            while not terminated:
                action = self.get_action(env, state)

                # 执行动作
                next_state, reward, terminated,info,done = env.step(action)
                total_reward += reward

                # 存储经验
                self.memory.push(state, action, reward, next_state, terminated)

                # 更新状态
                state = next_state

                # 训练网络（当经验足够时）
                if self.memory.size() >= self.args.batch_size:
                    self.train_one_batch()

                # 更新目标网络
                if episode % self.args.update_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                # 每隔UPDATE_TIME轮次，用训练的网络的参数来更新target网络的参数
                if self.time_step % self.args.update_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.save_model()

                # 衰减ε
                self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)

            # episode_rewards.append(total_reward)
            # print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    def get_action(self, env, state, greedy=False):
        # 选择动作（ε-greedy）
        if not greedy or np.random.rand() < self.epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
        return action


def main():
    parser = get_argparser_for_model_arguments()
    args = parser.parse_args()
    args = RLArguments(**vars(args))
    agent = Agent_DQN(args)
    agent.run()


if __name__ == '__main__':
    main()
