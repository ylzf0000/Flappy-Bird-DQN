import copy
import json
import os, sys, random
from collections import deque
from dataclasses import asdict
import numpy as np
import flappy_bird_gymnasium
import gymnasium
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from RLArguments import get_argparser_for_model_arguments, RLArguments
from util import init_logger, seed_everything, count_parameters
from model import *


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size)
        self.data = [None] * capacity
        self.data_ptr = 0  # 当前写入的叶子节点位置

    def add(self, priority, data):
        """添加数据及对应的优先级"""
        idx = self.data_ptr + self.capacity - 1  # 计算叶子节点索引
        self.data[self.data_ptr] = data
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity  # 循环覆盖

    def update(self, idx, priority):
        """更新指定索引的优先级"""
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        # 向上回溯更新父节点
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, s):
        """根据采样值s获取叶子节点"""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):  # 到达叶子节点
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]  # 根节点存储总优先级


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0  # 初始最大优先级

    def add(self, experience):
        """添加新经验，使用当前最大优先级初始化"""
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        """采样一批经验"""
        indices = []
        experiences = []
        weights = []
        total_priority = self.tree.total_priority

        # 分段采样保证均匀覆盖
        segment = total_priority / batch_size

        for i in range(batch_size):
            # 在每个分段中随机采样
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, exp = self.tree.get(s)

            indices.append(idx)
            experiences.append(exp)

            # 计算重要性采样权重
            sampling_prob = priority / total_priority
            weight = (sampling_prob * self.tree.capacity) ** -self.beta
            weights.append(weight)

        # 归一化权重
        weights = np.array(weights) / np.max(weights)
        return indices, experiences, weights

    def update_priorities(self, indices, deltas):
        """用新的TD误差更新优先级"""
        for idx, delta in zip(indices, deltas):
            priority = (abs(delta) + 1e-5) ** self.alpha
            self.tree.update(idx, priority)
            # 动态更新最大优先级
            if priority > self.max_priority:
                self.max_priority = priority


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


class AgentDQN_V1:
    def __init__(self, args: RLArguments):
        self.args = args
        self.tau = self.args.tau_start
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger, formatted_time = init_logger("logger", f'{args.mode}_{self.args.experience_name}')
        self.logger = logger
        self.start_time = formatted_time
        self.memory = PrioritizedReplayBuffer(args.replay_memory_size)  # init some parameters
        self.policy_net = eval(self.args.model_class)(input_dim=12).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        self.load_model()
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),
                                           lr=args.learning_rate,
                                           weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=self.args.train_steps,
                                           eta_min=1e-8)
        seed_everything(42)

    def log(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def save_model(self, global_time_step, pipe, reward):
        model_name = self.args.experience_name
        dir1 = "./checkpoints"
        dir2 = f"{model_name}_{self.start_time}"
        dir = os.path.join(dir1, dir2)
        os.makedirs(dir, exist_ok=True)
        filename = f'{global_time_step}_P{pipe}_R{int(reward)}.pth'
        newest_filename = "newest.pth"
        path = os.path.join(dir, filename)
        newest_path = os.path.join(dir, newest_filename)
        torch.save(self.policy_net.state_dict(), path)
        torch.save(self.policy_net.state_dict(), newest_path)
        self.log(f"save model param to {path} and {newest_path}.")

    def load_model(self):
        if self.args.load_model_checkpoint and os.path.exists(self.args.load_model_checkpoint):
            d = torch.load(self.args.load_model_checkpoint, weights_only=True)
            self.policy_net.load_state_dict(d)
            self.target_net.load_state_dict(d)
            self.log(f"load model param from {self.args.load_model_checkpoint}.")

    def train_one_batch(self):
        """
        # 采样经验
        indices, batch, weights = buffer.sample(batch_size=64)

        # 计算TD-error并更新网络
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            target_q = rewards + gamma * target_net(next_states).max(1)[0] * (1 - dones)
        current_q = q_net(states).gather(1, actions)
        loss = (current_q - target_q).pow(2) * weights  # 加权损失

        # 更新优先级
        new_deltas = (target_q - current_q).abs().detach().numpy()
        buffer.update_priorities(indices, new_deltas)

        :return:
        """
        # 从缓冲区采样
        self.policy_net.train()
        self.target_net.eval()
        indices, batch, weights = self.memory.sample(self.args.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 计算Q值
        current_q: torch.tensor = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        # Double DQN
        next_actions = self.policy_net(next_states).argmax(1)
        next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze().detach()

        # next_q = self.target_net(next_states).max(1)[0].detach()
        target_q: torch.tensor = rewards + self.args.gamma * next_q * (1 - dones)
        # 计算损失
        loss = self.criterion(current_q, target_q)

        # 更新优先级
        new_deltas = (target_q - current_q).abs().cpu().detach().numpy()
        self.memory.update_priorities(indices, new_deltas)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        return {
            "loss": loss,
            "current_q": current_q.mean(),
            "target_q": target_q.mean(),
            "delta_q": target_q.mean() - current_q.mean(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self):
        config = asdict(self.args)
        config["model_struct"] = str(self.target_net)
        config["count_parameters"] = count_parameters(self.policy_net),
        self.wandb = wandb.init(
            project=self.args.wandb_project,
            name=self.args.experience_name,
            config=config,
        )
        env = gymnasium.make("FlappyBird-v0", render_mode=self.args.render_mode, use_lidar=False)
        global_time_step = 0
        episode = 0
        delta_tau = (self.args.tau_start - self.args.tau_end) / self.args.train_steps
        delta_epsilon = (self.args.epsilon_start - self.args.epsilon_end) / self.args.train_steps
        epsilon = self.args.epsilon_start
        decision_interval_step = self.args.decision_interval_percent * self.args.train_steps if 0 < self.args.decision_interval_percent < 1 else int(self.args.decision_interval_percent)
        while global_time_step < self.args.train_steps:
            episode += 1
            state, _ = env.reset()
            state = np.array(state)
            total_reward = 0
            local_time_step = 0
            passed_pipe = 0
            terminated = False
            while not terminated and global_time_step < self.args.train_steps:
                local_time_step += 1
                global_time_step += 1

                dec = self.args.decision_interval_percent > 0 and self.args.decision_interval > 0 and local_time_step % self.args.decision_interval != 0
                if dec and global_time_step <= decision_interval_step:
                    action = 0
                # elif episode % self.args.greedy_episodes == 0:
                #     action = self.get_action_greedy(state)
                # self.log(
                #     f"episode: {episode}, GTS: {global_time_step}, LTS: {local_time_step}, greedy_action: {action}")
                else:
                    # action = self.get_action_epsilon_greedy(env, state)
                    probabilities, action = self.get_action_softmax_sample(state)
                    # self.log(
                    #     f"episode: {episode}, GTS: {global_time_step}, LTS: {local_time_step}, AP: {probabilities.cpu().detach().numpy()}")

                next_state, reward, terminated, _, info = env.step(action)
                next_state = np.array(next_state)
                total_reward += reward
                if reward >= np.float64(1.0):
                    passed_pipe += 1
                # 存储经验
                self.memory.add((state, action, reward, next_state, terminated))
                state = next_state

                # 衰减温度
                self.tau = max(self.args.tau_end, self.tau - delta_tau)

                # 衰减ε
                epsilon = max(self.args.epsilon_end, epsilon - delta_epsilon)

                # 训练网络（当经验足够时）
                # if self.memory.size() >= self.args.batch_size:
                if global_time_step >= 4 * self.args.batch_size:
                    train_log_dict = self.train_one_batch()
                else:
                    train_log_dict = {}

                # 动量更新target网络
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(self.args.target_net_update_tau * policy_param.data + (
                            1 - self.args.target_net_update_tau) * target_param.data)

                # if global_time_step % self.args.target_net_update_steps == 0:
                #     self.target_net.load_state_dict(self.policy_net.state_dict())

                # 测试模型性能并保存参数
                test_log_dict = {}
                if global_time_step % self.args.save_model_steps == 0:
                    res_list = [self.test() for _ in range(5)]
                    total_rewards, pipes = zip(*res_list)
                    reward_mean = np.mean(total_rewards)
                    pipes_mean = np.mean(pipes)
                    self.save_model(global_time_step, pipes_mean, reward_mean)
                    test_log_dict["Test/PipesMean"] = pipes_mean
                    test_log_dict["Test/RewardMean"] = reward_mean
                    self.log(
                        f'GTS: {global_time_step}, Episode: {episode}, TestPipesMean: {round(pipes_mean, 1)}, TestRewardMean: {round(reward_mean, 1)}, TestPipes: {list(map(round, pipes))}, TestReward: {list(map(round, total_rewards))}')

                self.wandb.log(train_log_dict | test_log_dict | {
                    "global_time_step": global_time_step,
                    "local_time_step": local_time_step,
                    "episode": episode,
                    "total_reward": total_reward,
                    "reward": reward,
                    "passed_pipe": passed_pipe,
                    "tau": self.tau
                    # "epsilon": self.epsilon,
                })
        env.close()

    def test(self):
        env = gymnasium.make("FlappyBird-v0", render_mode=self.args.render_mode, use_lidar=False)
        state, _ = env.reset()
        total_reward = 0
        pipes = 0
        terminated = False
        while not terminated:
            action = self.get_action_greedy(state)
            next_state, reward, terminated, info, done = env.step(action)
            total_reward += reward
            pipes += int(reward >= 1.0)
            state = next_state

        env.close()
        return total_reward, pipes

    def get_action_greedy(self, state):
        q_values = self.get_action_q_values(state)
        action = q_values.argmax().item()
        return action

    def get_action_softmax_sample(self, state):
        q_values = self.get_action_q_values(state)
        # 计算Boltzmann概率
        probabilities = torch.softmax(q_values / self.tau, dim=-1)
        # 按概率分布选择动作
        action = torch.multinomial(probabilities, 1).item()
        return probabilities, action

    def get_action_q_values(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
        return q_values

    def get_action_epsilon_greedy(self, env, state):
        # 选择动作（ε-greedy）
        r = np.random.rand()
        if r < self.epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            action = self.get_action_greedy(state)
        return action

    def run(self):
        args = asdict(self.args)
        args = json.dumps(args, indent=4, sort_keys=True, ensure_ascii=False)
        self.log(f'{args}')
        if self.args.mode == 'train':
            self.train()
        if self.args.mode == 'test':
            total_reward, pipes = self.test()
            self.log(f'pipes: {pipes}, total_reward: {total_reward}')


def main():
    parser = get_argparser_for_model_arguments()
    args = parser.parse_args()
    args = RLArguments(**vars(args))
    agent = AgentDQN_V1(args)
    agent.run()


if __name__ == '__main__':
    main()
