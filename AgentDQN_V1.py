import copy
import json
import os, sys, random
import time
import datetime
from collections import deque
from dataclasses import asdict

from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from RLArguments import get_argparser_for_model_arguments, RLArguments
import flappy_bird_gymnasium
import gymnasium
from model import *
from util import init_logger, seed_everything, count_parameters


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
        self.memory = ReplayMemory(args.replay_memory_size)  # init some parameters
        self.policy_net = eval(self.args.model_class)(input_dim=12).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        self.load_model()
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
        filename = f'{global_time_step}_{pipe}_{int(reward)}.pth'
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
        # 从缓冲区采样
        batch = self.memory.sample(self.args.batch_size)
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
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
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
                if global_time_step <= self.args.decision_interval_percent * self.args.train_steps and local_time_step % self.args.decision_interval != 0:
                    action = 0
                elif episode % self.args.greedy_episodes == 0:
                    action = self.get_action_greedy(state)
                    # self.log(
                    #     f"episode: {episode}, GTS: {global_time_step}, LTS: {local_time_step}, greedy_action: {action}")
                else:
                    probabilities, action = self.get_action_softmax_sample(state)
                    # self.log(
                    #     f"episode: {episode}, GTS: {global_time_step}, LTS: {local_time_step}, AP: {probabilities.cpu().detach().numpy()}")

                next_state, reward, terminated, _, info = env.step(action)
                next_state = np.array(next_state)
                total_reward += reward
                if reward >= np.float64(1.0):
                    passed_pipe += 1
                # 存储经验
                self.memory.push(state, action, reward, next_state, terminated)
                state = next_state

                # 衰减温度
                self.tau = max(self.args.tau_end, self.tau - delta_tau)

                # 训练网络（当经验足够时）
                if self.memory.size() >= self.args.batch_size:
                    train_log_dict = self.train_one_batch()
                else:
                    train_log_dict = {}

                # 动量更新target网络
                # for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                #     target_param.data.copy_(self.args.target_net_update_tau * policy_param.data + (
                #             1 - self.args.target_net_update_tau) * target_param.data)

                if global_time_step % self.args.target_net_update_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                # 测试模型性能并保存参数
                test_log_dict = {}
                if global_time_step % self.args.save_model_steps == 0:
                    res_list = [self.test() for _ in range(5)]
                    total_rewards, pipes = zip(*res_list)
                    reward_mean = np.mean(total_rewards)
                    pipes_mean = np.mean(pipes)
                    self.save_model(global_time_step, pipes_mean, reward_mean)
                    self.log(
                        f'GTS: {global_time_step}, Episode: {episode}, TestPipesMean: {pipes_mean}, TestRewardMean: {reward_mean}, TestPipes: {pipes}, TestReward: {total_rewards}')
                    test_log_dict["Test/PipesMean"] = pipes_mean
                    test_log_dict["Test/RewardMean"] = reward_mean

                # 衰减ε
                # self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)
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
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
        return q_values

    # def get_action(self, env, state):
    #     self.frame_counter += 1
    #     if self.frame_counter % self.args.decision_interval != 0:
    #         return 0  # 非决策帧不跳跃
    #
    #     # 选择动作（ε-greedy）
    #     r = np.random.rand()
    #     if r < self.epsilon:
    #         action = env.action_space.sample()  # 随机探索
    #     else:
    #         with torch.no_grad():
    #             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #             q_values = self.policy_net(state_tensor)
    #             action = q_values.argmax().item()
    #     return action

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
