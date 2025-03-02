import copy
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
        logger, formatted_time = init_logger("logger", self.args.model_name)
        self.logger = logger
        self.start_time = formatted_time
        self.flappy_bird_env = gymnasium.make("FlappyBird-v0", render_mode=self.args.render_mode, use_lidar=False)
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
                                           T_max=int(1e6),
                                           eta_min=1e-8)
        seed_everything(42)

    def log(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def save_model(self, global_time_step):
        model_name = self.args.model_name
        dir1 = "./checkpoints"
        dir2 = f"{model_name}_{self.start_time}"
        dir = os.path.join(dir1, dir2)
        os.makedirs(dir, exist_ok=True)
        filename = f'{global_time_step}.pth'
        path = os.path.join(dir, filename)
        torch.save(self.policy_net.state_dict(), path)
        self.log(f"save model param to {path}.")

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

        next_actions = self.policy_net(next_states).argmax(1)
        next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze().detach()

        # next_q = self.target_net(next_states).max(1)[0].detach()
        target_q: torch.tensor = rewards + self.args.gamma * next_q * (1 - dones)

        # 计算损失
        loss = self.criterion(current_q, target_q)

        wandb.log({
            "loss": loss,
            "current_q": current_q.mean(),
            "target_q": target_q.mean(),
            "delta_q": target_q.mean() - current_q.mean(),
            "lr": self.scheduler.get_last_lr()[0],
        })

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()

    def run(self):
        if self.args.mode == 'train':
            self.train()
        if self.args.mode == 'test':
            self.test()

    def train(self):
        config = asdict(self.args)
        config["model_struct"] = str(self.target_net)
        config["count_parameters"] = count_parameters(self.policy_net),
        self.wandb = wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=config,
        )
        env = self.flappy_bird_env
        global_time_step = 0
        episode = 0
        delta_tau = (self.args.tau_start - self.args.tau_end) / self.args.train_steps
        while global_time_step < self.args.train_steps:
            episode += 1
            state, _ = self.flappy_bird_env.reset()
            state = np.array(state)
            total_reward = 0
            local_time_step = 0
            terminated = False
            while not terminated:
                local_time_step += 1
                global_time_step += 1
                if local_time_step % self.args.decision_interval != 0:
                    action = 0
                elif episode % self.args.greedy_episodes == 0:
                    action = self.get_action_greedy(state)
                    self.log(
                        f"episode: {episode}, GTS: {global_time_step}, LTS: {local_time_step}, greedy_action: {action}")
                else:
                    probabilities, action = self.get_action_softmax_sample(state)
                    self.log(
                        f"episode: {episode}, GTS: {global_time_step}, LTS: {local_time_step}, AP: {probabilities.cpu().detach().numpy()}")

                next_state, reward, terminated, _, info = env.step(action)
                next_state = np.array(next_state)
                total_reward += reward
                # 存储经验
                self.memory.push(state, action, reward, next_state, terminated)
                state = next_state

                # 衰减温度
                self.tau = max(self.args.tau_end, self.tau - delta_tau)

                # 训练网络（当经验足够时）
                if self.memory.size() >= self.args.batch_size:
                    self.train_one_batch()

                if global_time_step % self.args.target_net_update_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if global_time_step % self.args.save_model_steps == 0:
                    self.save_model(global_time_step)

                # 衰减ε
                # self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)
                self.wandb.log({
                    "global_time_step": global_time_step,
                    "local_time_step": local_time_step,
                    "episode": episode,
                    "total_reward": total_reward,
                    "reward": reward,
                    "tau": self.tau
                    # "epsilon": self.epsilon,
                })

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
    def test(self):
        env = self.flappy_bird_env
        state, _ = self.flappy_bird_env.reset()
        total_reward = 0
        zz = 0
        terminated = False
        while not terminated:
            action = self.get_action_greedy(state)

            # 执行动作
            next_state, reward, terminated, info, done = env.step(action)
            total_reward += reward
            if reward >= 1.0:
                zz += 1
            print(f'zz: {zz}, total_reward: {total_reward}')
            state = next_state

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


def main():
    parser = get_argparser_for_model_arguments()
    args = parser.parse_args()
    args = RLArguments(**vars(args))
    agent = AgentDQN_V1(args)
    agent.run()


if __name__ == '__main__':
    main()
