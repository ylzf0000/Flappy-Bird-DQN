import copy
import os, sys, random
from collections import deque
from dataclasses import asdict
import wandb

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


class MultiFrameState:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state):
        while len(self.memory) < self.memory.maxlen:
            self.memory.append(state)

    def tolist(self):
        while len(self.memory) < self.memory.maxlen:
            self.memory.append(self.memory[-1])
        return np.array(list(self.memory))

    def __len__(self):
        return len(self.memory)

    def size(self):
        return len(self.memory)


class AgentDQN_V1:
    def __init__(self, args: RLArguments):
        self.args = args
        self.tau = self.args.tau_start
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flappy_bird_env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        self.memory = ReplayMemory(args.replay_memory_size)  # init some parameters
        self.policy_net = DuelingDeepNetworkSimple(
            input_dim=12
        ).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.target_net.eval()
        self.load_model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=args.learning_rate,
                                           weight_decay=args.weight_decay)

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
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

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
            "lr": self.optimizer.param_groups[0]['lr']
        })

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        if self.args.mode == 'train':
            self.train()
        if self.args.mode == 'test':
            self.test()

    def train(self):
        config = asdict(self.args)
        config["model_struct"] = str(self.target_net)
        self.wandb = wandb.init(
            project="AgentDQN",
            config=config,
        )
        env = self.flappy_bird_env
        global_time_step = 0
        for episode in range(100000):
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
                else:
                    action = self.get_action_softmax(env, state, local_time_step)
                next_state, reward, terminated, _, info = env.step(action)
                next_state = np.array(next_state)
                # print(f'reward: {reward}, terminated: {terminated}')
                total_reward += reward
                # 存储经验
                self.memory.push(state, action, reward, next_state, terminated)
                state = next_state

                # 训练网络（当经验足够时）
                if self.memory.size() >= self.args.batch_size:
                    self.train_one_batch()

                if global_time_step % self.args.update_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.save_model()

                # 衰减ε
                # self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)
                self.wandb.log({
                    "global_time_step": global_time_step,
                    "local_time_step": local_time_step,
                    "episode": episode,
                    "total_reward": total_reward,
                    "reward": reward,
                    # "epsilon": self.epsilon,
                })
                # print(f"TimeStep: {global_time_step}, Episode: {episode},  Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

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
        terminated = False
        while not terminated:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()

            # 执行动作
            next_state, reward, terminated, info, done = env.step(action)
            total_reward += reward
            state = next_state

    def get_action_softmax(self, env, state, local_time_step):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

        # 计算Boltzmann概率
        probabilities = torch.softmax(q_values / self.tau, dim=-1)
        print(f"local_time_step: {local_time_step}, action_probabilities", probabilities)
        self.wandb.log({
            "tau": self.tau
        })

        # 按概率分布选择动作
        action = torch.multinomial(probabilities, 1).item()

        # 衰减温度
        self.tau = max(self.args.tau_end, self.tau * self.args.tau_decay)
        return action


def main():
    parser = get_argparser_for_model_arguments()
    args = parser.parse_args()
    args = RLArguments(**vars(args))
    agent = AgentDQN_V1(args)
    agent.run()


if __name__ == '__main__':
    main()
