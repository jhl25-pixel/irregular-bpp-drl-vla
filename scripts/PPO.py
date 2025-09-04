import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch
import sys
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.env import MujocoPackingEnv, build_the_env
from env.state import N_OBJ_State
from env.param import param

#both actor and value network are state transformation
class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):

        super(PolicyNetwork, self).__init__()

        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),  # 注意大写
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):

        features = self.shared_layers(state)
        mean = self.mean_layer(features)
        log_std = self.log_std(features)
        return mean, log_std

class ValueNetwork(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.value = nn.Sequential([

            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ])

    def forward(self, state):
        return self.value(state)


class PPOModel:
    '''
    This model is the conpartment of the environment. 

    -> environment provides the interaction

    -> the ppo agent provides the learnable object.

    ------> both this two compose of the whole model

    The input is: The overall network architecture

    The output is: The final decision network
    '''
    def __init__(self, state_dim, hidden_dim, action_dim, gamma=0.99, eps_clip=0.2, k_epochs=4, lr=3e-4, entropy_coef=0.01):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef 

        # 初始化网络
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.old_policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        
        # 复制当前策略到旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        self.memory = []


    def select_action(self, state):
        """根据当前策略选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, log_std = self.old_policy(state)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.squeeze().numpy(), log_prob.item()

        
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def compute_advantages(self, rewards, values, next_values, dones):
        """计算优势函数（GAE）"""
        advantages = []
        advantage = 0
        
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            advantage = td_error + self.gamma * 0.95 * advantage * (1 - dones[i])  # GAE with λ=0.95
            advantages.insert(0, advantage)
        
        return torch.FloatTensor(advantages)
    
    def update(self):
        """更新策略和价值网络"""
        if len(self.memory) == 0:
            return
        
        # 提取经验数据
        states = torch.FloatTensor([t[0] for t in self.memory])
        actions = torch.FloatTensor([t[1] for t in self.memory])
        rewards = [t[2] for t in self.memory]
        next_states = torch.FloatTensor([t[3] for t in self.memory])
        dones = [t[4] for t in self.memory]
        old_log_probs = torch.FloatTensor([t[5] for t in self.memory])
        
        # 计算状态价值
        values = self.value(states).squeeze()
        next_values = self.value(next_states).squeeze()
        
        # 计算优势和目标价值
        advantages = self.compute_advantages(rewards, values.detach().numpy(), 
                                           next_values.detach().numpy(), dones)
        returns = advantages + values.detach()
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.k_epochs):
            # 计算新的策略概率
            mean, log_std = self.policy(states)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # 计算概率比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # 价值函数损失
            value_loss = F.mse_loss(self.value(states).squeeze(), returns)
            
            # 反向传播
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optimizer.step()
        
        # 更新旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # 清空经验缓冲区
        self.memory = []
        
def train_ppo(episodes=1000, max_steps=50):

    env = build_the_env()
    state_dim = env.state_agent.get_state_shape()
    state = env.state_agent.return_the_state()
    agent = PPOModel(state_dim=state_dim, hidden_dim=param.hidden_dim, action_dim=3)
    scores = deque(maxlen=100)
    episode_scores = []

    for episode in range(episodes):

        episode_score = 0

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            done = done
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            episode_score += reward
            
            if done:
                break
        
        # 更新网络
        agent.update()
        
        scores.append(episode_score)
        episode_scores.append(episode_score)
        
        # 打印进度
        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    return episode_scores
        
if __name__ == "__main__":
    # 训练智能体
    scores = train_ppo()
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('PPO Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()
    
    print("PPO训练完成！")