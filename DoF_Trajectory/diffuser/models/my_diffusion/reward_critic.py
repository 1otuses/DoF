"""
奖励预测 Critic 网络

将编码器的隐向量 z = (z_env, z_inter) 映射到标量奖励 r。

采用双 Q 网络结构（与 DoF 现有 Critic 风格一致），
以更稳定的方式拟合数据集中的真实奖励信号。

核心公式：
    r_pred = Q(z_env, z_inter)

验证目标：
    L_reward = MSE(r_pred, r_true) -> 证明隐向量保留了奖励相关信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardCritic(nn.Module):
    """
    奖励预测 Critic
    
    使用双 Q 网络 (Q1, Q2)，取 min 以缓解 overestimation。
    
    Args:
        env_dim:       z_env 的维度
        inter_dim:     z_inter 的维度
        hidden_dim:    隐藏层维度
        num_layers:    MLP 层数
        activation:    激活函数 ('mish' | 'relu' | 'silu')
        double_q:      是否使用双 Q 网络
    """
    
    def __init__(
        self,
        env_dim: int,
        inter_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = 'mish',
        double_q: bool = True,
    ):
        super().__init__()
        
        self.env_dim = env_dim
        self.inter_dim = inter_dim
        self.input_dim = env_dim + inter_dim
        self.double_q = double_q
        
        act_fn = {
            'mish': nn.Mish,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
        }[activation]
        
        # --- 构建 MLP 网络 ---
        def _build_q_net():
            layers = []
            in_dim = self.input_dim
            for i in range(num_layers):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    act_fn(),
                ])
                in_dim = hidden_dim
            # 输出标量 Q 值
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)
        
        self.q1_net = _build_q_net()
        if double_q:
            self.q2_net = _build_q_net()
        else:
            self.q2_net = None
    
    def forward(self, z_env: torch.Tensor, z_inter: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            z_env:  环境隐向量   [batch_size, env_dim]
            z_inter:交互隐向量   [batch_size, inter_dim]
            
        Returns:
            q1: Q1 网络预测的奖励 [batch_size, 1]
            q2: Q2 网络预测的奖励 [batch_size, 1] (如果 double_q=True, 否则返回 None)
        """
        z = torch.cat([z_env, z_inter], dim=-1)
        q1 = self.q1_net(z)
        
        if self.double_q:
            q2 = self.q2_net(z)
            return q1, q2
        else:
            return q1, None
    
    def predict(self, z_env: torch.Tensor, z_inter: torch.Tensor) -> torch.Tensor:
        """
        预测奖励（取双 Q 的最小值，用于保守估计）
        
        Args:
            z_env:  环境隐向量   [..., env_dim]
            z_inter:交互隐向量   [..., inter_dim]
            
        Returns:
            r_pred: 预测的奖励标量 [..., 1]
        """
        orig_shape_env = z_env.shape[:-1]
        orig_shape_inter = z_inter.shape[:-1]
        
        z_env_flat = z_env.reshape(-1, self.env_dim)
        z_inter_flat = z_inter.reshape(-1, self.inter_dim)
        
        q1, q2 = self.forward(z_env_flat, z_inter_flat)
        
        if self.double_q:
            r_pred = torch.min(q1, q2)
        else:
            r_pred = q1
        
        # 恢复原始形状
        r_pred = r_pred.reshape(*orig_shape_env, 1)
        return r_pred
    
    def predict_all_agents(
        self,
        z_env_all: torch.Tensor,
        z_inter_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量为所有 agent 预测奖励
        
        Args:
            z_env_all:   [batch_size, n_agents, env_dim]
            z_inter_all: [batch_size, n_agents, inter_dim]
            
        Returns:
            r_pred_all:  [batch_size, n_agents, 1]
        """
        return self.predict(z_env_all, z_inter_all)


def compute_reward_loss(
    critic: RewardCritic,
    z_env: torch.Tensor,
    z_inter: torch.Tensor,
    reward_true: torch.Tensor,
) -> dict:
    """
    计算奖励预测损失
    
    Args:
        critic:      RewardCritic 网络
        z_env:       [batch_size, env_dim]
        z_inter:     [batch_size, inter_dim]
        reward_true: [batch_size, 1] 或 [batch_size]
        
    Returns:
        dict containing 'loss_reward', 'q1_mean', 'q2_mean', etc.
    """
    if reward_true.dim() == 1:
        reward_true = reward_true.unsqueeze(-1)
    
    q1, q2 = critic(z_env, z_inter)
    loss_q1 = F.mse_loss(q1, reward_true)
    
    info = {
        'loss_reward': loss_q1,
        'q1_mean': q1.mean().detach(),
    }
    
    if q2 is not None:
        loss_q2 = F.mse_loss(q2, reward_true)
        info['loss_reward'] = loss_q1 + loss_q2
        info['q2_mean'] = q2.mean().detach()
    
    return info
