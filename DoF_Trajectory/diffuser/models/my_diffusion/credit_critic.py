"""
信用评估头 (Credit Critic Head)

模块二·Step 1: 从共享编码器隐变量 H 计算 Q_i 和 Q_tot。

架构:
    H = Encoder(x_k, k) → [B, T, N, hidden_dim] 共享隐变量
    
    AgentLocalCritic:
        H_i = H[:,:,:,i,:] → MLP → Q_i  [B, 1]  per-agent 局部价值
    
    QMixer (单调混合):
        Q_1, ..., Q_N → MixingNet(Q_1..Q_N | s) → Q_tot  [B, 1]
        混合网络权重由全局状态 s 通过 hypernetwork 生成,保证单调性
    
    CQLLoss:
        保守 Q 学习损失,防止离线 OOD 高估。
        L_cql = α * (E_{s∼D}[log Σ_a exp Q(s,a)] - E_{a∼D}[Q(s,a)])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentLocalCritic(nn.Module):
    """
    每个 agent 的局部信用评估头。
    输入个体隐向量 H_i，输出标量 Q_i。
    """

    def __init__(
        self,
        n_agents: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.n_agents = n_agents

        # 共享参数的 MLP: 所有 agent 使用同一套参数
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [B, N, hidden_dim] 每个 agent 的隐向量
        Returns:
            q_list: list of [B, 1] per-agent Q 值
        """
        q_list = []
        for i in range(self.n_agents):
            q_i = self.q_net(h[:, i])  # [B, 1]
            q_list.append(q_i)
        return q_list


class QMixer(nn.Module):
    """
    QMIX 风格的单调混合网络。
    将 per-agent Q_i 混合为 Q_tot，权重由全局 state 通过 hypernetwork 生成。
    
    保证单调性: 混合权重非负 (通过 abs()).
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Hypernetwork 从状态生成混合权重和偏置
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim),
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q_list, state):
        """
        Args:
            q_list: list of [B, 1] per-agent Q 值
            state:  [B, state_dim] 全局状态
        Returns:
            q_tot: [B, 1] 混合后的全局 Q 值
        """
        b = q_list[0].shape[0]
        q = torch.cat(q_list, dim=-1)  # [B, N]

        # 第一层 (N → hidden)
        w1 = torch.abs(self.hyper_w1(state)).view(b, self.n_agents, -1)
        b1 = self.hyper_b1(state).view(b, 1, -1)
        q_hidden = F.elu(torch.bmm(q.unsqueeze(1), w1) + b1)  # [B, 1, hidden]

        # 第二层 (hidden → 1)
        w2 = torch.abs(self.hyper_w2(state)).view(b, -1, 1)
        b2 = self.hyper_b2(state).view(b, 1, 1)
        q_tot = torch.bmm(q_hidden, w2) + b2  # [B, 1, 1]
        return q_tot.squeeze(-1)  # [B, 1]


class CQLLoss(nn.Module):
    """
    保守 Q 学习 (CQL) 损失。

    L_cql = α * (log Σ_a exp Q(s, a) - Q(s, a_true))

    对离线数据集中的状态 s,惩罚 OOD 动作的虚高 Q 值。
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        q_tot_current: torch.Tensor,
        q_tot_target: torch.Tensor,
        q_tot_obs: torch.Tensor,
    ):
        """
        Args:
            q_tot_current: [B, 1] 当前网络预测的 Q_tot
            q_tot_target:  [B, 1] TD target
            q_tot_obs:     [B, 1] 观察到动作的 Q_tot (应该就是 q_tot_current)
        Returns:
            loss: 标量, L_cql
        """
        # L_td 部分: TD error
        td_loss = F.mse_loss(q_tot_current, q_tot_target)

        # L_cql 部分: log-sum-exp 减去 observed
        cql_loss = self.alpha * (
            torch.logsumexp(q_tot_current, dim=0) - q_tot_obs.mean()
        )

        return td_loss + cql_loss
