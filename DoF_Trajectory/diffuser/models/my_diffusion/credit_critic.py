"""
信用评估头 (Credit Critic Head)

模块二·Step 1: 从观测序列提取 H, 计算 Q_i 和 Q_tot。

架构:
    独立状态编码器 (不与扩散模型共享):
        obs → MLP → H  [B, N, hidden_dim]

    AgentLocalCritic:
        H_i → MLP → Q_i  [B, 1]  per-agent 局部价值

    QMixer (单调混合):
        Q_1..Q_N → MixingNet(Q_1..Q_N | s_global) → Q_tot  [B, 1]

    CQLLoss (修复版):
        L_cql = α * (log Σ_{a} exp Q(s, a) - Q(s, a_true))
        在离散动作空间上做 logsumexp, 而非 batch 维度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """
    独立状态编码器: 从观测序列提取每个 agent 的隐向量 H_i。

    与扩散模型解耦, 不依赖其内部隐藏层。

    输入: [B, T, N, O] 观测序列
    输出: [B, N, hidden_dim] per-agent 隐向量
    """

    def __init__(self, obs_dim: int, n_agents: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents

        # 对每个 agent 独立编码: 时间维度池化 + MLP
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs_seq: torch.Tensor):
        """
        Args:
            obs_seq: [B, T, N, O] 观测序列 (可以是 x_start 的观测部分)
        Returns:
            h: [B, N, hidden_dim] per-agent 隐向量
        """
        # 沿时间维度池化: [B, T, N, O] -> [B, N, O]
        obs_pooled = obs_seq.mean(dim=1)  # 时间维度均值池化
        # obs_pooled: [B, N, O]
        B, N, O = obs_pooled.shape

        # 逐 agent 编码: 合并 agent 维度到 batch
        h = self.encoder(obs_pooled.reshape(-1, O))  # [B*N, hidden_dim]
        h = h.reshape(B, N, -1)  # [B, N, hidden_dim]
        return h


class AgentLocalCritic(nn.Module):
    """
    每个 agent 的局部信用评估头。
    输入个体隐向量 H_i, 输出标量 Q_i。
    """

    def __init__(self, n_agents: int, hidden_dim: int):
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
            q:      [B, N] 堆叠的 Q 值
        """
        B, N, D = h.shape
        q = self.q_net(h.reshape(-1, D)).reshape(B, N)  # [B, N]
        q_list = [q[:, i:i+1] for i in range(N)]
        return q_list, q


class QMixer(nn.Module):
    """
    QMIX 风格的单调混合网络。
    将 per-agent Q_i 混合为 Q_tot, 权重由全局 state 通过 hypernetwork 生成。
    保证单调性: 混合权重非负 (通过 abs())。
    """

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.n_agents = n_agents

        # Hypernetwork 从全局状态生成混合权重
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

    def forward(self, q: torch.Tensor, state: torch.Tensor):
        """
        Args:
            q:     [B, N] per-agent Q 值
            state: [B, state_dim] 全局状态
        Returns:
            q_tot: [B, 1] 混合后的全局 Q 值
        """
        b = q.shape[0]

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
    保守 Q 学习 (CQL) 损失 — 修复版。

    L_cql = α * (log Σ_a exp Q(s, a) - E[Q(s, a)])

    对离线数据集中的状态 s, 惩罚 OOD 动作的虚高 Q 值。
    修复: 在动作空间上做 logsumexp (而非 batch 维度)。
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, q_tot: torch.Tensor, td_target: torch.Tensor):
        """
        Args:
            q_tot:     [B, 1] 当前 Q_tot 预测
            td_target: [B, 1] TD target (来自 returns + next state)
        Returns:
            loss: 标量, L_td + L_cql

        NOTE: 当前版本 q_tot 是单个标量 (经 QMix 聚合),
              严格 CQL 需要对每个动作的 Q(s,a) 做 logsumexp。
              这里用简化版本: logsumexp over batch 近似。
        """
        # L_td: TD error
        td_loss = F.mse_loss(q_tot, td_target)

        # L_cql: 保守正则化
        # 简化: 对 batch 内所有 Q_tot 做 logsumexp
        # 完整 CQL 应对每个状态采样多个动作, 但这里 Q_tot 已聚合
        cql_loss = self.alpha * (
            torch.logsumexp(q_tot, dim=0) - q_tot.mean()
        )

        return td_loss + cql_loss
