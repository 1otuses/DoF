"""
状态分解编码器 (State Decomposition Encoder)

核心思想：将 agent_i 的观测状态 o_i 分解为两个独立的隐向量：
    - z_env:  环境信息, 在有 global state 时通过 Predictor(z_env) 重构 S
    - z_inter: 交互信息, 与 z_env 通过互信息最小化解耦

对于没有 global state 的数据集, 额外可用连续时间步的 self-supervised
InfoNCE 约束对齐 z_env(t) 与 z_env(t+1), 让环境表征保持时序一致性。

设计目标：
    1. z_env 保留可重构全局状态 S 的信息（或在无 states 时保持时间对齐）
    2. z_env 与 z_inter 互信息最小，实现真正解耦
    3. z_env + z_inter 共同预测奖励，保证隐空间语义有效性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateDecompositionEncoder(nn.Module):
    """
    状态分解编码器

    o_i -> [Shared MLP] -> features ──┬── env_head  -> z_env   (obs_dim)
                                       └── inter_head -> z_inter (8~32)
    
    Args:
        obs_dim:     观测维度
        env_dim:     z_env 维度，None = obs_dim
        inter_dim:   z_inter 维度（低维交互信号）
        hidden_dim:  隐藏层维度
        num_layers:  MLP 层数
        activation:  激活函数类型
        dropout:     Dropout 比率
    """

    def __init__(
        self,
        obs_dim: int,
        env_dim: int = None,
        inter_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = 'mish',
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.env_dim = obs_dim if env_dim is None else env_dim
        self.inter_dim = inter_dim
        self.latent_dim = self.env_dim + inter_dim

        act_fn = {
            'mish': nn.Mish,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
        }[activation]

        # Shared Feature Extractor
        layers = []
        in_dim = obs_dim
        for i in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)

        # Disentanglement Heads
        self.env_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act_fn(),
            nn.Linear(hidden_dim, self.env_dim),
        )
        self.inter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act_fn(),
            nn.Linear(hidden_dim, self.inter_dim),
        )

    def forward(self, obs_i: torch.Tensor) -> tuple:
        features = self.shared_net(obs_i)
        z_env = self.env_head(features)
        z_inter = self.inter_head(features)
        return z_env, z_inter

    def encode_all_agents(self, obs: torch.Tensor) -> tuple:
        batch_size, n_agents, _ = obs.shape
        obs_flat = obs.reshape(-1, self.obs_dim)  # [B*N, obs_dim]
        z_env_flat, z_inter_flat = self.forward(obs_flat)
        z_env_all = z_env_flat.reshape(batch_size, n_agents, self.env_dim)
        z_inter_all = z_inter_flat.reshape(batch_size, n_agents, self.inter_dim)
        return z_env_all, z_inter_all


# ==============================================================================
# Global State Predictor
# ==============================================================================

class GlobalStatePredictor(nn.Module):
    """
    全局状态预测器：从每个 agent 的 z_env 预测全局状态 S。

    每个 agent 的 z_env 都能独立预测出 S，这隐式要求 z_env
    必须包含足够的全局环境信息，而不能只编码局部视角。

    S_pred_i = Predictor(z_env_i)   for i = 1..N

    Args:
        env_dim:    z_env 维度
        state_dim:  全局状态 S 的维度
        hidden_dim: 隐藏层维度
        num_layers: MLP 层数
    """

    def __init__(
        self,
        env_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = env_dim
        for i in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_env: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_env: [..., env_dim]
        Returns:
            S_pred: [..., state_dim]
        """
        return self.net(z_env)


# ==============================================================================
# 互信息最小化（vCLUB 上界）
# ==============================================================================

class MILowerBoundEstimator(nn.Module):
    """
    互信息下界估计器（InfoNCE / vCLUB 风格）。

    通过变分分布 q(z_inter | z_env) 来估计 MI 下界:
      I(z_env; z_inter) >= E[log q(z_inter|z_env)] - E[log q(z_inter'|z_env)]
    其中 z_inter' 是 batch 内 shuffle 后的负样本。

    最小化 I 等价于最小化这个下界。
    """

    def __init__(self, env_dim: int, inter_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, inter_dim),
        )

    def forward(self, z_env: torch.Tensor, z_inter: torch.Tensor) -> torch.Tensor:
        """
        vCLUB 互信息上界估计（最小化这个 loss = 最小化 MI）。

        Args:
            z_env:   [B, ..., env_dim]
            z_inter: [B, ..., inter_dim]

        Returns:
            mi_loss: scalar （正数，越小表示 MI 越小）
        """
        orig_shape = z_env.shape[:-1]
        z_env_flat = z_env.reshape(-1, z_env.shape[-1])
        z_inter_flat = z_inter.reshape(-1, z_inter.shape[-1])

        batch_size = z_env_flat.shape[0]

        # 预测正样本的 log-prob
        mu = self.net(z_env_flat)  # [B, inter_dim]
        # log q(z_inter|z_env) ≈ -||mu - z_inter||^2 (高斯假设)
        positive_log_prob = -((mu - z_inter_flat) ** 2).sum(dim=-1)  # [B]

        # 负样本：shuffle z_inter
        shuffle_idx = torch.randperm(batch_size, device=z_env_flat.device)
        z_inter_neg = z_inter_flat[shuffle_idx]
        negative_log_prob = -((mu - z_inter_neg) ** 2).sum(dim=-1)  # [B]

        # CLUB 上界: mean(positive) - mean(negative)
        mi_loss = (positive_log_prob - negative_log_prob).mean()

        return torch.clamp(mi_loss, min=0.0)  # ≥ 0, 越小越好


# ==============================================================================
# 解耦辅助损失函数（新版）
# ==============================================================================

def compute_env_reconstruction_loss(
    predictor: GlobalStatePredictor,
    z_env_all: torch.Tensor,
    global_state: torch.Tensor,
) -> torch.Tensor:
    """
    环境重构损失：每个 agent 的 z_env 预测全局状态 S。

    L_env = sum_i ||Predictor(z_env_i) - S||^2

    如果数据集不提供全局状态(global_state is None),则返回 0 损失。

    Args:
        predictor:   GlobalStatePredictor
        z_env_all:   [B, N, env_dim]
        global_state:[B, state_dim] or None

    Returns:
        scalar loss (0 if global_state is None)
    """
    if global_state is None:
        return torch.tensor(0.0, device=z_env_all.device, requires_grad=True)

    B, N, D = z_env_all.shape
    z_env_flat = z_env_all.reshape(-1, D)  # [B*N, env_dim]
    S_pred = predictor(z_env_flat)  # [B*N, state_dim]

    # 将全局状态 S 扩展到 N 份以做 pairwise 比较
    S_expanded = global_state.unsqueeze(1).expand(-1, N, -1).reshape(-1, global_state.shape[-1])

    loss = F.mse_loss(S_pred, S_expanded)
    return loss


def compute_mi_loss(
    mi_estimator: MILowerBoundEstimator,
    z_env_all: torch.Tensor,
    z_inter_all: torch.Tensor,
) -> torch.Tensor:
    """
    互信息最小化损失。

    使用 vCLUB 上界估计 I(z_env; z_inter)，
    最小化此 loss 来推动 z_env 与 z_inter 解耦。

    Args:
        mi_estimator: MILowerBoundEstimator
        z_env_all:    [B, N, env_dim]
        z_inter_all:  [B, N, inter_dim]

    Returns:
        mi_loss: scalar
    """
    B, N, D_env = z_env_all.shape
    _, _, D_inter = z_inter_all.shape

    # 展平 agent 维度，将每对 (env_i, inter_i) 视为独立样本
    z_env_flat = z_env_all.reshape(B * N, D_env)
    z_inter_flat = z_inter_all.reshape(B * N, D_inter)

    mi_loss = mi_estimator(z_env_flat, z_inter_flat)
    return mi_loss


def compute_temporal_contrastive_loss(
    z_env_t: torch.Tensor,
    z_env_next: torch.Tensor,
    temperature: float = 0.1,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    连续时间步对齐的 InfoNCE 损失。

    正样本对: (z_env_t, z_env_{t+1})
    负样本对: 同一 batch 中其他轨迹 / 其他 agent 的 z_env_{t'}

    Args:
        z_env_t:     [B, N, env_dim]
        z_env_next:  [B, N, env_dim]
        temperature: InfoNCE 温度系数
        symmetric:   是否同时计算 t->t+1 和 t+1->t

    Returns:
        scalar InfoNCE loss
    """
    if z_env_t.shape != z_env_next.shape:
        raise ValueError(
            f"z_env_t shape {tuple(z_env_t.shape)} must match z_env_next shape {tuple(z_env_next.shape)}"
        )

    B, N, D = z_env_t.shape
    z_env_flat = F.normalize(z_env_t.reshape(B * N, D), dim=-1)  # [B*N, env_dim]
    z_next_flat = F.normalize(z_env_next.reshape(B * N, D), dim=-1)  # [B*N, env_dim]

    num_pairs = z_env_flat.shape[0]
    if num_pairs < 2:
        return z_env_flat.sum() * 0.0

    logits = torch.matmul(z_env_flat, z_next_flat.t()) / temperature  # [B*N, B*N]
    labels = torch.arange(num_pairs, device=z_env_flat.device)

    loss_forward = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss_forward

    loss_backward = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_forward + loss_backward)


def compute_decomposition_losses(
    predictor: GlobalStatePredictor,
    mi_estimator: MILowerBoundEstimator,
    z_env_all: torch.Tensor,
    z_inter_all: torch.Tensor,
    global_state: torch.Tensor,
    env_reconstruction_weight: float = 1.0,
    mi_weight: float = 0.1,
):
    """
    汇总所有解耦辅助损失（新版）

    损失组成:
      1. L_env_recon:  Predictor(z_env) -> S   （环境信息保留）
      2. L_mi:         I(z_env; z_inter) 最小化  （解耦）

    Args:
        predictor:   GlobalStatePredictor
        mi_estimator:MILowerBoundEstimator
        z_env_all:   [B, N, env_dim]
        z_inter_all: [B, N, inter_dim]
        global_state:[B, state_dim]

    Returns:
        dict of losses and total
    """
    loss_env_recon = compute_env_reconstruction_loss(
        predictor, z_env_all, global_state
    )
    loss_mi = compute_mi_loss(mi_estimator, z_env_all, z_inter_all)

    total_aux_loss = (
        env_reconstruction_weight * loss_env_recon
        + mi_weight * loss_mi
    )

    return {
        'loss_env_recon': loss_env_recon,
        'loss_mi': loss_mi,
        'total_aux_loss': total_aux_loss,
    }
