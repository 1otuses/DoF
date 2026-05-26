"""
VAE 状态分解模块 (Phase 1)

核心思想：将 agent_i 的局部观测 o_i 分解为两个隐变量：
    - z_env:   环境慢变特征 (slow-changing)，跨时间步保持一致性
    - z_inter: 交互快变特征 (fast-changing)，编码当前时刻的局部交互

Encoder: o_i → [Shared MLP] → z_env | z_inter
Decoder: (z_env, z_inter) → o_i_hat (重构原始观测)

维度约束：
    z_inter_dim > z_env_dim
    z_inter_dim + z_env_dim > obs_dim (略高于观测维度以保证信息容量)

Phase 1 损失函数：
    L1 = L_rec + α · L_NCE + β · L_decouple

    L_rec:      重构损失: MSE(o_i, o_i_hat)
    L_NCE:      InfoNCE 时序对比损失 (慢变环境特征对齐)
    L_decouple: 正交解耦损失 (z_env ⊥ z_inter)

参考公式 (论文式记法):
    L_NCE = -E_D[ log( exp(sim(z_env_t, z_env_{t+1}) / τ) /
        (exp(sim(z_env_t, z_env_{t+1}) / τ) + Σ_j exp(sim(z_env_t, z_env_{k_j}) / τ)) ) ]

    L_decouple = E_D[ ( (z_envᵀ · z_inter) / (||z_env|| · ||z_inter||) )² ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ==============================================================================
# VAE Encoder
# ==============================================================================

class VAEEncoder(nn.Module):
    """
    VAE 编码器：将观测 o_i 分解为 z_env + z_inter

    o_i → [Shared MLP] → features ──┬── env_head  → z_env
                                    └── inter_head → z_inter

    维度关系:z_inter_dim > z_env_dim, sum > obs_dim

    Args:
        obs_dim:     观测维度 o_i
        env_dim:     z_env 维度 (默认为 obs_dim // 3)
        inter_dim:   z_inter 维度 (默认为 obs_dim)
        hidden_dim:  共享 MLP 隐藏层维度
        num_layers:  MLP 层数
        activation:  激活函数
        dropout:     Dropout 比率
    """

    def __init__(
        self,
        obs_dim: int,
        env_dim: int = None,
        inter_dim: int = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = 'mish',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        # 维度约束：z_inter > z_env, sum > obs_dim
        if env_dim is None:
            env_dim = max(4, obs_dim // 3)
        if inter_dim is None:
            inter_dim = max(obs_dim, obs_dim - env_dim + 2)

        # 保证维度约束成立
        if inter_dim <= env_dim:
            inter_dim = env_dim + 2
        if env_dim + inter_dim <= obs_dim:
            inter_dim = obs_dim - env_dim + 2

        self.env_dim = env_dim
        self.inter_dim = inter_dim
        self.latent_dim = env_dim + inter_dim

        act_fn = {
            'mish': nn.Mish,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
        }[activation]

        # --- Shared Feature Extractor ---
        layers = []
        in_dim = obs_dim
        for i in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)

        # --- Disentanglement Heads ---
        self.env_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, env_dim),
        )
        self.inter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, inter_dim),
        )

    def forward(self, obs_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_i: [..., obs_dim]
        Returns:
            z_env:   [..., env_dim]
            z_inter: [..., inter_dim]
        """
        features = self.shared_net(obs_i)  # [..., hidden_dim]
        z_env = self.env_head(features)     # [..., env_dim]
        z_inter = self.inter_head(features) # [..., inter_dim]
        return z_env, z_inter

    def encode_all_agents(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量为所有 agent 编码。

        Args:
            obs: [B, N, obs_dim] 或 [B, T, N, obs_dim]

        Returns:
            z_env:   [B, N, env_dim] 或 [B, T, N, env_dim]
            z_inter: [B, N, inter_dim] 或 [B, T, N, inter_dim]
        """
        orig_shape = obs.shape
        obs_flat = obs.reshape(-1, self.obs_dim)  # [B*N, obs_dim] or [B*T*N, obs_dim]
        z_env_flat, z_inter_flat = self.forward(obs_flat)
        z_env = z_env_flat.reshape(*orig_shape[:-1], self.env_dim)
        z_inter = z_inter_flat.reshape(*orig_shape[:-1], self.inter_dim)
        return z_env, z_inter


# ==============================================================================
# VAE Decoder
# ==============================================================================

class VAEDecoder(nn.Module):
    """
    VAE 解码器：将 (z_env, z_inter) 重构回原始观测 o_i

    (z_env, z_inter) → concat → [Decoder MLP] → o_i_hat

    Args:
        env_dim:    z_env 维度
        inter_dim:  z_inter 维度
        obs_dim:    原始观测维度
        hidden_dim: 隐藏层维度
        num_layers: MLP 层数
        activation: 激活函数
    """

    def __init__(
        self,
        env_dim: int,
        inter_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = 'mish',
    ):
        super().__init__()

        self.env_dim = env_dim
        self.inter_dim = inter_dim
        self.obs_dim = obs_dim
        self.input_dim = env_dim + inter_dim

        act_fn = {
            'mish': nn.Mish,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
        }[activation]

        layers = []
        in_dim = self.input_dim
        for i in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, z_env: torch.Tensor, z_inter: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_env:   [..., env_dim]
            z_inter: [..., inter_dim]
        Returns:
            o_hat:   [..., obs_dim]
        """
        z = torch.cat([z_env, z_inter], dim=-1)  # [..., env_dim+inter_dim]
        return self.net(z)

    def decode_all_agents(
        self,
        z_env_all: torch.Tensor,
        z_inter_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量为所有 agent 解码。

        Args:
            z_env_all:   [B, N, env_dim] 或 [B, T, N, env_dim]
            z_inter_all: [B, N, inter_dim] 或 [B, T, N, inter_dim]

        Returns:
            o_hat: [B, N, obs_dim] 或 [B, T, N, obs_dim]
        """
        orig_shape_env = z_env_all.shape
        z_env_flat = z_env_all.reshape(-1, self.env_dim)
        z_inter_flat = z_inter_all.reshape(-1, self.inter_dim)
        o_hat_flat = self.forward(z_env_flat, z_inter_flat)
        return o_hat_flat.reshape(*orig_shape_env[:-1], self.obs_dim)


# ==============================================================================
# Phase 1 损失函数
# ==============================================================================

def compute_reconstruction_loss(
    decoder: VAEDecoder,
    z_env: torch.Tensor,
    z_inter: torch.Tensor,
    obs_true: torch.Tensor,
) -> torch.Tensor:
    """
    L_rec = MSE(o_i, o_i_hat)

    Args:
        decoder:   VAEDecoder
        z_env:     [..., env_dim]
        z_inter:   [..., inter_dim]
        obs_true:  [..., obs_dim]

    Returns:
        scalar loss
    """
    o_hat = decoder(z_env, z_inter)  # [..., obs_dim]
    return F.mse_loss(o_hat, obs_true)


def compute_temporal_infonce_loss(
    z_env_t: torch.Tensor,
    z_env_next: torch.Tensor,
    temperature: float = 0.1,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    时序 InfoNCE 对比损失 L_NCE。

    正样本: (z_env_t, z_env_{t+1}) — 同一 agent 的连续时间步
    负样本: batch 中其他时间步 / 其他 agent 的 z_env

    L_NCE = -E[ log( exp(sim(z_env_t, z_env_{t+1}) / τ) /
        (exp(sim(z_env_t, z_env_{t+1}) / τ) + Σ_j exp(sim(z_env_t, z_env_{kj}) / τ)) ) ]

    Args:
        z_env_t:     [B, N, env_dim]
        z_env_next:  [B, N, env_dim]
        temperature: τ 温度系数
        symmetric:   是否双向 NCE

    Returns:
        scalar loss
    """
    B, N, D = z_env_t.shape
    # Flatten to [B*N, D]
    z_env_flat = F.normalize(z_env_t.reshape(B * N, D), dim=-1)
    z_next_flat = F.normalize(z_env_next.reshape(B * N, D), dim=-1)

    num_pairs = z_env_flat.shape[0]
    if num_pairs < 2:
        return z_env_flat.sum() * 0.0  # zero grad

    # 相似度矩阵 [B*N, B*N]
    logits = torch.matmul(z_env_flat, z_next_flat.t()) / temperature
    labels = torch.arange(num_pairs, device=z_env_flat.device)

    loss_forward = F.cross_entropy(logits, labels)
    if not symmetric:
        return loss_forward

    loss_backward = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_forward + loss_backward)


def compute_decouple_loss(
    z_env: torch.Tensor,
    z_inter: torch.Tensor,
) -> torch.Tensor:
    """
    正交解耦损失 L_decouple。

    当 env_dim ≠ inter_dim 时，通过投影统一到 z_env 的维度空间，
    再计算余弦相似度的平方。

    L_decouple = E[ cosine_sim(z_env, project(z_inter))² ]

    通过惩罚余弦相似度的平方，推动 z_env ⊥ z_inter。

    Args:
        z_env:   [..., env_dim]
        z_inter: [..., inter_dim] （可能与 env_dim 不等）

    Returns:
        scalar loss ∈ [0, 1]
    """
    D_env = z_env.shape[-1]
    D_inter = z_inter.shape[-1]

    if D_env != D_inter:
        # 投影 z_inter → z_env 空间（使用固定正交随机投影）
        # 便捷方案：截断或线性投影
        if D_env < D_inter:
            # z_inter 取前 D_env 维（此时信息损失较小，
            # 因为正交性约束不需要 z_inter 的全部维度）
            z_inter_proj = z_inter[..., :D_env]
        else:
            z_inter_proj = z_inter  # env_dim > inter_dim 的情况（不应发生）
    else:
        z_inter_proj = z_inter

    z_env_norm = F.normalize(z_env, dim=-1)          # [..., env_dim]
    z_inter_norm = F.normalize(z_inter_proj, dim=-1)  # [..., env_dim]
    cosine_sim = (z_env_norm * z_inter_norm).sum(dim=-1)  # [...]
    return (cosine_sim ** 2).mean()


def compute_phase1_total_loss(
    decoder: VAEDecoder,
    z_env_t: torch.Tensor,
    z_inter_t: torch.Tensor,
    z_env_next: torch.Tensor,
    obs_t: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    contrastive_temperature: float = 0.1,
    verbose: bool = False,
) -> dict:
    """
    Phase 1 总损失。

    L1 = L_rec + α · L_NCE + β · L_decouple

    Args:
        decoder:               VAEDecoder
        z_env_t:               [B, N, env_dim] — 当前时刻的环境表征
        z_inter_t:             [B, N, inter_dim] — 当前时刻的交互表征
        z_env_next:            [B, N, env_dim] — 下一时刻的环境表征
        obs_t:                 [B, N, obs_dim] — 当前时刻的真实观测
        alpha:                 L_NCE 权重
        beta:                  L_decouple 权重
        contrastive_temperature: InfoNCE 温度
        verbose:               是否打印详细信息

    Returns:
        dict: {
            'loss_total':      total,
            'loss_rec':        L_rec,
            'loss_nce':        L_NCE,
            'loss_decouple':   L_decouple,
        }
    """
    loss_rec = compute_reconstruction_loss(decoder, z_env_t, z_inter_t, obs_t)
    loss_nce = compute_temporal_infonce_loss(z_env_t, z_env_next, temperature=contrastive_temperature)
    loss_decouple = compute_decouple_loss(z_env_t, z_inter_t)

    loss_total = loss_rec + alpha * loss_nce + beta * loss_decouple

    if verbose:
        print(
            f"  L_rec={loss_rec.item():.6f} | "
            f"L_NCE={loss_nce.item():.6f} | "
            f"L_decouple={loss_decouple.item():.6f} | "
            f"L_total={loss_total.item():.6f}"
        )

    return {
        'loss_total': loss_total,
        'loss_rec': loss_rec,
        'loss_nce': loss_nce,
        'loss_decouple': loss_decouple,
    }


# ==============================================================================
# VAE 完整模型 (Encoder + Decoder)
# ==============================================================================

class ObservationVAE(nn.Module):
    """
    完整的观测 VAE 模型。

    包含：
        encoder: VAEEncoder — o_i → (z_env, z_inter)
        decoder: VAEDecoder — (z_env, z_inter) → o_i_hat

    用法：
        vae = ObservationVAE(obs_dim=obs_dim, env_dim=env_dim, inter_dim=inter_dim)
        z_env, z_inter = vae.encode(obs)
        o_hat = vae.decode(z_env, z_inter)
    """

    def __init__(
        self,
        obs_dim: int,
        env_dim: int = None,
        inter_dim: int = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = 'mish',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = VAEEncoder(
            obs_dim=obs_dim,
            env_dim=env_dim,
            inter_dim=inter_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
        self.decoder = VAEDecoder(
            env_dim=self.encoder.env_dim,
            inter_dim=self.encoder.inter_dim,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )

        self.obs_dim = obs_dim
        self.env_dim = self.encoder.env_dim
        self.inter_dim = self.encoder.inter_dim

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码 o 到 (z_env, z_inter)"""
        return self.encoder(obs)

    def encode_all_agents(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量编码所有 agent"""
        return self.encoder.encode_all_agents(obs)

    def decode(
        self, z_env: torch.Tensor, z_inter: torch.Tensor
    ) -> torch.Tensor:
        """解码回 o_hat"""
        return self.decoder(z_env, z_inter)

    def decode_all_agents(
        self,
        z_env_all: torch.Tensor,
        z_inter_all: torch.Tensor,
    ) -> torch.Tensor:
        """批量解码所有 agent"""
        return self.decoder.decode_all_agents(z_env_all, z_inter_all)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播。

        Args:
            obs: [..., obs_dim]

        Returns:
            (z_env, z_inter, o_hat)
        """
        z_env, z_inter = self.encode(obs)
        o_hat = self.decode(z_env, z_inter)
        return z_env, z_inter, o_hat

    @property
    def summary(self) -> str:
        return (
            f"ObservationVAE: {self.obs_dim} → z_env({self.env_dim}) + "
            f"z_inter({self.inter_dim}) → {self.obs_dim}  "
            f"(sum={self.env_dim + self.inter_dim} > {self.obs_dim}? "
            f"{self.env_dim + self.inter_dim > self.obs_dim})"
        )


# ==============================================================================
# Freeze helper (Phase 2 用)
# ==============================================================================

def freeze_vae(vae: ObservationVAE):
    """冻结 VAE 的编码器 + 解码器参数 (Phase 2 调用)"""
    for param in vae.parameters():
        param.requires_grad = False
