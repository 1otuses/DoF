"""
VAE-Diffusion 集成模块 (Phase 2)

Phase 2 工作流：
    1. 加载已冻结的 VAE (encoder 参数固定)
    2. 对离线数据集中所有观测 o_i 预编码: o_i → (z_env_i, z_inter_i)
    3. z_env_i 作为扩散模型的条件之一(类似 returns_condition)
    4. z_inter_i 替换原始观测 o_i 进入扩散模型的去噪循环
    5. 扩散模型对 z_inter 进行去噪: x_t(z_inter) → x_0(z_inter)
    6. 解码: 冻结的 Decoder(z_env, z_inter_denoised) → o_i_hat

核心修改：
    - 数据管道的转换:obs → z_env + z_inter
    - 扩散模型输入维度从 obs_dim 变为 inter_dim
    - z_env 作为条件注入扩散模型
    - 去噪后的 z_inter + z_env 通过冻结 Decoder 还原 o_i

依赖：
    - ObservationVAE (vae.py): 已训练的 VAE 编码器/解码器
    - GaussianDiffusion (diffusion.py): DoF 扩散模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import einops

from .vae import ObservationVAE, freeze_vae


# ==============================================================================
# VAE 预编码数据集包装器
# ==============================================================================

class VAEPreEncodedDataWrapper:
    """
    将离线数据中的 obs 预编码为 z_env + z_inter,
    并调整为 diffusion 模型所需的格式。

    原始数据格式：
        dataset[i] → { 'x': [horizon, n_agents, obs_dim + action_dim],
                        'cond': { 'x': [history_horizon, n_agents, obs_dim] },
                        'returns': [...], ... }

    转换后格式：
        保留 action 不变
        obs 部分 → z_inter (扩散目标) + z_env (条件)

    注意: VAE 预编码后，编码是一次性的（预处理），不需要梯度。
    """

    def __init__(
        self,
        vae: ObservationVAE,
        original_dataset,
        device: str = 'cuda',
        batch_size: int = 64,
    ):
        """
        Args:
            vae:  已冻结的 ObservationVAE
            original_dataset: 原始 SequenceDataset 实例
            device: 执行预编码的设备
            batch_size: 预编码的批大小
        """
        self.vae = vae
        self.original_dataset = original_dataset
        self.device = device
        self.batch_size = batch_size

        # 记录维度的改变
        self.original_obs_dim = vae.obs_dim  # 原始观测维度
        self.inter_dim = vae.inter_dim        # z_inter 维度 (扩散目标)
        self.env_dim = vae.env_dim            # z_env 维度 (条件)

        # VAE 冻结
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        # 不预编码整个数据集（太大），而是在 __getitem__ 时动态编码
        self._pre_encoding_cache = {}  # 可选：缓存已编码的结果

    def encode_obs_slice(self, obs_slice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 obs 切片进行 VAE 编码。

        Args:
            obs_slice: [..., n_agents, obs_dim]

        Returns:
            z_env:   [..., n_agents, env_dim]
            z_inter: [..., n_agents, inter_dim]
        """
        with torch.no_grad():
            z_env, z_inter = self.vae.encode_all_agents(obs_slice)
        return z_env, z_inter

    def decode_z_slice(self, z_env: torch.Tensor, z_inter: torch.Tensor) -> torch.Tensor:
        """
        对 z 切片进行 VAE 解码。

        Args:
            z_env:   [..., n_agents, env_dim]
            z_inter: [..., n_agents, inter_dim]

        Returns:
            obs_hat: [..., n_agents, obs_dim]
        """
        with torch.no_grad():
            obs_hat = self.vae.decode_all_agents(z_env, z_inter)
        return obs_hat


# ==============================================================================
# VAE-Conditioned Diffusion 包装器
# ==============================================================================

class VAEDiffusionWrapper(nn.Module):
    """
    Phase 2 核心：将 VAE 的 z_env 条件注入扩散模型。

    架构：
        Diffusion(z_inter | z_env, returns) → 去噪后的 z_inter
        Decoder(z_env, z_inter_denoised) → obs_hat

    修改点：
        1. diffusion.transition_dim 从 (obs_dim + action_dim) 变为
           (inter_dim + action_dim) — 因为去噪对象从 obs 变成 z_inter
        2. 在 conditional_sample 中：
           a. 噪声从 z_inter 空间采样
           b. 条件中除了 obs conditioning,还有 z_env conditioning
           c. 去噪完成后，用冻结 Decoder 将 z_inter → obs
        3. p_losses 中：
           x_start 的 obs 部分换为 z_inter

    Args:
        vae:             已训练的 ObservationVAE (冻结)
        diffusion_model: 原始的 GaussianDiffusion 实例
    """

    def __init__(
        self,
        vae: ObservationVAE,
        diffusion_model: nn.Module,
    ):
        super().__init__()

        # 冻结 VAE
        freeze_vae(vae)
        self.vae = vae

        # 包装扩散模型
        self.diffusion = diffusion_model

        # 更新维度信息
        self.original_obs_dim = vae.obs_dim
        self.inter_dim = vae.inter_dim
        self.env_dim = vae.env_dim
        self.n_agents = diffusion_model.n_agents
        self.horizon = diffusion_model.horizon
        self.history_horizon = diffusion_model.history_horizon
        self.action_dim = diffusion_model.action_dim

        # 注意: 扩散模型内部的 self.model (如 TemporalUnet)
        # 其 transition_dim 在构造时已经固定为 (obs_dim + action_dim)
        # 此处我们不修改扩散模型内部结构，而是：
        #   - 在输入扩散前，将 z_inter 映射回 obs_dim 空间（或不做映射）
        # 实际上需要修改 diffusion_model.transition_dim 或在外部做维度转换
        #
        # 最简单的做法：在扩散模型外部将 inter_dim → obs_dim 做线性投影，
        # 然后用原扩散模型不变。但这样会增加参数。
        #
        # 更优雅的做法：将 diffusion.model (TemporalUnet) 的 transition_dim
        # 替换为 (inter_dim + action_dim)，重建整个扩散架构。
        #
        # 折中方案（推荐）：保留原扩散模型不变，但使用 adapter 层：
        #   z_inter → [Linear] → z_inter_padded → concat(action) → 输入扩散
        #   输出扩散 ← 拆分 → inter 部分 → [Linear] → z_inter_decoded
        #   这样可以复用原始 DoF 的预训练权重。

        self.inter_to_obs = nn.Linear(self.inter_dim, self.original_obs_dim)
        self.obs_to_inter = nn.Linear(self.original_obs_dim, self.inter_dim)

        print(f"\n[VAEDiffusionWrapper]")
        print(f"  obs_dim={self.original_obs_dim}, inter_dim={self.inter_dim}, env_dim={self.env_dim}")
        print(f"  VAE frozen: {not any(p.requires_grad for p in vae.parameters())}")
        print(f"  Adapter: inter→obs (Linear) + obs→inter (Linear)")

    def obs_to_z(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将原始观测编码为 (z_env, z_inter)。

        Args:
            obs: [B, T, N, obs_dim] or [B, N, obs_dim]

        Returns:
            z_env:   same batch shape but [...env_dim]
            z_inter: same batch shape but [...inter_dim]
        """
        return self.vae.encode_all_agents(obs)

    def z_to_obs(self, z_env: torch.Tensor, z_inter: torch.Tensor) -> torch.Tensor:
        """
        将隐变量解码回观测。

        Args:
            z_env:   [..., env_dim]
            z_inter: [..., inter_dim]

        Returns:
            obs_hat: [..., obs_dim]
        """
        return self.vae.decode_all_agents(z_env, z_inter)

    def prepare_diffusion_input(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 (obs, action) 转换到扩散模型输入空间。

        流程：
            obs → VAE => z_inter
            z_inter → inter_to_obs → obs'_projected
            concat(obs'_projected, action) → x_input

        Args:
            obs:     [B, T, N, obs_dim]
            actions: [B, T, N, action_dim]

        Returns:
            x: [B, T, N, obs_dim + action_dim] (兼容原扩散)
        """
        _, z_inter = self.obs_to_z(obs)  # [B, T, N, inter_dim]
        z_projected = self.inter_to_obs(z_inter)  # [B, T, N, obs_dim]
        x = torch.cat([z_projected, actions], dim=-1)  # [B, T, N, obs_dim+act_dim]
        return x

    def recover_obs_from_z(
        self,
        z_inter: torch.Tensor,
        z_env: torch.Tensor,
    ) -> torch.Tensor:
        """
        从去噪后的 z_inter + z_env 恢复观测。

        Args:
            z_inter: [B, T, N, inter_dim] — 去噪后的
            z_env:   [B, T, N, env_dim]   — 条件

        Returns:
            obs_hat: [B, T, N, obs_dim]
        """
        return self.vae.decode_all_agents(z_env, z_inter)

    @torch.no_grad()
    def conditional_sample(
        self,
        cond: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        horizon: int = None,
        attention_masks: Optional[torch.Tensor] = None,
        verbose: bool = True,
        return_diffusion: bool = False,
        **kwargs,
    ):
        """
        条件采样：在 z_env 条件下对 z_inter 去噪。

        与原始 conditional_sample 的区别：
            添加了一个后处理步骤：去噪后的 z_inter → Decoder → obs_hat
        """
        # 调用原始扩散模型的 conditional_sample
        # 输入的是 z_inter 对应的观测空间（经过 inter_to_obs）
        # 输出也是观测空间 → 需要转换回 z_inter → 解码
        result = self.diffusion.conditional_sample(
            cond=cond,
            returns=returns,
            env_ts=env_ts,
            horizon=horizon,
            attention_masks=attention_masks,
            verbose=verbose,
            return_diffusion=return_diffusion,
        )

        if return_diffusion:
            x, diffusion = result
        else:
            x = result
            diffusion = None

        # x: [B, T, N, obs_dim] — 观测空间输出
        # 分离 obs 部分并转换到 z_inter 空间
        x_obs = x[..., :self.original_obs_dim]  # [B, T, N, obs_dim]
        z_inter = self.obs_to_inter(x_obs)       # [B, T, N, inter_dim]

        # 需要 z_env 来解码
        # z_env 可以从 cond 中的观测计算
        if 'x' in cond:
            cond_obs = cond['x']  # [B, history_horizon, N, obs_dim]
            z_env_cond, _ = self.obs_to_z(cond_obs)  # [B, H, N, env_dim]
            # 取最后一个历史步的 z_env 作为整个规划 horizon 的 z_env
            z_env = z_env_cond[:, -1, :, :].unsqueeze(1).expand(
                -1, x.shape[1], -1, -1
            )  # [B, T, N, env_dim]
        else:
            # 若无条件观测，z_env 设为零（不太理想但可工作）
            z_env = torch.zeros(
                x.shape[0], x.shape[1], self.n_agents, self.env_dim
            ).to(x.device)

        # 解码回观测
        obs_hat = self.vae.decode_all_agents(z_env, z_inter)  # [B, T, N, obs_dim]
        x_out = torch.cat([obs_hat, x[..., self.original_obs_dim:]], dim=-1)

        if return_diffusion:
            return x_out, diffusion
        return x_out

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_timestep: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        前向传播：直接委托给扩散模型的 model。
        """
        return self.diffusion.model(
            x, t,
            returns=returns,
            env_timestep=env_timestep,
            attention_masks=attention_masks,
            states=states,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
            **kwargs,
        )


# ==============================================================================
# 数据转换辅助函数
# ==============================================================================

def convert_dataset_batch_for_vae(
    batch: Dict[str, torch.Tensor],
    vae: ObservationVAE,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """
    将原始数据 batch 中的 obs 替换为 z_env + z_inter。

    Args:
        batch: 原始数据 batch,含 'x' 和 'cond' 字段
        vae:   冻结的 VAE 模型
        device: 计算设备

    Returns:
        batch_vae: 转换后的 batch
            - 'z_inter': z_inter 值 (替代 obs)
            - 'z_env':   z_env 值 (额外条件)
            - 'x':       (z_inter_projected, action) concat
    """
    vae.eval()
    with torch.no_grad():
        # 转换 cond.x (历史观测)
        if 'cond' in batch and 'x' in batch['cond']:
            cond_x = batch['cond']['x'].to(device)  # [B, H, N, obs_dim]
            z_env_cond, z_inter_cond = vae.encode_all_agents(cond_x)
            z_inter_cond_proj = nn.Linear(
                vae.inter_dim, vae.obs_dim, device=device
            )(z_inter_cond)  # 若需要投影
            batch['cond']['z_env'] = z_env_cond
            # cond['x'] 保持原样供 apply_conditioning 使用

        # 转换 x (obs + action)
        if 'x' in batch:
            x = batch['x'].to(device)  # [B, T, N, obs_dim + action_dim]
            obs_part = x[..., :vae.obs_dim]  # [B, T, N, obs_dim]
            act_part = x[..., vae.obs_dim:]   # [B, T, N, action_dim]

            z_env, z_inter = vae.encode_all_agents(obs_part)

            batch['z_inter'] = z_inter
            batch['z_env'] = z_env

            # 替换 obs 部分为 z_inter 投影
            inter_to_obs = nn.Linear(vae.inter_dim, vae.obs_dim, device=device)
            z_inter_proj = inter_to_obs(z_inter)  # [B, T, N, obs_dim]
            batch['x'] = torch.cat([z_inter_proj, act_part], dim=-1)
            batch['original_obs'] = obs_part  # 保留原始 obs 备用

    return batch
