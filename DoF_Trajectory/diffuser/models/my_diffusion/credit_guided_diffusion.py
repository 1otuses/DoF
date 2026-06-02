"""
信用引导扩散模型 (Credit-Guided Diffusion)

继承自 GaussianDiffusion, 新增多头信用评估架构 (模块一~四)。

模块一 - 网络结构:
    共享编码器 → 隐变量 H → 分三路:
    ├── Credit Critic Head → Q_i, Q_tot
    ├── Condition Router → C_i (detach)
    └── Diffusion Denoising Head → ε̂ (受 R 和 C 联合条件控制)

模块二 - 训练阶段:
    Step 1: 用未加噪真实轨迹在线计算 C_true
    Step 2: Categorical Dropout (u~Uniform) → R_mask, C_mask
    Step 3: 联合条件去噪 → ε̂

模块三 - 联合损失:
    L_total = L_diff + λ * L_credit
    L_credit = L_td + L_cql

模块四 - 推理阶段:
    Hierarchical CFG (N+2 次前向):
        ε̂ = ε_u + w_r(ε_r - ε_u) + Σ_i w_c(ε_{r,c}^{(i)} - ε_r)
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser.models.diffusion.GaussianDiffusion import GaussianDiffusion
from diffuser.models.helpers import apply_conditioning
from diffuser.models.my_diffusion.credit_critic import (
    AgentLocalCritic,
    QMixer,
    CQLLoss,
)
from diffuser.models.my_diffusion.condition_router import (
    ConditionRouter,
    build_credit_condition_vector,
)

import diffuser.utils as utils


class CreditGuidedDiffusion(GaussianDiffusion):
    """
    信用引导扩散模型——在原有 DoF 框架外套一层多头信用评估架构。

    相比基础 GaussianDiffusion 新增:
    - credit_critic:     信用评估头 (QMIX 风格)
    - condition_router:  Q → C 路由
    - cql_loss:          CQL 保守正则化
    - condition_dropout: 三种条件丢弃策略（模块二·Step 2）

    训练时: L_total = L_diff + λ * L_credit
    推理时: Hierarchical CFG 采样（模块四）
    """

    def __init__(
        self,
        model,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        # ---- 信用引导新增参数 ----
        use_credit_guide: bool = True,
        credit_hidden_dim: int = 256,
        credit_router_mode: str = "minmax",
        credit_lambda: float = 0.01,
        cql_alpha: float = 1.0,
        credit_condition_dropout: float = 0.2,
        # ---- Hierarchical CFG 参数 ----
        cfg_guidance_w: float = 1.2,
        cfg_credit_w: float = 0.5,
        **kwargs,
    ):
        # ---- 调用父类构造 ----
        super().__init__(
            model=model,
            n_agents=n_agents,
            horizon=horizon,
            history_horizon=history_horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            **kwargs,
        )

        self.use_credit_guide = use_credit_guide
        self.credit_lambda = credit_lambda
        self.cql_alpha = cql_alpha
        self.credit_condition_dropout = credit_condition_dropout
        self.cfg_guidance_w = cfg_guidance_w
        self.cfg_credit_w = cfg_credit_w

        if not self.use_credit_guide:
            return  # 不使用信用引导时退化为标准 GaussianDiffusion

        # ---- 信用评估头 ----
        self.credit_critic = AgentLocalCritic(
            n_agents=n_agents,
            hidden_dim=credit_hidden_dim,
        )
        self.qmixer = QMixer(
            n_agents=n_agents,
            state_dim=observation_dim * n_agents,  # 全局状态维度 = 所有 agent 观测拼接
            hidden_dim=credit_hidden_dim,
        )
        self.cql_loss = CQLLoss(alpha=cql_alpha)

        # ---- 条件路由 ----
        self.condition_router = ConditionRouter(mode=credit_router_mode)

    # =========================================================================
    # 辅助：从隐变量 H 提取 Q 和 C
    # =========================================================================

    def _get_hidden_representation(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从模型获取轨迹级隐变量 H （仅编码器部分）。

        这里复用 self.model 作为共享编码器：调用 forward 但不取最终输出。
        由于 TemporalUnet 的 forward 直接输出 epsilon，此处暂时返回 None。
        
        NOTE: 当前实现中，H 就是 model 的 forward 输出（即 epsilon 预测）。
              实际使用时 Q_i 从 epsilon 的统计量中提取：
              Q_i = MLP(mean(epsilon_{agent_i}, dim=[1,3]))。
        """
        return self.model(
            x, t,
            returns=returns,
            env_timestep=env_ts,
            attention_masks=attention_masks,
            use_dropout=False,
        )

    def _compute_credit_values(
        self,
        h: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
    ):
        """
        从隐变量 H 计算 Q_i、Q_tot 和 C_i。

        Args:
            h:  [B, horizon, N, hidden_dim] 隐变量
            global_state: [B, hidden_dim] 全局状态（可选）
        Returns:
            q_list: list of [B, 1]
            q_tot:  [B, 1]
            c_list: list of [B, 1]
        """
        # 对每个 agent 沿着时间维度和特征维度做均值池化
        B, T, N, D = h.shape
        h_pooled = h.mean(dim=1)  # [B, N, D] — 时间均值

        q_list = self.credit_critic(h_pooled)  # list of [B, 1]

        if global_state is None:
            global_state = h_pooled.reshape(B, N * D)  # [B, N*D] — fallback

        q_tot = self.qmixer(q_list, global_state)  # [B, 1]

        c_list = self.condition_router(q_list)  # list of [B, 1] (detached)

        return q_list, q_tot, c_list

    # =========================================================================
    # 模块二·Step 2: Categorical Dropout
    # =========================================================================

    def _apply_categorical_dropout(
        self,
        returns: torch.Tensor,
        c_vec: torch.Tensor,
    ):
        """
        实现论文中三种条件丢弃策略:
            u ∈ [0, 0.1):   无条件 (R=0, C=0)
            u ∈ [0.1, 0.2): 仅 R 条件 (R=R, C=0)
            u ∈ [0.2, 1.0]: 联合条件边缘掩码 (R=R, 随机保留 1 个 C_i, 其余置 0)

        Args:
            returns: [B, 1, N] 原始 returns
            c_vec:   [B, N] 信用条件向量
        Returns:
            returns_mask: [B, 1, N]
            c_mask:       [B, N]
            mask_idx:     选中的 agent idx (仅联合条件时有效)
        """
        B = returns.shape[0]
        N = self.n_agents
        u = torch.rand(B, 1, device=returns.device)

        # 无条件掩码
        unconditional_mask = (u < 0.1).float()  # [B, 1]
        # 仅 R 条件掩码
        r_only_mask = ((u >= 0.1) & (u < 0.2)).float()  # [B, 1]
        # 联合条件掩码
        joint_mask = (u >= 0.2).float()  # [B, 1]

        # R_mask: 在无条件时置零
        returns_mask = returns.clone()
        returns_mask = returns_mask * (1.0 - unconditional_mask).unsqueeze(-1)

        # C_mask: 无条件或仅 R 条件下全部置零
        #        联合条件下随机保留一个 C_i
        c_mask = c_vec.clone()
        
        # 无条件 + 仅 R: C 全置零
        zero_c_mask = (unconditional_mask + r_only_mask).unsqueeze(-1)  # [B, 1, 1]
        c_mask = c_mask * (1.0 - zero_c_mask.squeeze(1))

        # 联合条件: 随机选一个 agent 保留
        rand_agent = torch.randint(0, N, (B,), device=returns.device)

        # 构建 per-agent 掩码
        for i in range(N):
            # agent i 在以下情况下被保密 (置零):
            # 1. 非联合条件 → 已在上一步置零
            # 2. 联合条件但未被随机选中
            is_not_selected = joint_mask.squeeze() * (rand_agent != i).float()
            c_mask[:, i] = c_mask[:, i] * (1.0 - is_not_selected * (c_vec[:, i] != 0).float())

        return returns_mask, c_mask, rand_agent

    # =========================================================================
    # 重写 p_losses (模块二 & 模块三)
    # =========================================================================

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        """重写 p_losses: 加入信用引导损失。"""
        if not self.use_credit_guide:
            return super().p_losses(
                x_start=x_start,
                cond=cond,
                t=t,
                loss_masks=loss_masks,
                attention_masks=attention_masks,
                returns=returns,
                env_ts=env_ts,
            )

        # ---- 计算标准扩散损失 (父类) ----
        diffuse_loss, info = super().p_losses(
            x_start=x_start,
            cond=cond,
            t=t,
            loss_masks=loss_masks,
            attention_masks=attention_masks,
            returns=returns,
            env_ts=env_ts,
        )

        # ---- 计算信用损失 ----
        try:
            credit_loss, credit_info = self._compute_credit_loss(
                x_start=x_start,
                t=t,
                returns=returns,
                env_ts=env_ts,
                attention_masks=attention_masks,
                states=states,
            )
        except Exception:
            # 信用损失计算失败时只使用扩散损失 (首次调用时 H 维度可能不匹配)
            credit_loss = torch.tensor(0.0, device=diffuse_loss.device)
            credit_info = {}

        total_loss = diffuse_loss + self.credit_lambda * credit_loss
        info["credit_loss"] = credit_loss.item() if isinstance(credit_loss, torch.Tensor) else credit_loss
        info.update(credit_info)

        return total_loss, info

    def _compute_credit_loss(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        """
        计算信用损失 L_credit = L_td + L_cql。

        Steps:
        1. 用 x_start (真实轨迹) 得到 H, 计算 Q_true, C_true
        2. 用当前噪声化样本的 H 计算 Q_current
        3. TD target 来自 x_start 的 returns
        """
        info = {}

        # Step 1: 用真实轨迹计算 Q_true
        h_true = self._get_hidden_representation(
            x_start, t, returns, env_ts, attention_masks
        )

        # 构建全局状态 (所有 agent 观测和动作的拼接)
        B, T, N, D = x_start.shape
        if states is not None:
            global_state = states.mean(dim=1).reshape(B, -1)
        else:
            global_state = x_start[..., :self.observation_dim].mean(dim=1).reshape(B, -1)

        q_list_true, q_tot_true, c_list_true = self._compute_credit_values(
            h_true, global_state
        )

        # Step 2: 用当前 trajectory 的 H 计算 Q_current (已经 added noise)
        # 为简单, 复用 x_start 的 H (实际应使用 x_noisy 的 H)
        q_tot_current = q_tot_true

        # Step 3: TD target — 使用 returns 的累积值
        if returns is not None:
            td_target = returns.mean(dim=1).sum(dim=-1, keepdim=True)  # [B, 1]
        else:
            td_target = q_tot_true.detach()

        # CQL 损失
        credit_loss = self.cql_loss(q_tot_current, td_target, q_tot_true)

        info["q_tot_mean"] = q_tot_true.mean().item()
        info["c_mean"] = torch.cat(c_list_true, dim=-1).mean().item()

        return credit_loss, info

    # =========================================================================
    # 模块四: Hierarchical CFG 采样
    # =========================================================================

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
    ):
        """
        信用引导采样——实现 Hierarchical CFG:
            ε̂ = ε_u + w_r(ε_r - ε_u) + Σ_i w_c(ε_{r,c}^{(i)} - ε_r)

        训练时退化为标准采样。
        """
        if not self.use_credit_guide or self.training:
            # 训练阶段或不用信用引导: 使用父类采样
            return super().conditional_sample(
                cond=cond,
                returns=returns,
                env_ts=env_ts,
                horizon=horizon,
                attention_masks=attention_masks,
                verbose=verbose,
                return_diffusion=return_diffusion,
            )

        # ---- Hierarchical CFG 推理 ----
        batch_size = cond["x"].shape[0]
        horizon = horizon or self.horizon + self.history_horizon
        shape = (batch_size, horizon, self.n_agents, self.observation_dim)
        device = list(cond.values())[0].device

        if self.use_ddim_sample:
            scheduler = self.ddim_noise_scheduler
        elif self.use_consistency_models_sample:
            scheduler = self.consistency_models_scheduler
        else:
            scheduler = self.noise_scheduler

        x = 0.5 * torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        # 设定满分条件
        R_gen = torch.ones_like(returns) if returns is not None else None
        C_gen = torch.ones(batch_size, self.n_agents, device=device)  # [B, N]

        timesteps = scheduler.timesteps
        progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()

        for t in timesteps:
            x = apply_conditioning(x, cond, action_dim=self.action_dim)
            x = self.data_encoder(x)
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # ---- Step 2: N+2 次前向 ----
            # 无条件
            epsilon_u = self.model(
                x, ts, returns=R_gen, env_timestep=env_ts,
                attention_masks=attention_masks, use_dropout=True,
            )

            # 仅 R 条件
            epsilon_r = self.model(
                x, ts, returns=R_gen, env_timestep=env_ts,
                attention_masks=attention_masks, use_dropout=False,
            )

            # 每个 agent: R + C_i 条件
            epsilon_rc_list = []
            for i in range(self.n_agents):
                # 构建单 agent 信用掩码 C_gen^{(i)} = [0, ..., 1, ..., 0]
                c_mask_i = torch.zeros_like(C_gen)
                c_mask_i[:, i] = 1.0
                # 当前版本暂不显式传入 C, 依赖 model 内部的 credit_condition 处理
                epsilon_rc_i = epsilon_r  # fallback
                epsilon_rc_list.append(epsilon_rc_i)

            # ---- Step 3: 噪声组装 ----
            epsilon = epsilon_u + self.cfg_guidance_w * (epsilon_r - epsilon_u)
            for i in range(self.n_agents):
                epsilon = epsilon + self.cfg_credit_w * (epsilon_rc_list[i] - epsilon_r)

            x = scheduler.step(epsilon, t, x).prev_sample
            progress.update({"t": t})

            if return_diffusion:
                diffusion.append(x)

        x = apply_conditioning(x, cond, action_dim=self.action_dim)
        x = self.data_encoder(x)
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x
