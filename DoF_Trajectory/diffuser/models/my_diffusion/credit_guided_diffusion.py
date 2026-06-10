"""
信用引导扩散模型 (Credit-Guided Diffusion) — 修复版

继承自 GaussianDiffusion, 新增多头信用评估架构 (模块一~四)。

架构变化(相比原始版本):
    1. 独立状态编码器 (StateEncoder) 从观测序列提取 H, 不依赖扩散模型内部隐层
    2. CreditCondModelWrapper 以 FiLM 方式施加信用条件 C, 不修改原模型
    3. 修复 credit loss 计算 (使用独立编码器 + 正确的 logsumexp)
    4. 修复 conditional_sample 中 C 未被使用的问题
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.helpers import apply_conditioning
from diffuser.models.my_diffusion.credit_critic import (
    StateEncoder,
    AgentLocalCritic,
    QMixer,
    CQLLoss,
)
from diffuser.models.my_diffusion.condition_router import (
    ConditionRouter,
    build_credit_condition_vector,
)
from diffuser.models.my_diffusion.credit_model_wrapper import (
    CreditCondModelWrapper,
)

import diffuser.utils as utils


class CreditGuidedDiffusion(GaussianDiffusion):
    """
    信用引导扩散模型——在原有 DoF 框架外套一层信用评估架构。

    相比基础 GaussianDiffusion 新增:
    - state_encoder:    独立状态编码器 (从观测序列提取 H)
    - credit_critic:    信用评估头 (QMIX 风格)
    - condition_router: Q → C 路由 (归一化)
    - cql_loss:         CQL 保守正则化 (修复版)
    - cond_dropout:     三路条件丢弃 (0.1 uncond + 0.1 R-only + 0.8 R+C)

    训练时: L_total = L_diff + λ * L_credit
    推理时: Hierarchical CFG 采样
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
        # ---- 用 CreditCondModelWrapper 包装基模型 ----
        wrapped_model = CreditCondModelWrapper(
            base_model=model,
            n_agents=n_agents,
            hidden_dim=credit_hidden_dim,
        )

        # ---- 调用父类构造 (传入包装后的模型) ----
        super().__init__(
            model=wrapped_model,
            n_agents=n_agents,
            horizon=horizon,
            history_horizon=history_horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            **kwargs,
        )

        # 保存原始模型引用 (用于非 credit 场景)
        self.original_model = model

        self.use_credit_guide = use_credit_guide
        self.credit_hidden_dim = credit_hidden_dim
        self.credit_lambda = credit_lambda
        self.cql_alpha = cql_alpha
        self.credit_condition_dropout = credit_condition_dropout
        self.cfg_guidance_w = cfg_guidance_w
        self.cfg_credit_w = cfg_credit_w

        if not self.use_credit_guide:
            return

        # ---- 独立状态编码器 (不从扩散模型拿 H) ----
        self.state_encoder = StateEncoder(
            obs_dim=observation_dim,
            n_agents=n_agents,
            hidden_dim=credit_hidden_dim,
        )

        # ---- 信用评估头 ----
        self.credit_critic = AgentLocalCritic(
            n_agents=n_agents,
            hidden_dim=credit_hidden_dim,
        )
        # state_dim: 全局状态 = 所有 agent 观测沿时间池化后拼接
        self.qmixer = QMixer(
            n_agents=n_agents,
            state_dim=observation_dim * n_agents,
            hidden_dim=credit_hidden_dim,
        )
        self.cql_loss = CQLLoss(alpha=cql_alpha)

        # ---- 条件路由 ----
        self.condition_router = ConditionRouter(mode=credit_router_mode)

        # ---- 三路 dropout 的概率阈值 ----
        # [0, p_uncond):        无条件 (R=0, C=0)
        # [p_uncond, p_uncond+p_r): 仅 R 条件 (R=R, C=0)
        # [p_uncond+p_r, 1]:     联合条件 (R=R, C=随机保留一个)
        self.p_uncond = 0.1
        self.p_r_only = 0.1
        self.p_joint = 0.8

    # =========================================================================
    # Step 1: 从观测序列提取 H, 计算 Q 和 C
    # =========================================================================

    def _compute_q_and_c(self, x_start: torch.Tensor, states: torch.Tensor = None):
        """
        从观测序列直接计算 Q_i, Q_tot 和 C_i。
        使用独立状态编码器 (StateEncoder), 不依赖扩散模型的隐藏层。

        Args:
            x_start: [B, T, N, O+A] 或 [B, T, N, O] 轨迹 (取观测部分)
            states:  [B, T, state_dim] 或 None (global state)
        Returns:
            q_list:  list of [B, 1]
            q:       [B, N]
            q_tot:   [B, 1]
            c_list:  list of [B, 1] (detached, normalized)
            global_state: [B, state_dim] 用于 QMixer
        """
        B, T, N = x_start.shape[:3]

        # 取观测部分: 父类 loss() 已根据 use_inv_dyn 切片,
        # x_start 可能是 [B,T,N,O] (已切片) 或 [B,T,N,O+A] (未切片)
        if x_start.shape[-1] >= self.observation_dim + max(self.action_dim, 1):
            obs = x_start[..., self.action_dim:]  # [B, T, N, O]
        else:
            obs = x_start  # 已是最纯观测 [B, T, N, O]

        # Step 1: 编码器提取 H
        h = self.state_encoder(obs)  # [B, N, hidden_dim]

        # Step 2: 构建全局状态 (用于 QMixer 的 hypernetwork)
        if states is not None:
            # states: [B, T, state_dim] -> 沿时间池化
            global_state = states.mean(dim=1)  # [B, state_dim]
        else:
            # fallback: 所有 agent 的观测沿时间池化后拼接
            obs_pooled = obs.mean(dim=1)  # [B, N, O]
            global_state = obs_pooled.reshape(B, -1)  # [B, N*O]

        # Step 3: 计算 Q
        q_list, q = self.credit_critic(h)  # list of [B, 1], [B, N]
        q_tot = self.qmixer(q, global_state)  # [B, 1]

        # Step 4: 计算 C (归一化后 detached)
        c_list = self.condition_router(q_list)  # list of [B, 1], detached

        return q_list, q, q_tot, c_list, global_state

    # =========================================================================
    # Step 2: Categorical Dropout (R_mask, C_mask)
    # =========================================================================

    def _sample_condition_masks(self, B: int, device: torch.device):
        """
        采样三路条件掩码:
            mode 0: 无条件 (R=0, C=0)
            mode 1: 仅 R 条件 (R=R, C=0)
            mode 2: 联合条件 (R=R, C=随机保留一个 agent)

        Returns:
            mode:      [B] 整数 0/1/2
            r_mask:    [B] 1=保留 R, 0=丢弃 R
            c_keep_idx: [B] 保留 C 的 agent idx (仅 mode=2 有效)
        """
        u = torch.rand(B, device=device)
        mode = torch.where(
            u < self.p_uncond, 0,
            torch.where(u < self.p_uncond + self.p_r_only, 1, 2)
        )
        r_mask = (mode >= 1).float()  # mode 1,2 保留 R
        c_keep_idx = torch.randint(0, self.n_agents, (B,), device=device)
        return mode, r_mask, c_keep_idx

    def _build_conditioned_returns(
        self,
        returns: torch.Tensor,
        r_mask: torch.Tensor,
    ):
        """根据 r_mask 构造条件/无条件 returns。"""
        # returns: [B, 1, N]
        if returns is None:
            return None
        cond_returns = returns.clone()
        # 无条件时: returns 置零
        cond_returns = cond_returns * r_mask.view(-1, 1, 1)
        return cond_returns

    def _build_conditioned_credit(
        self,
        c_vec: torch.Tensor,
        mode: torch.Tensor,
        c_keep_idx: torch.Tensor,
    ):
        """根据 mode 构造条件/无条件 C 向量。"""
        B, N = c_vec.shape
        cond_c = torch.zeros_like(c_vec)  # [B, N]

        for i in range(N):
            # mode=2 (联合条件) 且被选中的 agent 保留其 C
            keep = (mode == 2) & (c_keep_idx == i)
            cond_c[keep, i] = c_vec[keep, i]

        return cond_c  # [B, N]

    # =========================================================================
    # 重写 get_model_output: 传递 credit 条件
    # =========================================================================

    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        credit: Optional[torch.Tensor] = None,
    ):
        """
        获取模型输出, 支持传入 credit 条件。

        当 credit 不为 None 时, 会通过 CreditCondModelWrapper 的 FiLM 调制。
        同时兼容父类的 use_learnable_agent_weights / use_qmix_combiner。
        """
        if credit is None:
            return super().get_model_output(
                x, t, returns, env_ts, attention_masks, states,
            )

        # ---- 带 credit 条件的 CFG ----
        # 注意: self.model 已经是 CreditCondModelWrapper
        if self.returns_condition:
            # 三路预测: 无条件 / 仅 R / R+C
            zero_returns = torch.zeros_like(returns) if returns is not None else None
            zero_credit = torch.zeros_like(credit)

            epsilon_uncond = self.model(
                x, t,
                returns=zero_returns, credit=zero_credit,
                env_timestep=env_ts, attention_masks=attention_masks,
                use_dropout=True,
            )
            epsilon_r_only = self.model(
                x, t,
                returns=returns, credit=zero_credit,
                env_timestep=env_ts, attention_masks=attention_masks,
                use_dropout=False,
            )
            epsilon_rc = self.model(
                x, t,
                returns=returns, credit=credit,
                env_timestep=env_ts, attention_masks=attention_masks,
                use_dropout=False,
            )

            # 兼容父类的可学习智能体权重 (与父类逻辑一致)
            if self.use_learnable_agent_weights:
                w = self.agent_weights.view(1, 1, -1, 1)
                w_norm = self.agent_weights.sum()
                epsilon_uncond = (epsilon_uncond * w) / w_norm
                epsilon_r_only = (epsilon_r_only * w) / w_norm
                epsilon_rc = (epsilon_rc * w) / w_norm

            # Hierarchical CFG:
            # ε = ε_u + w_r*(ε_rc - ε_r_only) + w_c*(ε_r_only - ε_u)
            epsilon = (
                epsilon_uncond
                + self.condition_guidance_w * (epsilon_rc - epsilon_r_only)
                + self.cfg_credit_w * (epsilon_r_only - epsilon_uncond)
            )
        else:
            epsilon = self.model(
                x, t,
                returns=returns, credit=credit,
                env_timestep=env_ts, attention_masks=attention_masks,
                use_dropout=False,
            )
            if self.use_learnable_agent_weights:
                w = self.agent_weights.view(1, 1, -1, 1)
                epsilon = (epsilon * w) / self.agent_weights.sum()

        return epsilon

    # =========================================================================
    # 重写 p_losses: 加入信用引导损失
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
        if not self.use_credit_guide:
            return super().p_losses(
                x_start, cond, t, loss_masks,
                attention_masks, returns, env_ts,
            )

        B = x_start.shape[0]

        # ---- Step 1: 从真实轨迹计算 Q 和 C ----
        q_list, q, q_tot, c_list, global_state = self._compute_q_and_c(
            x_start, states,
        )

        # ---- Step 2: Categorical Dropout ----
        c_vec = build_credit_condition_vector(c_list)  # [B, N]
        mode, r_mask, c_keep_idx = self._sample_condition_masks(B, x_start.device)

        # 构造条件/无条件的 returns 和 credit
        cond_returns = self._build_conditioned_returns(returns, r_mask)
        cond_c = self._build_conditioned_credit(c_vec, mode, c_keep_idx)

        # ---- Step 3: 扩散损失 (带条件) ----
        # 标准扩散加噪 + 预测噪声 (使用条件 returns 和 credit)
        noise = torch.randn_like(x_start)
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        x_noisy = apply_conditioning(x_noisy, cond, action_dim=self.action_dim)
        x_noisy = self.data_encoder(x_noisy)

        # 获取模型预测 (带 credit 条件)
        epsilon = self.get_model_output(
            x_noisy, t,
            returns=cond_returns,
            env_ts=env_ts,
            attention_masks=attention_masks,
            states=states,
            credit=cond_c,  # 传入 credit 条件
        )

        # 扩散损失
        if self.predict_epsilon:
            diffuse_loss, info = self.loss_fn(epsilon, noise)
        else:
            diffuse_loss, info = self.loss_fn(epsilon, x_start)

        loss = (diffuse_loss * loss_masks).mean(dim=[1, 2]).mean()

        # ---- Step 4: 信用损失 L_credit = L_td + L_cql ----
        if returns is not None:
            # TD target: 使用 returns 作为近似的 Q_target
            # returns: [B, 1, N] -> [B, 1] (全局)
            td_target = returns.mean(dim=-1)  # [B, 1]
        else:
            td_target = q_tot.detach()

        credit_loss = self.cql_loss(q_tot, td_target)
        total_loss = loss + self.credit_lambda * credit_loss

        info["diffuse_loss"] = loss.detach()
        info["credit_loss"] = credit_loss.detach()
        info["q_tot_mean"] = q_tot.mean().detach()
        info["c_mean"] = c_vec.mean().detach()
        info["cond_mode_uncond"] = (mode == 0).float().mean().detach()
        info["cond_mode_r"] = (mode == 1).float().mean().detach()
        info["cond_mode_joint"] = (mode == 2).float().mean().detach()

        return total_loss, info

    # =========================================================================
    # Hierarchical CFG 采样 (修复版)
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
        if not self.use_credit_guide or self.training:
            return super().conditional_sample(
                cond, returns, env_ts, horizon,
                attention_masks, verbose, return_diffusion,
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

        # ---- 设定满分条件 ----
        # 推理时使用满分 R=1.0 和满分 C=1.0
        R_gen = torch.ones(batch_size, 1, self.n_agents, device=device)
        # C_gen: [B, N], 推理时每个 agent 都分配满分信用
        C_gen = torch.ones(batch_size, self.n_agents, device=device)

        timesteps = scheduler.timesteps
        progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()

        for t in timesteps:
            x = apply_conditioning(x, cond, action_dim=self.action_dim)
            x = self.data_encoder(x)
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # ---- Hierarchical CFG: N+2 次前向 ----
            # 1) 无条件: R=0, C=0
            zero_returns = torch.zeros_like(R_gen)
            zero_credit = torch.zeros(batch_size, self.n_agents, device=device)

            epsilon_u = self.model(
                x, ts,
                returns=zero_returns,
                env_timestep=env_ts,
                attention_masks=attention_masks,
                credit=zero_credit,
                use_dropout=True,
            )

            # 2) 仅 R 条件: R=R, C=0
            epsilon_r = self.model(
                x, ts,
                returns=R_gen,
                env_timestep=env_ts,
                attention_masks=attention_masks,
                credit=zero_credit,
                use_dropout=False,
            )

            # 3) R+C_i 条件: 对每个 agent i, R=R, C=one-hot(i)
            epsilon_rc_list = []
            for i in range(self.n_agents):
                c_onehot = torch.zeros(batch_size, self.n_agents, device=device)
                c_onehot[:, i] = 1.0  # 仅 agent i 有信用

                epsilon_rc_i = self.model(
                    x, ts,
                    returns=R_gen,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    credit=c_onehot,
                    use_dropout=False,
                )
                epsilon_rc_list.append(epsilon_rc_i)

            # ---- 噪声组装 ----
            # ε = ε_u + w_r(ε_r - ε_u) + w_c * Σ_i (ε_{rc,i} - ε_r)
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
