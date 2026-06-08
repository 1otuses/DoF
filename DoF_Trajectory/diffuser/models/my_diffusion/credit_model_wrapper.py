"""
信用条件模型包装器 (Credit Condition Model Wrapper)

在不修改原模型架构的前提下，通过 FiLM（Feature-wise Linear Modulation）
在模型输出层施加信用条件 C 的调制。

原理:
    out_base = model(x, t, returns, ...)
    out = out_base * γ(C) + β(C)   ← FiLM 调制，不影响模型内部计算

优点:
    - 完全不需要修改原模型 (TemporalUnet/SharedConvAttentionDeconv/...)
    - 原模型权重可直接加载复用
    - 新增的 γ_net, β_net 参数量很小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CreditCondModelWrapper(nn.Module):
    """
    信用条件模型包装器。

    通过 FiLM 调制在模型输出层施加信用条件 C 的影响:
        out = base_model(x, t, returns, ...)
        out = out * gamma(C) + beta(C)

    C 的形状: [B, N] (per-agent 信用分配值)
    gamma, beta: [B, N, 1] → broadcast 到 [B, T, N, O]
    """

    def __init__(self, base_model, n_agents: int, hidden_dim: int = 64):
        super().__init__()
        self.base_model = base_model
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # FiLM 参数生成网络: C → gamma (乘性), beta (加性)
        # 每个 agent 独立 gamma/beta
        self.gamma_net = nn.Sequential(
            nn.Linear(n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )

        # 初始化 gamma ≈ 1, beta ≈ 0 (即无调制)
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def has_credit(self, **kwargs):
        """检查是否传入了 credit 条件 (兼容 **kwargs 和显式参数)。"""
        return "credit" in kwargs and kwargs["credit"] is not None

    def forward(self, x, time, returns=None, **kwargs):
        """
        Args:
            x:       [B, T, N, O] 输入
            time:    [B] 扩散时间步
            returns: [B, 1, N] 或 None
            **kwargs: 透传给 base_model, 包含 credit: [B, N] 或 None
        Returns:
            out: [B, T, N, O] 调制后的输出
        """
        # 1. 调用基模型 (透传所有参数, credit 被 **kwargs 吸收)
        out = self.base_model(
            x, time,
            returns=returns,
            **{k: v for k, v in kwargs.items() if k != "credit"},
        )
        # out: [B, T, N, O] — 基模型的噪声预测

        # 2. 如果提供了 credit, 施加 FiLM 调制
        if self.has_credit(**kwargs):
            credit = kwargs["credit"]  # [B, N]
            gamma = self.gamma_net(credit)  # [B, N]
            beta = self.beta_net(credit)    # [B, N]

            # unsqueeze 以匹配 out 的维度: [B, 1, N, 1] -> broadcast
            gamma = gamma.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            beta = beta.unsqueeze(1).unsqueeze(-1)    # [B, 1, N, 1]

            out = out * gamma + beta

        return out
