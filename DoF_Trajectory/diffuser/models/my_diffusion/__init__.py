"""
my_diffusion 包 — 信用引导多智能体扩散模型 (Credit-Guided Diffusion for MARL)

架构总览:
    共享编码器 → 隐变量 H ─┬─ Credit Critic Head → Q_i, Q_tot
                           ├─ Condition Router → C_i (detach, 归一化)
                           └─ Diffusion Denoising Head → ε̂

模块:
    - credit_critic.py:           信用评估头 (AgentLocalCritic/QMixer/CQLLoss)
    - condition_router.py:        条件路由与归一化 (ConditionRouter)
    - credit_guided_diffusion.py: 信用引导扩散模型 (CreditGuidedDiffusion)
"""

from diffuser.models.my_diffusion.credit_critic import (
    AgentLocalCritic,
    QMixer,
    CQLLoss,
)
from diffuser.models.my_diffusion.condition_router import (
    ConditionRouter,
    build_credit_condition_vector,
)
from diffuser.models.my_diffusion.credit_guided_diffusion import (
    CreditGuidedDiffusion,
)

__all__ = [
    "AgentLocalCritic",
    "QMixer",
    "CQLLoss",
    "ConditionRouter",
    "build_credit_condition_vector",
    "CreditGuidedDiffusion",
]
