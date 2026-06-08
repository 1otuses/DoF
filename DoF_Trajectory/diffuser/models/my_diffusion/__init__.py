"""
my_diffusion 包 — 信用引导多智能体扩散模型 (Credit-Guided Diffusion for MARL)

架构总览:
    观测序列 → StateEncoder → H ─┬─ Credit Critic → Q_i → QMIX → Q_tot
                                   ├─ ConditionRouter → C_i (detach, 归一化)
                                   └─ (与 x,t,returns 一起送入) →
                                      CreditCondModelWrapper → ε̂ (FiLM 调制)

流水线:
    训练: L_total = L_diff + λ * L_credit
         L_diff:  标准扩散 MSE (受 R 和 C 条件引导)
         L_credit: TD + CQL (从观测序列独立计算)
    推理: Hierarchical CFG
         ε̂ = ε_u + w_r(ε_r - ε_u) + w_c * Σ_i(ε_{rc,i} - ε_r)

关键设计:
    - StateEncoder 独立于扩散模型, 不提 H 从共享编码器
    - CreditCondModelWrapper 以 FiLM 方式施加 C, 不改原模型代码
"""

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
from diffuser.models.my_diffusion.credit_guided_diffusion import (
    CreditGuidedDiffusion,
)

__all__ = [
    "StateEncoder",
    "AgentLocalCritic",
    "QMixer",
    "CQLLoss",
    "ConditionRouter",
    "build_credit_condition_vector",
    "CreditCondModelWrapper",
    "CreditGuidedDiffusion",
]
