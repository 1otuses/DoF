"""
条件路由与归一化层 (Condition Router)

模块二·Step 4 & 模块三: 将 Credit Critic 输出的 Q_i 归一化为信用条件向量 C。
提供 Min-Max 和 Softmax 两种归一化模式。
"""
import torch
import torch.nn as nn


class ConditionRouter(nn.Module):
    """
    条件路由: Q_i → C_i (detach → 归一化)。
    
    两种归一化模式:
    - minmax: C_i = (Q_i - Q_min) / (Q_max - Q_min + ε), 值域 [0, 1]
    - softmax: C_i = softmax(Q_1..Q_N / τ)_i, 值域 (0, 1), 总和为 1
    """

    def __init__(self, mode: str = "minmax", temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        assert mode in ("minmax", "softmax"), f"Unsupported mode: {mode}"
        self.mode = mode
        self.temperature = temperature
        self.eps = eps

    def forward(self, q_list):
        """
        Args:
            q_list: list of [B, 1] per-agent Q values
        Returns:
            c_list: list of [B, 1] per-agent credit conditions (detached, normalized)
        """
        q = torch.cat(q_list, dim=-1)  # [B, N]
        q = q.detach()  # 截断梯度,信用条件不反向传播到 Q 学习

        if self.mode == "minmax":
            q_min = q.min(dim=-1, keepdim=True).values  # [B, 1]
            q_max = q.max(dim=-1, keepdim=True).values  # [B, 1]
            c = (q - q_min) / (q_max - q_min + self.eps)  # [B, N]
        elif self.mode == "softmax":
            c = torch.softmax(q / self.temperature, dim=-1)  # [B, N]

        c_list = [c[:, i:i+1] for i in range(c.shape[-1])]
        return c_list


def build_credit_condition_vector(c_list, mask=None):
    """
    将归一化的 C_i 列表拼接为完整信用条件向量。
    
    Args:
        c_list: list of [B, 1] per-agent credit conditions
        mask:   optional list of bool, True 表示该 agent 的 C 被遮挡
    Returns:
        c_vec: [B, N] 信用条件向量
    """
    c_vec = torch.cat(c_list, dim=-1)  # [B, N]
    if mask is not None:
        for i, masked in enumerate(mask):
            if masked:
                c_vec[:, i] = 0.0  # 遮挡位置置零
    return c_vec
