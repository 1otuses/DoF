from typing import Tuple

import einops
import torch
import torch.nn as nn

from .basic import TemporalUnet, TemporalValue


class SharedIndependentTemporalUnet(nn.Module):
    """
    共享参数时间U-Net —— 所有智能体共享同一个 TemporalUnet 的参数
    核心思想：
    - agent_share_parameters = True: 所有agent共用一套网络参数
    - 将[B, T, A, F] reshape为[B*A, T, F],把agent维度合并到batch维度
    - 一次性送入共享的 TemporalUnet 处理
    - 这样每个agent看到的是相同参数的U-Net,但处理各自的观测序列
    - 虽然在参数级别不区分agent,但通过不同的returns条件可以产生差异化输出
    
    计算效率：比 TemporalUnet 节省 A 倍参数,适合同构agent场景
    """
    agent_share_parameters = True

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        transition_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,
        max_path_length: int = 100,
    ):
        super().__init__()

        self.n_agents = n_agents

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition
        self.history_horizon = history_horizon

        # 所有agent共享一个TemporalUnet
        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            env_ts_condition=env_ts_condition,
            condition_dropout=condition_dropout,
            kernel_size=kernel_size,
            max_path_length=max_path_length,
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout=True,
        force_dropout=False,
        **kwargs,
    ):
        """
        Args:
            x: [B, T, N, O] 多智能体输入
            time: [B] 扩散时间步
            returns: [B, 1, N] 条件returns
        Returns:
            x: [B, T, N, F] 去噪后输出
        """
        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        # [B, T, N, O] -> [B, N, T, O] 将agent提到batch旁
        x = einops.rearrange(x, "b t a f -> b a t f")
        bs = x.shape[0]

        # [关键] 将agent维度合并到batch: [B, A, T, F] -> [B*A, T, F]
        # 这样所有agent共享同一网络，每个agent独立前向
        x = self.net(
            x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
            time=torch.cat([time for _ in range(x.shape[1])], dim=0),  # 复制A份time
            returns=torch.cat(
                [returns[:, :, a_idx] for a_idx in range(self.n_agents)], dim=0
            )
            if returns is not None
            else None,
            env_timestep=torch.cat([env_timestep for _ in range(x.shape[1])], dim=0)
            if env_timestep is not None
            else None,
            # attention_masks=torch.cat(
            #     [attention_masks for _ in range(x.shape[1])], dim=0
            # )
            # if attention_masks is not None
            # else None,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        # 恢复shape: [B*A, T, F] -> [B, A, T, F] -> [B, T, A, F]
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        x = einops.rearrange(x, "b a t f -> b t a f")
        return x

class SharedIndependentTemporalValue(nn.Module):
    """
    共享参数时间价值网络 —— 与 SharedIndependentTemporalUnet 配套的价值网络
    
    所有agent共享一个 TemporalValue 网络,同样通过reshape将agent合并到batch维度
    最后输出每个agent的价值并返回。
    """
    agent_share_parameters = True

    def __init__(
        self,
        horizon,
        transition_dim,
        n_agents,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        self.n_agents = n_agents
        # 所有agent共享一个TemporalValue
        self.net = TemporalValue(
            horizon=horizon,
            transition_dim=transition_dim,
            dim=dim,
            dim_mults=dim_mults,
            out_dim=out_dim,
        )

    def forward(self, x, time, *args):
        """
        Args:
            x: [B, T, A, F] 多智能体观测
            time: [B] 扩散时间步
        Returns:
            out: [B, A] 每个agent独立预测的价值
        """
        assert x.shape[2] == self.n_agents, f"Expected {self.n_agents}, got {x.shape}"

        x = einops.rearrange(x, "b t a f -> b a t f")
        bs = x.shape[0]

        # 合并agent到batch: [B*A, T, F]
        out = self.net(
            x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
            time=torch.cat([time for _ in range(x.shape[1])], dim=0),
        )
        # 恢复: [B*A, 1] -> [B, A, 1]
        out = out.reshape(bs, out.shape[0] // bs, out.shape[1])

        return out
