from typing import Tuple

import einops
import torch
import torch.nn as nn

from .basic import (
    TemporalUnet,
    Downsample1d,
    ResidualTemporalBlock,
    SinusoidalPosEmb,
)


class ConcatenatedTemporalUnet(nn.Module):
    """
    拼接式时间U-Net —— 将所有智能体的观测拼成一个向量,送入单个U-Net处理
    
    核心思想：
    - agent_share_parameters = False: 不共享参数,但通过拼接将所有agent信息融合
    - 每个agent的观测在 feature 维度拼接: [B, T, A, F] -> [B, T, A*F]
    - 拼接后由单个 TemporalUnet 处理,天然捕捉agent间的交互
    - returns条件: 对所有agent的returns取平均作为全局条件
    
    适用场景:智能体数量少且固定,需要agent间信息交互的任务
    """
    agent_share_parameters = False

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = False,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        # 单个U-Net，输入维度为所有agent的拼接: transition_dim * n_agents
        self.net = TemporalUnet(
            horizon=horizon,
            history_horizon=history_horizon,
            transition_dim=transition_dim * n_agents,  # [关键] 所有agent拼成一个大向量
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
        use_dropout: bool = True,
        force_dropout: bool = False,
    ):
        """
        Args:
            x: [B, T, A, F] 多智能体观测/噪声
            time: [B] 扩散时间步
            returns: [B, 1, A] 条件returns
        Returns:
            x: [B, T, A, F] 去噪后的输出
        """
        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        # 在feature维度拼接所有agent: [B, T, A, F] -> [B, T, A*F]
        concat_x = einops.rearrange(x, "b h a f -> b h (a f)")
        # returns取所有agent的均值作为全局条件
        concat_x = self.net(
            concat_x,
            time=time,
            returns=returns.mean(dim=2) if returns is not None else None,
            env_timestep=env_timestep,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        # 拆回多agent: [B, T, A*F] -> [B, T, A, F]
        x = einops.rearrange(concat_x, "b h (a f) -> b h a f", a=self.n_agents)

        return x


class IndependentTemporalUnet(nn.Module):
    """
    独立时间U-Net —— 每个智能体拥有独立的 TemporalUnet 网络
    
    核心思想：
    - agent_share_parameters = False: 每个agent有自己独立的网络参数
    - 每个agent独立进行U-Net编码-解码,agent间没有显式信息交互
    - 计算量随agent数量线性增长
    
    适用场景:agent间基本独立/弱交互的任务,或作为baseline对比
    """
    agent_share_parameters = False

    def __init__(
        self,
        n_agents: int,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = False,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        # 每个agent一个独立的TemporalUnet，参数不共享
        self.nets = nn.ModuleList(
            [
                TemporalUnet(
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
                for _ in range(n_agents)
            ]
        )

    def forward(
        self,
        x,
        time,
        returns=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        Args:
            x: [B, T, A, F] 多智能体观测/噪声
            time: [B] 扩散时间步
            returns: [B, 1, A] 条件returns
        Returns:
            [B, T, A, F] 每个agent独立去噪后的输出
        """
        assert x.shape[2] == self.n_agents, f"{x.shape}, {self.n_agents}"

        x_list = []
        # 每个agent独立前向传播
        for a_idx in range(self.n_agents):
            x_list.append(
                self.nets[a_idx](
                    x[:, :, a_idx, :],            # [B, T, F] 取第a_idx个agent
                    time=time,
                    returns=returns[:, :, a_idx] if returns is not None else None,
                    env_timestep=env_timestep,
                    use_dropout=use_dropout,
                    force_dropout=force_dropout,
                )
            )
        x_list = torch.stack(x_list, dim=2)       # [B, T, A, F]
        return x_list


class ConcatTemporalValue(nn.Module):
    """
    拼接式时间价值网络 —— 将所有agent拼接后预测联合状态价值
    
    核心思想：
    - 与 ConcatenatedTemporalUnet 配套的价值网络版本
    - 将所有agent在feature维度拼接,通过卷积下采样 + 全连接输出标量价值
    - 用于 returns_loss_guided 中的价值引导
    
    结构:Conv1D下采样 -> 残差块 -> 全连接 -> 标量价值
    """
    agent_share_parameters = False

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

        # 输入维度 = transition_dim * n_agents (所有agent拼接)
        dims = [transition_dim * n_agents, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.n_agents = n_agents
        # 时间步嵌入MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print("ConcatTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # 每个分辨率层级：两个残差块 + 下采样
            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim),
                        ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        # 中间层: 进一步压缩特征
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim)
        fc_dim = mid_dim_3 * max(horizon, 1)

        # 最终全连接: 特征+时间嵌入 -> 标量价值
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, time, *args):
        """
        Args:
            x: [B, T, A, F] 多智能体观测
            time: [B] 扩散时间步
        Returns:
            out: [B, 1] 预测的标量价值
        """
        assert x.shape[2] == self.n_agents, f"Expected {self.n_agents}, got {x.shape}"

        # 拼接所有agent: [B, T, A, F] -> [B, T, A*F] -> [B, F', T]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = einops.rearrange(x, "b t f -> b f t")
        t = self.time_mlp(time)

        # 卷积编码器
        for layer_idx, (resnet, resnet2, downsample) in enumerate(self.blocks):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        # 中间层
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # 展平 + 全连接输出
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))

        return out
