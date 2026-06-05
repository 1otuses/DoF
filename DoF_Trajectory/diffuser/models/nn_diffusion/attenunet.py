from typing import Tuple

import einops
import torch
from torch import nn
from torch.distributions import Bernoulli


from .basic import (
    SelfAttention,
    Downsample1d,
    ResidualTemporalBlock,
    SinusoidalPosEmb,
    TemporalUnet,
    TemporalSelfAttention,  # 时间步感知的跨智能体自注意力
)


class ConvAttentionDeconv(nn.Module):
    """
    卷积-注意力-解卷积网络 —— 在U-Net编解码的瓶颈层和跳连处插入跨智能体注意力
    
    核心架构：
    - agent_share_parameters = False: 每个agent独立拥有一个完整的 TemporalUnet
    - 在U-Net瓶颈层(最底部)和每级上采样前,插入 SelfAttention / TemporalSelfAttention 模块
    - 注意力在 agent 维度上进行,实现agent间的信息交互
    
    与 IndependentTemporalUnet 的区别：
    - 每个agent各自编解码,但通过注意力层共享信息
    - 计算量更大但具备更好的agent间协同能力
    
    两种注意力模式：
    1. use_temporal_attention=True: TemporalSelfAttention(带时间步嵌入)
    2. use_temporal_attention=False: SelfAttention(普通自注意力)
    """
    agent_share_parameters = False

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        dim: int = 128,
        history_horizon: int = 0,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        n_agents: int = 2,
        returns_condition: bool = False,
        env_ts_condition: bool = False,
        condition_dropout: float = 0.1,
        kernel_size: int = 5,
        residual_attn: bool = True,
        use_layer_norm: bool = False,
        max_path_length: int = 100,
        use_temporal_attention: bool = True,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.use_temporal_attention = use_temporal_attention

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition

        # 每个分辨率层级的输入/输出维度
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 每个agent独立的TemporalUnet
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
                    max_path_length=max_path_length,
                    kernel_size=kernel_size,
                )
                for _ in range(n_agents)
            ]
        )

        # --- 构建跨智能体注意力层 ---
        # 为每个分辨率层级（瓶颈+各级上采样前）创建注意力模块
        if self.use_temporal_attention:
            print("\n USE TEMPORAL ATTENTION !!! \n")
            AttentionModule = TemporalSelfAttention
            self.self_attn = [
                AttentionModule(
                    in_out[-1][1], in_out[-1][1] // 16, in_out[-1][1] // 4,
                    residual=residual_attn, embed_dim=2,
                )
            ]
            for dims in reversed(in_out):
                self.self_attn.append(
                    AttentionModule(
                        dims[1], dims[1] // 16, dims[1] // 4,
                        residual=residual_attn, embed_dim=2,
                    )
                )
        else:
            self.self_attn = [
                SelfAttention(in_out[-1][1], in_out[-1][1] // 16, in_out[-1][1] // 4, residual=residual_attn)
            ]
            for dims in reversed(in_out):
                self.self_attn.append(
                    SelfAttention(dims[1], dims[1] // 16, dims[1] // 4, residual=residual_attn)
                )
        self.self_attn = nn.ModuleList(self.self_attn)

        # --- 可选LayerNorm ---
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            horizon_ = horizon
            self.layer_norm = []
            for dims in in_out:
                self.layer_norm.append(nn.LayerNorm([dims[1], horizon_]))
                horizon_ = horizon_ // 2
            horizon_ = horizon_ * 2
            self.layer_norm.append(nn.LayerNorm([in_out[-1][1], horizon_]))
            self.layer_norm = list(reversed(self.layer_norm))
            self.layer_norm = nn.ModuleList(self.layer_norm)

            horizon_ = horizon
            self.layer_norm_cat = []
            for dims in in_out:
                self.layer_norm_cat.append(nn.LayerNorm([dims[1] * 2, horizon_]))
                horizon_ = horizon_ // 2
            self.layer_norm_cat = list(reversed(self.layer_norm_cat))
            self.layer_norm_cat = nn.ModuleList(self.layer_norm_cat)

    def forward(
        self,
        x,
        time,
        returns=None,
        states=None,
        env_timestep=None,
        attention_masks=None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        **kwargs,
    ):
        """
        Args:
            x: [B, T, A, F] 多智能体输入
            time: [B] 扩散时间步
            returns: [B, T, A] 条件returns
        Returns:
            x: [B, T, A, F] 去噪后输出
        """
        assert x.shape[2] == self.n_agents, f"Expected {self.n_agents}, got {x.shape}"

        # [B, T, A, F] -> [B, A, F, T]  (Conv1D需要的格式: channel维度在-2)
        x = einops.rearrange(x, "b t a f -> b a f t")
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # 拆成A个 [B, F, T]

        # 时间步嵌入: 每个agent独立
        t = [self.nets[i].time_mlp(time) for i in range(self.n_agents)]

        # --- Returns条件嵌入（可选的CFG）---
        if self.returns_condition:
            assert returns is not None
            returns_embed = [
                self.nets[i].returns_mlp(returns[:, :, i]) for i in range(self.n_agents)
            ]
            if use_dropout:
                mask = (self.nets[0].mask_dist.sample(
                    sample_shape=(returns_embed[0].size(0), 1)
                ).to(returns_embed[0].device))
                returns_embed = [returns_embed[i] * mask for i in range(len(returns_embed))]
            if force_dropout:
                returns_embed = [returns_embed[i] * 0 for i in range(len(returns_embed))]
            t = [torch.cat([t[i], returns_embed[i]], dim=-1) for i in range(len(t))]

        # --- 环境时间步嵌入 ---
        if self.env_ts_condition:
            assert env_timestep is not None
            env_ts_embed = [
                self.nets[i].env_ts_mlp(env_timestep) for i in range(self.n_agents)
            ]
            t = [torch.cat([t[i], env_ts_embed[i]], dim=-1) for i in range(len(t))]

        # ========== 编码器（下采样） ==========
        h = [[] for _ in range(self.n_agents)]
        for layer_idx in range(len(self.nets[0].downs)):
            for i in range(self.n_agents):
                resnet, resnet2, downsample = self.nets[i].downs[layer_idx]
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                h[i].append(x[i])       # 保存跳连
                x[i] = downsample(x[i])  # 时间维下采样

        # ========== 瓶颈层 ==========
        for i in range(self.n_agents):
            x[i] = self.nets[i].mid_block1(x[i], t[i])
            x[i] = self.nets[i].mid_block2(x[i], t[i])

        # ---------- 跨智能体注意力（瓶颈处）----------
        # 在agent维度做attention，融合各agent的瓶颈特征
        x = self.self_attn[0](torch.stack(x, dim=1))  # [B, A, F, T]
        if self.use_layer_norm:
            x = self.layer_norm[0](x)
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # 拆回A个

        # ========== 解码器（上采样） ==========
        for layer_idx in range(len(self.nets[0].ups)):
            # 跳连特征: 收集所有agent的对应层隐藏状态
            hiddens = torch.stack([hid.pop() for hid in h], dim=1)  # [B, A, F, T]
            if self.use_layer_norm:
                hiddens = self.layer_norm[layer_idx + 1](hiddens)
            # 在跳连处也做跨智能体注意力
            hiddens = self.self_attn[layer_idx + 1](hiddens)
            for i in range(self.n_agents):
                resnet, resnet2, upsample = self.nets[i].ups[layer_idx]
                x[i] = torch.cat((x[i], hiddens[:, i]), dim=1)  # 跳连拼接
                if self.use_layer_norm:
                    x[i] = self.layer_norm_cat[layer_idx](x[i])
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                x[i] = upsample(x[i])  # 时间维上采样

        # 最终卷积
        for i in range(self.n_agents):
            x[i] = self.nets[i].final_conv(x[i])

        # 恢复原始shape: [B, A, F, T] -> [B, T, A, F]
        x = torch.stack(x, dim=1)
        x = einops.rearrange(x, "b a f t -> b t a f")

        return x


class ConvAttentionTemporalValue(nn.Module):
    """
    卷积注意力时间价值网络 —— 每个agent独立编码+注意力融合的多智能体价值网络
    
    与 ConvAttentionDeconv 配套的价值网络版本：
    - 每个agent独立卷积编码(含时间嵌入)
    - 每层编码后通过SelfAttention融合agent间信息
    - 最后对所有agent的价值取平均得到全局价值
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

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.n_agents = n_agents
        # 每个agent独立的时间嵌入MLP
        self.time_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    SinusoidalPosEmb(dim),
                    nn.Linear(dim, dim * 4),
                    nn.Mish(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(n_agents)
            ]
        )

        # 每个agent独立的卷积编码器
        self.blocks = nn.ModuleList([nn.ModuleList([]) for _ in range(n_agents)])
        num_resolutions = len(in_out)

        print("ConvAttentionTemporalValue: ", in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            for i in range(n_agents):
                self.blocks[i].append(
                    nn.ModuleList([
                        ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim),
                        ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ])
                )
            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        # 每个agent独立的中间层
        self.mid_block1 = nn.ModuleList([
            ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim) for _ in range(n_agents)
        ])
        self.mid_block2 = nn.ModuleList([
            ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim) for _ in range(n_agents)
        ])
        fc_dim = mid_dim_3 * max(horizon, 1)

        # 每个agent独立的输出头
        self.final_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fc_dim + time_dim, fc_dim // 2),
                nn.Mish(),
                nn.Linear(fc_dim // 2, out_dim),
            ) for _ in range(n_agents)
        ])
        # 每层编码后的跨智能体注意力
        self.self_attn = nn.ModuleList(
            [SelfAttention(dim[1], dim[1] // 16) for dim in in_out]
        )

    def forward(self, x, time, *args):
        """
        Args:
            x: [B, T, A, F] 多智能体观测
            time: [B] 扩散时间步
        Returns:
            out: [B, 1] 全局价值(所有agent均值)
        """
        assert x.shape[2] == self.n_agents, f"Expected {self.n_agents}, got {x.shape}"

        x = einops.rearrange(x, "b t a f -> b a f t")
        x = [x[:, a_idx] for a_idx in range(x.shape[1])]  # A个 [B, F, T]

        t = [self.time_mlp[i](time) for i in range(self.n_agents)]

        # 编码器：每层后做跨智能体注意力
        for layer_idx in range(len(self.blocks[0])):
            for i in range(self.n_agents):
                resnet, resnet2, downsample = self.blocks[i][layer_idx]
                x[i] = resnet(x[i], t[i])
                x[i] = resnet2(x[i], t[i])
                x[i] = downsample(x[i])
            # 跨智能体注意力融合
            x = self.self_attn[layer_idx](torch.stack(x, dim=1))
            x = [x[:, a_idx] for a_idx in range(x.shape[1])]

        # 每个agent独立预测价值
        for i in range(self.n_agents):
            x[i] = self.mid_block1[i](x[i], t[i])
            x[i] = self.mid_block2[i](x[i], t[i])
            x[i] = x[i].view(len(x[i]), -1)
            x[i] = self.final_block[i](torch.cat([x[i], t[i]], dim=-1))
        x = torch.stack(x, dim=1).squeeze(-1)  # [B, A, 1]

        # 对所有agent取平均得到全局价值
        out = x.mean(axis=1, keepdim=True)  # [B, 1]
        return out