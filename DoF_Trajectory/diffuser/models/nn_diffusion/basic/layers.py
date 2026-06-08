import math
from typing import Tuple
from numbers import Number
import numpy as np
import einops
import torch
import torch.nn as nn
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from .modules import Conv1dBlock, Downsample1d, SinusoidalPosEmb, Upsample1d


class Residual(nn.Module): # 残差层
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module): # 预归一化层
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # 2D 实例归一化,对每个样本独立做归一化
        # affine=True → 可学习的缩放和平移参数
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class TemporalLinearAttention(nn.Module):
    """
    ============ 时间线性注意力层 (Temporal Linear Attention) ============
    一种高效的注意力机制, 使用线性注意力替代标准 Softmax 注意力,
    将计算复杂度从 O(n^2) 降低到 O(n)
    核心思想 (Katharopoulos et al., 2020):
        用特征映射 φ 近似 Softmax:  attention = φ(Q) · (φ(K)^T · V)
        先计算 K^T V (全局上下文), 再与 Q 相乘, 避免显式构建 n×n 注意力矩阵
    这里的具体实现:
        - 对 key 做 softmax (而不是常规的 linear attention)
        - 通过矩阵结合律: (K^T V) @ Q 替代 K^T @ V 的注意力矩阵
        - 融入时间步嵌入作为偏置

    输入形状: [batch, n_agents, feat_dim, horizon]
    输出形状: [batch, n_agents, feat_dim, horizon]
    ====================================================================
    """
    def __init__(self, dim, embed_dim: int, heads=4, dim_head=128, residual: bool = False,):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        # 2D 卷积同时生成 Q, K, V (共享卷积, 输出通道 = hidden_dim * 3)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)  # 输出投影
        # 时间嵌入 -> QKV 偏置 (调制注意力)
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, hidden_dim * 3)     # embed_dim -> hidden_dim*3
        )
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))  # 可学习的残差缩放系数

    def forward(self, x, time):
        """
        x    : [batch, n_agents, feat_dim, horizon]   多智能体特征
        time : [batch, n_agents, embed_dim]           时间步嵌入
        """
        y = x.clone()
        # 维度重排: 交换智能体和特征维度
        x = rearrange(x, "b a f t -> b f a t")   # [b, f, a, t]
        time = self.time_mlp(time)                # [b, a, hidden*3]
        time = rearrange(time, "b a f -> b f a 1") # [b, f, a, 1] 这里f=hidden_dim*3
        b, c, h, w = x.shape  # b=batch, c=feat_dim, h=n_agents, w=horizon

        # 卷积生成 QKV 并加上时间偏置
        qkv = self.to_qkv(x) + time  # [b, hidden*3, a, t]

        # 拆分为 Q, K, V, 并重组为多头的形式
        # (b h) = batch*heads, heads = nheads
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv (b h) heads c w",
            heads=self.heads, qkv=3
        )  # 每个: [(b*h), heads, c, w]  其中 c = dim_head, w = horizon

        # ★ 线性注意力核心: 先对 K 做 softmax, 再计算 context = K^T V
        k = k.softmax(dim=-1)  # 在宽度 (horizon) 维度上做 softmax

        # context = K^T @ V : [heads, c, w] @ [heads, c, w]^? ...
        # 实际: einsum("bhdn,bhen->bhde") 表示 k[b,h,d,n] @ v[b,h,e,n] -> context[b,h,d,e]
        # 即: (n=horizon 维度求和) → 得到全局上下文 [b, h, dim_head, dim_head]
        context = torch.einsum("bhdn,bhen->bhde", k, v)  # [b*h, heads, c, c]

        # out = Q @ context : [b*h, heads, c, w] @ [b*h, heads, c, c] -> [b*h, heads, c, w]
        out = torch.einsum("bhde,bhdn->bhen", context, q)  # [b*h, heads, c, w]

        # 重排回原始形状
        out = rearrange(
            out, "(b h) heads c w -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        out = self.to_out(out)       # [b, hidden_dim, a, t] -> [b, feat_dim, a, t]
        out = rearrange(out, "b f a t -> b a f t")  # 恢复为 [b, a, f, t]

        if self.residual:
            out = y + self.gamma * out  # 可学习缩放残差连接
        return out


class TemporalSelfAttention(nn.Module):
    """
    ============ 时间自注意力层 (Temporal Self-Attention) ============
    在多智能体轨迹的 **智能体维度** 上做 Self-Attention (跨智能体注意力)

    核心功能:
        每个智能体生成 Q, 所有智能体之间互相计算注意力,
        让智能体之间能够相互通信和协调

    结构:
        - 独立 1x1 Conv 生成 Q, K, V
        - 加上时间嵌入偏置 (time-conditioned attention)
        - 多头注意力 + Softmax
        - 可选残差连接

    关键维度变化:
        输入:  (b*a, f, t)  →  展平 batch 和 agent 维度
        Q/K/V: (b*a, h*d, t)  其中 h=nheads, d=dim_per_head
        注意力: 在时间维度 t 上做 d*t 维度的匹配
        实际注意力维度: (h, b, a, d*t)  →  (h, b, a1, a2) 跨智能体
        → 这是 **跨智能体注意力** 而非跨时间步注意力!
    ===============================================================
    """
    def __init__(
        self,
        n_channels: int,         # 输入通道数 (特征维度)
        qk_n_channels: int,      # Q/K 每个头的通道数
        v_n_channels: int,       # V 每个头的通道数
        embed_dim: int,          # 时间嵌入维度
        nheads: int = 4,         # 注意力头数
        residual: bool = False,  # 是否使用残差连接
    ):
        super().__init__()
        self.nheads = nheads

        # 1x1 Conv 分别生成 Q, K, V (共享所有时间步的投影权重)
        # 输出通道 = 每头维度 × 头数
        self.query_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, v_n_channels * nheads, kernel_size=1)

        # 时间嵌入 → Q/K/V 的偏置 (条件调制注意力)
        self.query_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, qk_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),  # 增加 horizon 维度以便广播
        )
        self.key_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, qk_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )
        self.value_time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, v_n_channels * nheads),
            Rearrange("batch t -> batch t 1"),
        )

        self.attend = nn.Softmax(dim=-1)  # 在最后一个维度上做归一化
        self.residual = residual
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))  # 可学习残差缩放

    def forward(self, x, time):
        """
        x    : [batch, n_agents, feat_dim, horizon]   多智能体-时间特征
        time : [batch, n_agents, embed_dim]           时间步嵌入
        """
        # 展平 batch 和 agent 维度: [b, a, f, t] -> [(b*a), f, t]
        x_flat = rearrange(x, "b a f t -> (b a) f t")
        time = rearrange(time, "b a f -> (b a) f")

        # 分别生成 Q, K, V, 并加上时间条件偏置
        query, key, value = (
            self.query_layer(x_flat) + self.query_time_mlp(time),  # [(b*a), (h*d_qk), t]
            self.key_layer(x_flat) + self.key_time_mlp(time),     # [(b*a), (h*d_qk), t]
            self.value_layer(x_flat) + self.value_time_mlp(time), # [(b*a), (h*d_v), t]
        )

        # 重组为多头: 将通道维度拆分为 heads × dim_per_head
        # 并将时间步维度 t 展平到特征维度中 → (d * t) 作为一个整体特征
        # 这样注意力是在智能体之间计算的, 每个智能体将整个时间序列作为特征
        query = rearrange(
            query, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )  # [h, b, a, d_qk*t]
        key = rearrange(
            key, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )  # [h, b, a, d_qk*t]
        value = rearrange(
            value, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )  # [h, b, a, d_v*t]

        # === 跨智能体注意力计算 ===
        # dots = Q @ K^T / sqrt(d_k): [h, b, a1, d*t] @ [h, b, a2, d*t] -> [h, b, a1, a2]
        # 每个智能体 a1 关注所有其他智能体 a2
        dots = einsum(query, key, "h b a1 f, h b a2 f -> h b a1 a2") / math.sqrt(
            query.shape[-1]
        )
        attn = self.attend(dots)  # Softmax 归一化: [h, b, a1, a2]

        # 加权求和: attn @ V: [h, b, a1, a2] @ [h, b, a2, f] -> [h, b, a1, f]
        out = einsum(attn, value, "h b a1 a2, h b a2 f -> h b a1 f")  # [h, b, a, d_v*t]

        # 恢复原始形状: 合并多头
        out = rearrange(out, "h b a f -> b a (h f)")  # [b, a, h*d_v*t]
        out = out.reshape(x.shape)  # [b, a, f, t]

        if self.residual:
            out = x + self.gamma * out
        return out


class TemporalMlpBlock(nn.Module):
    """
    ============ 时间 MLP 块 (Temporal MLP Block) ============
    在时间维度 (horizon) 上做逐点的全连接变换, 同时融入时间步嵌入。

    结构:
        block0: Linear(dim_in -> dim_out) + act_fn, 并加上时间条件偏置
        block1: Linear(dim_out -> dim_out) + out_act_fn

    输入 x 形状: [batch_size, inp_channels, horizon]
    输出形状:     [batch_size, out_channels, horizon]
    每个 horizon 位置独立做线性变换, 类似逐时间步的 MLP。
    ==========================================================
    """
    def __init__(self, dim_in, dim_out, embed_dim, act_fn, out_act_fn):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),  # 逐时间步线性变换: dim_in -> dim_out
                    act_fn,
                ),
                nn.Sequential(
                    nn.Linear(dim_out, dim_out),  # 第二个线性层: dim_out -> dim_out
                    out_act_fn,
                ),
            ]
        )
        # 时间步嵌入 -> 维度匹配 dim_out, 作为偏置加到 block0 的输出上
        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, dim_out),  # embed_dim -> dim_out
        )

    def forward(self, x, t):
        """
        x : [batch_size, inp_channels, horizon]  特征图
        t : [batch_size, embed_dim]              时间步嵌入
        返回:
        out : [batch_size, out_channels, horizon]
        """
        # block0 输出 + 时间嵌入偏置 → 实现时间条件调制
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out


class ResidualTemporalBlock(nn.Module):
    """
    ============ 残差时间块 (Residual Temporal Block) ============
    这是 UNet 中最核心的基本构建单元, 用于在时间维度上做特征提取。

    结构 (类似 ResNet 瓶颈块):
        Conv1dBlock (Conv1d + GroupNorm + Mish)  ← 融入时间条件
        Conv1dBlock (Conv1d + GroupNorm + Mish)
        + 残差连接 (shortcut, 若通道数变化则用 1x1 Conv 对齐)

    时间条件注入:
        时间嵌入 t -> MLP -> [batch_size, out_channels, 1]
        加到第一个卷积块输出上 (通道级偏置调制)
    ==============================================================
    """
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5, mish=True):
        super().__init__()

        # 两个 Conv1dBlock 序列
        # block0: inp_channels -> out_channels (升/降维)
        # block1: out_channels -> out_channels (保持维度)
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
                Conv1dBlock(out_channels, out_channels, kernel_size, mish),
            ]
        )

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        # 时间嵌入 -> 通道数偏置 (加到第一个卷积块的输出上)
        # [batch, embed_dim] -> [batch, out_channels] -> [batch, out_channels, 1]
        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),           # embed_dim -> out_channels
            Rearrange("batch t -> batch t 1"),            # 增加时间维度便于广播相加
        )

        # 残差连接的 shortcut: 若通道数不同, 用 1x1 Conv 对齐
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [batch_size, inp_channels, horizon]  输入特征
        t : [batch_size, embed_dim]              时间步嵌入
        返回:
        out : [batch_size, out_channels, horizon]

        前向逻辑:
        ┌──────┐    ┌──────────────┐    ┌──────┐
        │  x   │───→│ Conv1dBlock 0 │───→│  +   │───→ Conv1dBlock 1 ───→ + ───→ out
        └──────┘    └──────┬───────┘    └──┬───┘                     ↑
                           │  time_mlp(t)  │                         │
                           └───────────────┘                  residual_conv(x)
        """
        # 第一个卷积输出 + 时间条件偏置 (特征调制)
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        # 残差连接 (恒等映射或 1x1 卷积对齐)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    """
    =========================  Temporal UNet (轨迹UNet)  =========================
    这是扩散模型中的 **噪声预测网络** (ε_θ)，基于 1D 时序 U-Net 架构

    输入:
        - 带噪轨迹 x:            [batch_size, horizon, transition_dim]
        - 扩散时间步 time:        [batch_size]  (标量)
        - 可选条件: returns / env_timestep
    输出:
        - 预测噪声 ε:            [batch_size, horizon, transition_dim]
    =========================================================================
    """

    agent_share_parameters = True  # 多智能体设置: 所有智能体共享此网络参数

    def __init__(
        self,
        horizon: int,
        transition_dim: int,             # 单步状态/动作维度 (如 obs_dim + act_dim)
        history_horizon: int = 0,
        dim: int = 128,
        dim_mults: Tuple[int] = (1, 2, 4, 8),  # 每层通道倍增因子
        returns_condition: bool = False,  # 是否以 returns-to-go 为条件
        env_ts_condition: bool = False,   # 是否以环境时间步为条件
        condition_dropout: float = 0.1,  # 条件 dropout 概率 (classifier-free guidance)
        kernel_size: int = 5,            # 卷积核大小
        max_path_length: int = 100,      # 最大轨迹长度 (用于 Embedding)
        n_agents: int = 1,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # 1) 构建编码-解码通道维度列表
        #    dims = [transition_dim, dim*1, dim*2, dim*4, dim*8]
        #    in_out = [(transition_dim, dim), (dim, 2*dim), ...]
        #    形状变化: transition_dim -> 128 -> 256 -> 512 -> 1024
        # ------------------------------------------------------------------
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        mish = True
        act_fn = nn.Mish()

        self.time_dim = dim      # 扩散时间步嵌入的维度 (128)
        self.returns_dim = dim   # returns 嵌入的维度

        # ------------------------------------------------------------------
        # 2) 扩散时间步嵌入 (Time Embedding)
        #    时间标量 t -> Sinusoidal 编码 (dim) -> MLP 映射到 dim
        #    t: [batch_size] -> [batch_size, dim]
        # ------------------------------------------------------------------
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),        # 正弦位置编码: 标量 -> dim 维向量
            nn.Linear(dim, dim * 4),      # 升维
            act_fn,
            nn.Linear(dim * 4, dim),      # 降回 dim
        )
        embed_dim = dim  # 当前嵌入总维度

        self.returns_condition = returns_condition
        self.env_ts_condition = env_ts_condition
        self.condition_dropout = condition_dropout
        self.history_horizon = history_horizon
        self.n_agents = n_agents

        # ------------------------------------------------------------------
        # 3) Returns-to-Go 条件嵌入 (可选)
        #    returns: [batch_size, horizon] -> MLP -> [batch_size, dim]
        #    用于 classifier-free guidance: training 时随机 mask 掉条件
        # ------------------------------------------------------------------
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),        # 每个时间步的 returns 标量 -> dim
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),  # 最终映射到 dim
            )
            # Bernoulli 采样 mask，概率 p=1-dropout 保留条件
            self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)
            embed_dim += dim  # 总嵌入维度累加

        # ------------------------------------------------------------------
        # 4) 环境时间步条件嵌入 (可选)
        #    env_timestep: [batch_size, horizon] 中的第 history_horizon 步
        #    用 Embedding 表将整数时间步映射到 dim 维向量
        # ------------------------------------------------------------------
        if self.env_ts_condition:
            self.env_ts_mlp = nn.Sequential(
                nn.Embedding(max_path_length + 1, dim),  # 可学习的整数索引 -> 向量
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            embed_dim += dim

        self.embed_dim = embed_dim  # 总的条件嵌入维度 (time + returns + env_ts)

        # ------------------------------------------------------------------
        # 5) 构建 Encoder-Decoder (U-Net 结构)
        #    - Encoder (downs):  逐步下采样, 通道数递增
        #    - Decoder (ups):    逐步上采样, 通道数递减
        #    - Mid (bottleneck): 最底层, 保持最高维度
        # ------------------------------------------------------------------
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        # ===================== Encoder (下采样路径) =====================
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in, dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),  # 残差块1: dim_in -> dim_out
                        ResidualTemporalBlock(
                            dim_out, dim_out,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),  # 残差块2: dim_out -> dim_out
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                        # 下采样 (步长2卷积, 时间维度减半), 最后一层不做下采样
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2  # 下采样后时间维度减半

        # ===================== Bottleneck (中间层) =====================
        mid_dim = dims[-1]  # 最大通道数, 如 1024
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            mish=mish,
        )

        # ===================== Decoder (上采样路径) =====================
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in,   # ★ 跳跃连接后通道加倍, 先降维
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in,
                            embed_dim=embed_dim,
                            kernel_size=kernel_size,
                            mish=mish,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                        # 上采样 (转置卷积, 时间维度加倍)
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2  # 上采样后时间维度加倍

        # ===================== 输出卷积层 =====================
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            # 最后过渡回 transition_dim
            nn.Conv1d(dim, transition_dim, 1),  # 1x1 卷积: dim -> transition_dim
        )

    def forward(
        self,
        x,                # 带噪轨迹: [batch_size, horizon, transition_dim]
        time,             # 扩散时间步: [batch_size]
        returns=None,     # returns-to-go 条件: [batch_size, horizon] (可选)
        env_timestep=None, # 环境时间步条件: [batch_size, horizon] (可选)
        attention_masks=None,
        use_dropout=True,  # 训练时是否使用条件 dropout
        force_dropout=False, # 是否强制将所有条件置零 (用于无条件推理)
    ):
        """
        --- 前向传播 (预测噪声 ε_θ) ---
        输入:
            x : [batch_size, horizon, transition_dim]  带噪轨迹
            time : [batch_size]                        扩散时间步 (标量)
            returns : [batch_size, horizon] (可选)      returns-to-go 条件
            env_timestep : [batch_size, horizon] (可选) 环境时间步条件
        输出:
            out : [batch_size, horizon, transition_dim] 预测噪声
        """

        # ========== Step 1: 调整维度 ==========
        # 原始形状: [b, t, f] -> Conv1D 所需: [b, f, t] (通道在前)
        # 其中 f = transition_dim (特征通道), t = horizon (时间维度)
        x = einops.rearrange(x, "b t f -> b f t")  # [b, f, t]

        # ========== Step 2: 构建条件嵌入 ==========
        # 扩散时间步嵌入: 标量 time -> [b, embed_dim]
        t = self.time_mlp(time)  # [b, dim]

        # ----- returns-to-go 条件 -----
        if self.returns_condition:
            assert returns is not None
            # returns: [b, horizon] -> MLP逐点映射 -> [b, horizon, dim]
            # 然后取均值? 实际上 returns_mlp 接收 [b*horizon, 1] -> ... 看代码
            # 这里 returns_mlp(nn.Linear(1, dim)) 接收 [..., 1],
            # 但 returns 形状 [b, horizon], 所以 unsqueeze(-1) -> [b, horizon, 1]
            returns_embed = self.returns_mlp(returns)  # [b, horizon, dim]
            if use_dropout:
                # 随机 mask 条件, 实现 classifier-free guidance 训练
                mask = self.mask_dist.sample(
                    sample_shape=(returns_embed.size(0), 1)
                ).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed  # 推理时无条件生成
            # 注意: returns_embed 维度是 [b, horizon, dim],
            # 而 t 是 [b, dim], 需要先处理维度不一致问题
            # ★ 这里代码有潜在维度问题, 取决于调用方是否做了展平处理
            t = torch.cat([t, returns_embed], dim=-1)

        # ----- 环境时间步条件 -----
        if self.env_ts_condition:
            assert env_timestep is not None
            env_timestep = env_timestep.to(dtype=torch.int64)
            # 取历史窗口后的那个时间步: [b, horizon] -> [b]
            env_timestep = env_timestep[:, self.history_horizon]
            # Embedding 表查询: 整数 -> [b, dim] -> MLP -> [b, dim]
            env_ts_embed = self.env_ts_mlp(env_timestep)  # [b, dim]
            t = torch.cat([t, env_ts_embed], dim=-1)  # [b, embed_dim]

        # ========== Step 3: UNet 前向 ==========
        h = []  # 跳跃连接 (skip-connections) 存储栈

        # ----- Encoder: 下采样 -----
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)     # 残差块1: [b, dim_in, t] -> [b, dim_out, t]
            x = resnet2(x, t)    # 残差块2: [b, dim_out, t] -> [b, dim_out, t]
            h.append(x)          # 保存跳跃连接 (供 decoder 使用)
            x = downsample(x)    # 下采样: 时间维度减半 [b, dim_out, t/2]

        # ----- Bottleneck: 中间层 -----
        x = self.mid_block1(x, t)  # [b, mid_dim, t] -> [b, mid_dim, t]
        x = self.mid_block2(x, t)  # [b, mid_dim, t] -> [b, mid_dim, t]

        # ----- Decoder: 上采样 -----
        for resnet, resnet2, upsample in self.ups:
            # 跳跃连接拼接: 从栈中 pop 出对应 encoder 层的输出
            x = torch.cat((x, h.pop()), dim=1)  # [b, dim_out*2, t] ← 通道拼接
            x = resnet(x, t)                     # [b, dim_out*2, t] -> [b, dim_in, t]
            x = resnet2(x, t)                    # [b, dim_in, t] -> [b, dim_in, t]
            x = upsample(x)                      # 上采样: 时间维度加倍 [b, dim_in, t*2]

        # ========== Step 4: 最终输出 ==========
        x = self.final_conv(x)   # [b, dim, t] -> [b, transition_dim, t]

        # 恢复为原始维度顺序: [b, f, t] -> [b, t, f]
        x = einops.rearrange(x, "b f t -> b t f")
        return x  # [batch_size, horizon, transition_dim]  ← 预测的噪声


class TemporalValue(nn.Module):
    """
    ============ 时间值函数网络 (Temporal Value Network) ============
    基于 UNet 编码器结构的轨迹值函数, 用于 Diffusion-QL / IQL 等算法中

    功能:
        输入带噪轨迹 x 和时间步 t, 输出标量值 V(x, t)

    结构:
        UNet Encoder (不含 Decoder) + 全局平均池化 + MLP Head

    与 TemporalUnet 的区别:
        - 只有编码器 (下采样), 没有解码器 (上采样)
        - 最终输出为标量 (值), 而非轨迹 (噪声)
        - 中间层的通道收缩更快 (mid_dim -> mid_dim/4 -> mid_dim/16)
    ===============================================================
    """
    agent_share_parameters = True

    def __init__(
        self,
        horizon,           # 轨迹长度 (时间步数)
        transition_dim,    # 单步状态/动作维度
        dim=32,            # 基础通道数 (比 UNet 小, 值函数通常更轻量)
        dim_mults=(1, 2, 4, 8),  # 通道倍增因子
        out_dim=1,         # 输出维度 (默认 1, 标量值)
    ):
        super().__init__()

        # 构建通道维度列表: [transition_dim, dim, 2*dim, 4*dim, 8*dim]
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        # 时间步嵌入: 标量 -> Sinusoidal -> MLP -> time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # ========== 编码器 (下采样) ==========
        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in, dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        ResidualTemporalBlock(
                            dim_out, dim_out,
                            kernel_size=5,
                            embed_dim=time_dim,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2  # 下采样后时间维度减半

        # ========== 瓶颈层 (快速收缩通道) ==========
        # 相比 UNet 的瓶颈, 这里快速降维: 1024 -> 256 -> 64
        mid_dim = dims[-1]   # 最大通道数
        mid_dim_2 = mid_dim // 4
        mid_dim_3 = mid_dim // 16

        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim
        )
        fc_dim = mid_dim_3 * max(horizon, 1)  # 展平后的特征维度

        # ========== 输出 Head ==========
        # 将空间特征展平 + 时间嵌入 → MLP → 标量值
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        """
        --- 前向传播 (轨迹值函数估计) ---
        输入:
            x : [batch_size, horizon, transition_dim]  带噪轨迹
            time : [batch_size]                        扩散时间步
        输出:
            out : [batch_size, out_dim]                标量值 (V值)
        """
        # 调整维度: [b, h, t] -> [b, t, h] (t=通道, h=时间)
        x = einops.rearrange(x, "b h t -> b t h")

        # 时间步嵌入
        t = self.time_mlp(time)  # [b, time_dim]

        # === 编码器下采样 ===
        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)     # [b, dim_in, t] -> [b, dim_out, t]
            x = resnet2(x, t)    # [b, dim_out, t] -> [b, dim_out, t]
            x = downsample(x)    # 时间维度减半

        # === 瓶颈层 (通道快速收缩) ===
        x = self.mid_block1(x, t)  # [b, mid_dim, t] -> [b, mid_dim/4, t]
        x = self.mid_block2(x, t)  # [b, mid_dim/4, t] -> [b, mid_dim/16, t]

        # === 全局池化 + MLP 输出 ===
        x = x.view(len(x), -1)  # 展平: [b, mid_dim/16 * t]
        # 将展平特征和时间嵌入拼接, 输入 MLP Head
        out = self.final_block(torch.cat([x, t], dim=-1))  # [b, out_dim]

        return out  # [batch_size, out_dim]  ← 轨迹值函数输出 (标量值)