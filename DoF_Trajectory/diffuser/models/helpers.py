import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

import diffuser.utils as utils


class SinusoidalPosEmb(nn.Module): # 位置编码
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module): # 下采样
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1) # 3x3卷积, 2步长, 1填充

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module): # 上采样
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1) # 4x4转置卷积, 2步长, 1填充

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module): # 卷积块
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()
        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            act_fn,
        ) # 卷积层, 归一化层, 激活函数

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module): # 卷积自注意力
    def __init__(
        self,
        n_channels: int,
        qk_n_channels: int, # 查询和键的通道数
        v_n_channels: int, # 值的通道数
        nheads: int = 4, # 头数
        residual: bool = False, # 是否使用残差连接
        use_state: bool = False, # 是否使用全局状态信息进行注意力计算
    ):
        super().__init__()
        self.nheads = nheads
        self.query_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.key_layer = nn.Conv1d(n_channels, qk_n_channels * nheads, kernel_size=1)
        self.value_layer = nn.Conv1d(n_channels, v_n_channels * nheads, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.residual = residual
        self.use_state = use_state
        if use_state:
            self.state_query_layer = nn.Conv1d(n_channels, qk_n_channels, kernel_size=1)
            self.state_key_layer = nn.Conv1d(n_channels, qk_n_channels, kernel_size=1)
            self.state_value_layer = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        if residual:
            self.gamma = nn.Parameter(torch.zeros([1]))

    def forward(self, x, states: torch.Tensor = None):
        x_flat = rearrange(x, "b a f t -> (b a) f t")
        query, key, value = (
            self.query_layer(x_flat),
            self.key_layer(x_flat),
            self.value_layer(x_flat),
        )

        query = rearrange(
            query, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )
        key = rearrange(
            key, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )
        value = rearrange(
            value, "(b a) (h d) t -> h b a (d t)", h=self.nheads, a=x.shape[1]
        )

        if self.use_state:
            assert states is not None  
            state_query, state_key, state_value = (
                self.state_query_layer(states),
                self.state_key_layer(states),
                self.state_value_layer(states),
            )
            state_query = rearrange(
                state_query, "b (h d) t -> h b 1 (d t)", h=self.nheads
            )
            state_key = rearrange(state_key, "b (h d) t -> h b 1 (d t)", h=self.nheads)
            state_value = rearrange(
                state_value, "b (h d) t -> h b 1 (d t)", h=self.nheads
            )
            query = torch.cat((query, state_query), dim=2)
            key = torch.cat((key, state_key), dim=2)
            value = torch.cat((value, state_value), dim=2)

        dots = einsum(query, key, "h b a1 f, h b a2 f -> h b a1 a2") / math.sqrt(
            query.shape[-1]
        )
        attn = self.attend(dots)
        out = einsum(attn, value, "h b a1 a2, h b a2 f -> h b a1 f")

        out = rearrange(out, "h b a f -> b a (h f)")
        out = out.reshape(x.shape)
        if self.residual:
            out = x + self.gamma * out
        return out


class PositionalEncoding(nn.Module): # 位置编码
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout: float = 0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class MlpSelfAttention(nn.Module): # MLP自注意力
    def __init__(self, dim_in, dim_hidden=128):
        super().__init__()
        self.query_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.key_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_in),
        )

    def forward(self, x):
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)
        query, key, value = (
            self.query_layer(x_flat),
            self.key_layer(x_flat),
            self.value_layer(x_flat),
        )
        query = query.reshape(x.shape[0], x.shape[1], -1)
        key = key.reshape(x.shape[0], x.shape[1], -1)
        value = value.reshape(x.shape[0], x.shape[1], -1)

        beta = F.softmax(
            torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1
        )
        output = torch.bmm(beta, value).reshape(x.shape)
        return output


def extract(a, t, x_shape): # 从张量中提取元素
    b, *_ = t.shape
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32): # 余弦调度
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32): # 线性调度
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

def vp_beta_schedule(timesteps, dtype=torch.float32): # VP调度
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

def apply_conditioning(x, conditions, action_dim): # 应用条件
    
    apply_basic_cond = False # 是否应用基本条件
    for t, val in conditions.items(): # 遍历条件字典，t可以是字符串（特殊条件）或整数/元组、列表（基本条件）
        if isinstance(t, str): # 字符串
            if t == "player_idxs": # 玩家索引
                assert apply_basic_cond
                if x.shape[-1] < 4:  
                    x = torch.cat([val, x], dim=-1)
                else:
                    x[:, :, :, 0] = val
            elif t == "player_hoop_sides": # 篮筐方向
                assert apply_basic_cond
                if x.shape[-1] < 4:  
                    x = torch.cat([x, val], dim=-1)
                else:
                    x[:, :, :, -1] = val
            else:
                continue

        elif isinstance(t, int): # 基本条件 - 整数索引 表示某一时间步的条件
            x[:, t, :, action_dim:] = val.clone() # 复制条件值到对应位置   [B, 1, N, O]
            apply_basic_cond = True
        elif isinstance(t, tuple) or isinstance(t, list): # 基本条件 - 元组或列表索引 表示一个时间步范围的条件
            assert len(t) == 2, t # 确保元组或列表长度为2 t[0]是起始时间步，t[1]是结束时间步
            cond_value = val.clone()
            if "agent_idx" in conditions:
                x[:, t[0] : t[1] - 1, :, action_dim:] = cond_value[:, :-1] # [B, T_range-1, N, O]
                index = (
                    conditions["agent_idx"][0] 
                    .long()
                    .repeat(1, 1, x.shape[-1] - action_dim)
                )
                x[:, t[1] - 1].scatter_(1, index, cond_value[:, -1].gather(1, index)) # [B, 1, N, O] 最后一时间步的条件值根据agent_idx索引进行散布
            else:
                x[:, t[0] : t[1], :, action_dim:] = cond_value  # [B, T_range, N, O]
                
            apply_basic_cond = True
        else:
            raise TypeError(type(t))
    return x

class WeightedLoss(nn.Module): # 加权损失

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class WeightedStateLoss(nn.Module): # 加权状态损失(针对状态预测的加权损失)

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights) # 注册权重为缓冲区，确保在模型加载时保持不变

    def forward(self, pred, targ):
        loss = self._loss(pred, targ)
        weighted_loss = loss * self.weights
        info = {"a0_loss": weighted_loss.mean()} # 计算平均损失
        return weighted_loss, info

class WeightedL1(WeightedLoss): # 加权L1损失

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss): # 加权L2损失

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none') # reduction='none'返回每个元素的损失值，而不是平均或求和后的结果

class WeightedStateL2(WeightedStateLoss): # 加权状态L2损失

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'state_l2': WeightedStateL2,
}


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model): # 更新模型移动平均值
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new): # 更新移动平均值
        if old is None:
            return new
        return self.beta * old + (1 - self.beta) * new
