import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffuser.utils as utils

class WeightedLoss(nn.Module): # 处理动作状态的损失基类,用于对不同轨迹的损失进行加权
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer("weights", weights) # 权重矩阵,用于对不同轨迹的损失进行加权
        self.action_dim = action_dim

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        # weighted_loss = (loss * self.weights).mean()
        if self.action_dim > 0:
            a0_loss = (
                loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]
            ).mean() # 计算第一个时间步的损失,并除以权重矩阵的第一个元素
            info = {"a0_loss": a0_loss} # 记录第一个时间步的损失,用于评估
        else:
            info = {}
        return loss * self.weights, info
        # return weighted_loss, {"a0_loss": a0_loss}


class WeightedStateLoss(nn.Module): # 带权重的状态损失基类,用于对状态预测进行加权
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean() # 对损失进行加权,并取平均值
        return loss * self.weights, {"a0_loss": weighted_loss}
        # return weighted_loss, {"a0_loss": weighted_loss}


class ValueLoss(nn.Module): # 价值函数损失基类,用于对奖励预测进行加权
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(), utils.to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = { # 记录奖励预测的统计信息,用于评估
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": utils.to_torch(corr, device=pred.device),
        }

        return loss, info


class WeightedL1(WeightedLoss): # 带权重的L1损失
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss): # 带权重的L2损失
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class WeightedStateL2(WeightedStateLoss): # 带权重的状态L2损失
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


class ValueL1(ValueLoss): # L1奖励损失
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss): # L2奖励损失
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
    "state_l2": WeightedStateL2,
    "value_l1": ValueL1,
    "value_l2": ValueL2,
}


def apply_conditioning(x, conditions): # 应用条件,对输入进行进行条件化
    cond_masks = conditions["masks"].to(bool) # 获取条件掩码,用于标识哪些位置需要应用条件
    x[cond_masks] = conditions["x"][cond_masks].clone() # 对需要应用条件的位置,使用条件值进行替换

    if "player_idxs" in conditions.keys():
        if x.shape[-1] < 4:  # pure position information w.o. player info
            x = torch.cat([conditions["player_idxs"], x], dim=-1)
            x = torch.cat([x, conditions["player_hoop_sides"]], dim=-1)
        else:
            x[:, :, :, 0] = conditions["player_idxs"]
            x[:, :, :, -1] = conditions["player_hoop_sides"]

    return x
