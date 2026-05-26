"""
验证脚本：训练状态分解编码器并验证其拟合能力

本脚本独立于 DoF 整体算法，仅用于验证核心设计假设：
  —— 状态 o_i 可以被分解为 z_env (环境信息) + z_inter (交互信息),
     且分解后的隐向量能够通过 Critic 网络准确预测真实奖励 r。

使用方法：
    python -m diffuser.models.my_diffusion.verification

流程：
    1. 加载离线数据集 (obs.npy, rewards.npy)
    2. 训练 StateDecompositionEncoder + RewardCritic
    3. 评估奖励预测精度 (MSE, R², 相关性)
    4. 评估解耦质量 (环境一致性, 交互多样性, 正交性)
    5. 对比不同质量数据集 (Expert/Medium/Poor) 的表现差异
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# 直接导入模块（跳过 diffuser.models.__init__ 中因 helpers.py 缺失导致的依赖链问题）
# 本验证脚本独立于 DoF 主流程，仅验证编码器设计的有效性
import importlib.util as _iu

def _load_module(name: str, filepath: str):
    spec = _iu.spec_from_file_location(name, filepath)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_my_diffusion_dir = os.path.dirname(os.path.abspath(__file__))
_state_encoder = _load_module('state_encoder', os.path.join(_my_diffusion_dir, 'state_encoder.py'))
_reward_critic = _load_module('reward_critic', os.path.join(_my_diffusion_dir, 'reward_critic.py'))

StateDecompositionEncoder = _state_encoder.StateDecompositionEncoder
GlobalStatePredictor = _state_encoder.GlobalStatePredictor
MILowerBoundEstimator = _state_encoder.MILowerBoundEstimator
compute_mi_loss = _state_encoder.compute_mi_loss
compute_decomposition_losses = _state_encoder.compute_decomposition_losses
compute_temporal_contrastive_loss = _state_encoder.compute_temporal_contrastive_loss
RewardCritic = _reward_critic.RewardCritic


# ==============================================================================
# 数据集类
# ==============================================================================

class OfflineStateRewardDataset(Dataset):
    """
    离线 MARL 数据集，提供基于连续时间步的 (obs_t, obs_{t+1}, reward_t, state_t)。

    数据格式：
        obs:         [N, n_agents, obs_dim]
        rewards:     [N, n_agents]
        path_lengths:[n_episodes]
        states:      [N, state_dim] or None
    """

    def __init__(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        path_lengths: np.ndarray,
        states: np.ndarray = None,
        max_samples: Optional[int] = None,
    ):
        self.obs = torch.from_numpy(obs).float()
        self.rewards = torch.from_numpy(rewards).float()
        self.states = torch.from_numpy(states).float() if states is not None else None
        self.path_lengths = np.asarray(path_lengths, dtype=np.int64)
        self.pair_indices = self._build_pair_indices(self.path_lengths)
        if max_samples is not None and max_samples < len(self.pair_indices):
            sample_ids = np.random.choice(len(self.pair_indices), max_samples, replace=False)
            self.pair_indices = self.pair_indices[sample_ids]
        assert len(self.obs) == len(self.rewards), \
            f"obs ({len(self.obs)}) and rewards ({len(self.rewards)}) must match"

    def _build_pair_indices(self, path_lengths: np.ndarray):
        pair_indices = []
        offset = 0
        for path_length in path_lengths:
            if path_length < 2:
                offset += int(path_length)
                continue
            for t in range(int(path_length) - 1):
                pair_indices.append((offset + t, offset + t + 1))
            offset += int(path_length)
        return np.asarray(pair_indices, dtype=np.int64)

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        cur_idx, next_idx = self.pair_indices[idx]
        item = {
            'obs': self.obs[cur_idx],           # [n_agents, obs_dim]
            'obs_next': self.obs[next_idx],     # [n_agents, obs_dim]
            'reward': self.rewards[cur_idx],    # [n_agents]
        }
        if self.states is not None:
            item['state'] = self.states[cur_idx]  # [state_dim]
        return item


# ==============================================================================
# 训练器
# ==============================================================================

class DecompositionTrainer:
    """
    训练状态分解编码器 + 奖励 Critic。

    损失组成：
        1. L_reward:  奖励预测 MSE(主损失,保证隐藏层语义)
        2. L_env_recon: Predictor(z_env) 重构全局状态 S
        3. L_mi:       z_env 与 z_inter 的互信息最小化(解耦)
    """

    def __init__(
        self,
        encoder: StateDecompositionEncoder,
        critic: RewardCritic,
        predictor: Optional[GlobalStatePredictor],
        mi_estimator: MILowerBoundEstimator,
        device: str = 'cuda',
        lr: float = 1e-3,
        reward_loss_weight: float = 1.0,
        env_reconstruction_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        contrastive_temperature: float = 0.1,
        mi_weight: float = 0.1,
        grad_clip: float = 1.0,
    ):
        self.encoder = encoder.to(device)
        self.critic = critic.to(device)
        self.predictor = predictor.to(device) if predictor is not None else None
        self.mi_estimator = mi_estimator.to(device)
        self.device = device

        self.reward_loss_weight = reward_loss_weight
        self.env_reconstruction_weight = env_reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.mi_weight = mi_weight
        self.grad_clip = grad_clip
        self.use_temporal_contrastive = self.predictor is None

        # 联合优化所有网络
        all_params = (
            list(encoder.parameters())
            + list(critic.parameters())
            + list(mi_estimator.parameters())
        )
        if self.predictor is not None:
            all_params += list(self.predictor.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )

        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)

    def train_step(self, batch: dict) -> dict:
        self.encoder.train()
        self.critic.train()
        if self.predictor is not None:
            self.predictor.train()
        self.mi_estimator.train()

        obs = batch['obs'].to(self.device)            # [B, N, D]
        obs_next = batch['obs_next'].to(self.device)   # [B, N, D]
        reward_true = batch['reward'].to(self.device)  # [B, N]
        global_state = batch.get('state', None)
        if global_state is not None:
            global_state = global_state.to(self.device)  # [B, state_dim]

        # 1. 编码
        z_env_all, z_inter_all = self.encoder.encode_all_agents(obs)
        # z_env_all: [B, N, env_dim], z_inter_all: [B, N, inter_dim]
        z_env_next_all, _ = self.encoder.encode_all_agents(obs_next)  # [B, N, env_dim]

        B, N = z_env_all.shape[:2]
        z_env_flat = z_env_all.reshape(-1, self.encoder.env_dim)      # [B*N, env_dim]
        z_inter_flat = z_inter_all.reshape(-1, self.encoder.inter_dim) # [B*N, inter_dim]

        # 2. 奖励预测 (Value Loss)
        q1, q2 = self.critic(z_env_flat, z_inter_flat)
        reward_true_flat = reward_true.reshape(-1, 1)  # [B*N, 1]
        loss_q1 = F.mse_loss(q1, reward_true_flat)
        loss_reward = loss_q1 + (F.mse_loss(q2, reward_true_flat) if q2 is not None else 0.0)

        # 3. 解耦辅助损失
        loss_mi = compute_mi_loss(self.mi_estimator, z_env_all, z_inter_all)
        if self.use_temporal_contrastive:
            loss_env_align = compute_temporal_contrastive_loss(
                z_env_all, z_env_next_all, temperature=self.contrastive_temperature
            )
            total_aux_loss = self.contrastive_weight * loss_env_align + self.mi_weight * loss_mi
            aux_losses = {
                'loss_env_align': loss_env_align,
                'loss_temporal_nce': loss_env_align,
                'loss_mi': loss_mi,
                'total_aux_loss': total_aux_loss,
            }
        else:
            aux_losses = compute_decomposition_losses(
                self.predictor, self.mi_estimator,
                z_env_all, z_inter_all,
                global_state,
                env_reconstruction_weight=self.env_reconstruction_weight,
                mi_weight=self.mi_weight,
            )
            aux_losses = {
                **aux_losses,
                'loss_env_align': aux_losses['loss_env_recon'],
            }

        # 4. 组合
        total_loss = self.reward_loss_weight * loss_reward + aux_losses['total_aux_loss']

        # 5. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
            clip_params = (
                list(self.encoder.parameters())
                + list(self.critic.parameters())
                + list(self.mi_estimator.parameters())
            )
            if self.predictor is not None:
                clip_params += list(self.predictor.parameters())
            nn.utils.clip_grad_norm_(
                clip_params,
                max_norm=self.grad_clip,
            )
        self.optimizer.step()

        info = {
            'loss_total': total_loss.item(),
            'loss_reward': loss_reward.item(),
            'q1_mean': q1.mean().item(),
            'reward_true_mean': reward_true_flat.mean().item(),
        }
        info.update({k: v.item() for k, v in aux_losses.items()})
        return info

    @torch.no_grad()
    def eval_step(self, batch: dict) -> dict:
        self.encoder.eval()
        self.critic.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.mi_estimator.eval()

        obs = batch['obs'].to(self.device)
        obs_next = batch['obs_next'].to(self.device)
        reward_true = batch['reward'].to(self.device)
        global_state = batch.get('state', None)
        if global_state is not None:
            global_state = global_state.to(self.device)

        z_env_all, z_inter_all = self.encoder.encode_all_agents(obs)
        z_env_next_all, _ = self.encoder.encode_all_agents(obs_next)
        z_env_flat = z_env_all.reshape(-1, self.encoder.env_dim)
        z_inter_flat = z_inter_all.reshape(-1, self.encoder.inter_dim)
        reward_flat = reward_true.reshape(-1, 1)

        r_pred = self.critic.predict(z_env_flat, z_inter_flat)
        mse = F.mse_loss(r_pred, reward_flat).item()
        mae = F.l1_loss(r_pred, reward_flat).item()

        ss_res = ((reward_flat - r_pred) ** 2).sum()
        ss_tot = ((reward_flat - reward_flat.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        r_pred_centered = r_pred - r_pred.mean()
        r_true_centered = reward_flat - reward_flat.mean()
        pearson = (r_pred_centered * r_true_centered).sum() / (
            torch.sqrt((r_pred_centered ** 2).sum()) *
            torch.sqrt((r_true_centered ** 2).sum()) + 1e-8
        )

        loss_mi = compute_mi_loss(self.mi_estimator, z_env_all, z_inter_all)
        if self.use_temporal_contrastive:
            loss_env_align = compute_temporal_contrastive_loss(
                z_env_all, z_env_next_all, temperature=self.contrastive_temperature
            )
            aux_losses = {
                'loss_env_align': loss_env_align,
                'loss_temporal_nce': loss_env_align,
                'loss_mi': loss_mi,
                'total_aux_loss': self.contrastive_weight * loss_env_align + loss_mi,
                'loss_env_recon': torch.tensor(0.0, device=self.device),
            }
        else:
            aux_losses = compute_decomposition_losses(
                self.predictor, self.mi_estimator,
                z_env_all, z_inter_all,
                global_state,
                env_reconstruction_weight=1.0, mi_weight=1.0,
            )
            aux_losses = {
                **aux_losses,
                'loss_env_align': aux_losses['loss_env_recon'],
            }

        return {
            'mse': mse, 'mae': mae, 'r2': r2.item(), 'pearson': pearson.item(),
            'reward_pred_mean': r_pred.mean().item(),
            'reward_true_mean': reward_flat.mean().item(),
            'loss_env_recon': aux_losses.get('loss_env_recon', torch.tensor(0.0)).item(),
            'loss_env_align': aux_losses.get('loss_env_align', torch.tensor(0.0)).item(),
            'loss_temporal_nce': aux_losses.get('loss_temporal_nce', torch.tensor(0.0)).item(),
            'loss_mi': aux_losses.get('loss_mi', 0.).item(),
        }

    def train(self, train_loader, val_loader, num_epochs=50, log_interval=5):
        for epoch in range(num_epochs):
            epoch_info = defaultdict(list)
            for batch in train_loader:
                info = self.train_step(batch)
                for k, v in info.items():
                    epoch_info[k].append(v)

            epoch_avg = {k: np.mean(v) for k, v in epoch_info.items()}
            for k, v in epoch_avg.items():
                self.train_losses[k].append(v)

            self.scheduler.step()

            if (epoch + 1) % log_interval == 0:
                val_info = defaultdict(list)
                for batch in val_loader:
                    info = self.eval_step(batch)
                    for k, v in info.items():
                        val_info[k].append(v)
                val_avg = {k: np.mean(v) for k, v in val_info.items()}
                for k, v in val_avg.items():
                    self.val_losses[k].append(v)

                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch+1:4d}/{num_epochs} | "
                    f"Loss: {epoch_avg['loss_total']:.4f} | "
                    f"Reward MSE: {epoch_avg['loss_reward']:.4f} | "
                    f"Val R²: {val_avg['r2']:.4f} | "
                    f"Val Pearson: {val_avg['pearson']:.4f} | "
                    f"Val MAE: {val_avg['mae']:.4f} | "
                    f"LR: {lr:.6f}"
                )
                print(
                    f"          | "
                    f"Env_Align: {val_avg['loss_env_align']:.6f} | "
                    f"MI: {val_avg['loss_mi']:.6f}"
                )

        return self.train_losses, self.val_losses


# ==============================================================================
# 数据加载工具
# ==============================================================================

def load_offline_data(data_dir: str) -> tuple:
    """
    加载离线数据集
    
    Args:
        data_dir: 包含 obs.npy, rewards.npy 的目录
        
    Returns:
        obs:         [N, n_agents, obs_dim]
        rewards:     [N, n_agents]
        path_lengths:[n_episodes]
        states:      [N, state_dim] or None
    """
    obs_path = os.path.join(data_dir, 'obs.npy')
    rewards_path = os.path.join(data_dir, 'rewards.npy')
    path_lengths_path = os.path.join(data_dir, 'path_lengths.npy')
    state_path = os.path.join(data_dir, 'states.npy')
    
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"obs.npy not found at {obs_path}")
    if not os.path.exists(rewards_path):
        raise FileNotFoundError(f"rewards.npy not found at {rewards_path}")
    
    obs = np.load(obs_path)
    rewards = np.load(rewards_path)
    path_lengths = np.load(path_lengths_path) if os.path.exists(path_lengths_path) else np.array([len(obs)])
    states = np.load(state_path) if os.path.exists(state_path) else None
    
    print(f"Loaded data from {data_dir}:")
    print(f"  obs shape:     {obs.shape}")
    print(f"  rewards shape: {rewards.shape}")
    print(f"  path_lengths:   {path_lengths.shape}")
    if states is not None:
        print(f"  states shape:   {states.shape}")
    
    return obs, rewards, path_lengths, states


def create_dataloaders(
    obs: np.ndarray,
    rewards: np.ndarray,
    path_lengths: np.ndarray,
    states: np.ndarray = None,
    batch_size: int = 512,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    max_samples: int = None,
):
    """创建训练/验证 DataLoader"""
    dataset = OfflineStateRewardDataset(
        obs, rewards, path_lengths, states=states, max_samples=max_samples
    )
    
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def _parse_group_specs(group_specs):
    groups = []
    for spec in group_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid group spec: {spec}. Use NAME=dir1,dir2")
        name, paths = spec.split("=", 1)
        dirs = [p for p in paths.split(",") if p]
        if not dirs:
            raise ValueError(f"Group {name} has no data dirs")
        groups.append((name, dirs))
    return groups


def _build_groups(args):
    if args.group_specs:
        return _parse_group_specs(args.group_specs)
    return [(os.path.basename(p.rstrip("/")), [p]) for p in args.data_dirs]


def _load_and_concat_data(data_dirs):
    all_obs, all_rewards, all_states, all_path_lengths = [], [], [], []
    has_any_state = False
    has_all_state = True
    for data_dir in data_dirs:
        obs, rewards, path_lengths, states = load_offline_data(data_dir)
        all_obs.append(obs)
        all_rewards.append(rewards)
        all_path_lengths.append(path_lengths)

        if states is not None:
            all_states.append(states)
            has_any_state = True
        else:
            all_states.append(None)
            has_all_state = False

    obs = np.concatenate(all_obs, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    path_lengths = np.concatenate(all_path_lengths, axis=0)
    states = None
    if has_all_state:
        states = np.concatenate(all_states, axis=0)
    elif has_any_state:
        # 混合：部分有 states, 部分没有 → 丢弃 states
        print("  [WARNING] Mixed dataset: some dirs have states.npy, some don't. "
              "Discarding global state for consistency.")
    return obs, rewards, path_lengths, states


def _print_group_comparison(metrics_by_group):
    group_names = list(metrics_by_group.keys())
    if len(group_names) < 2:
        return
    metric_keys = [
        "mse",
        "mae",
        "r2",
        "pearson",
        "loss_env_align",
        "loss_mi",
    ]

    print(f"\n{'='*60}")
    print("Group Comparison (A - B)")
    print(f"{'='*60}")
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name_a = group_names[i]
            name_b = group_names[j]
            print(f"\n{name_a} - {name_b}:")
            for k in metric_keys:
                if k in metrics_by_group[name_a] and k in metrics_by_group[name_b]:
                    delta = metrics_by_group[name_a][k] - metrics_by_group[name_b][k]
                    print(f"  {k}: {delta:.6f}")


def _run_single_group(data_dirs, args, device, group_name):
    print(f"\n{'='*60}")
    print(f"Dataset Group: {group_name}")
    print(f"{'='*60}\n")

    obs, rewards, path_lengths, states = _load_and_concat_data(data_dirs)
    n_agents = obs.shape[1]
    obs_dim = obs.shape[2]
    has_state = states is not None
    state_dim = states.shape[-1] if has_state else None

    print(
        f"\nCombined dataset: {len(obs)} transitions, {n_agents} agents, obs_dim={obs_dim}"
    )
    if has_state:
        print(f"  states shape:  {states.shape}")
    else:
        print(f"  [INFO] states.npy not available — using temporal contrastive alignment")
    print(
        f"Reward statistics: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
        f"min={rewards.min():.4f}, max={rewards.max():.4f}"
    )

    encoder = StateDecompositionEncoder(
        obs_dim=obs_dim,
        env_dim=args.env_dim,
        inter_dim=args.inter_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    resolved_env_dim = encoder.env_dim

    critic = RewardCritic(
        env_dim=resolved_env_dim,
        inter_dim=args.inter_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        double_q=args.double_q,
    )

    # 只有在提供全局状态时才构建 Predictor
    predictor = None
    if has_state:
        predictor = GlobalStatePredictor(
            env_dim=resolved_env_dim,
            state_dim=state_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        predictor = None

    mi_estimator = MILowerBoundEstimator(
        env_dim=resolved_env_dim,
        inter_dim=args.inter_dim,
        hidden_dim=args.hidden_dim,
    )

    print(f"\nEncoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
    print(f"Critic:  {sum(p.numel() for p in critic.parameters()):,} parameters")
    if has_state:
        print(f"Predictor: {sum(p.numel() for p in predictor.parameters()):,} parameters")
    else:
        print(f"Predictor: skipped (no states.npy)")
    print(f"MI Estimator: {sum(p.numel() for p in mi_estimator.parameters()):,} parameters")
    print(
        f"  o_i={obs_dim} -> z_env={resolved_env_dim}, z_inter={args.inter_dim}, hidden_dim={args.hidden_dim}"
    )

    train_loader, val_loader = create_dataloaders(
        obs,
        rewards,
        path_lengths,
        states=states,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
    )

    trainer = DecompositionTrainer(
        encoder=encoder,
        critic=critic,
        predictor=predictor,
        mi_estimator=mi_estimator,
        device=device,
        lr=args.lr,
        reward_loss_weight=args.reward_loss_weight,
        env_reconstruction_weight=args.env_reconstruction_weight if has_state else 0.0,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        mi_weight=args.mi_weight,
    )

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs...")
    print(f"{'='*60}\n")

    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
    )

    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")

    val_info_agg = defaultdict(list)
    for batch in val_loader:
        info = trainer.eval_step(batch)
        for k, v in info.items():
            val_info_agg[k].append(v)

    final_metrics = {k: np.mean(v) for k, v in val_info_agg.items()}

    print("\n  Reward Prediction:")
    print(f"    MSE:     {final_metrics['mse']:.6f}")
    print(f"    MAE:     {final_metrics['mae']:.6f}")
    print(f"    R^2:     {final_metrics['r2']:.6f}")
    print(f"    Pearson: {final_metrics['pearson']:.6f}")

    print("\n  Disentanglement Quality:")
    print(
        f"    Env Alignment:       {final_metrics['loss_env_align']:.6f}  (down is better)"
    )
    print(
        f"    MI:                  {final_metrics['loss_mi']:.6f}  (down is better)"
    )

    return final_metrics


# ==============================================================================
# 主函数
# ==============================================================================

def run_verification(args):
    """
    运行验证实验
    
    验证指标：
        1. 奖励预测 MSE / R² / Pearson 相关系数
        2. 环境一致性 (z_env 在 agent 间的一致性)
        3. 交互多样性 (z_inter 在 agent 间的差异)
        4. 正交性 (z_env 与 z_inter 的独立性)
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'  # 为了验证过程的稳定性和可复现性,暂时使用CPU
    print(f"\n{'='*60}")
    print(f"State Decomposition Encoder - Verification")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    if args.compare_groups:
        groups = _build_groups(args)
        metrics_by_group = {}
        for name, data_dirs in groups:
            metrics_by_group[name] = _run_single_group(
                data_dirs, args, device, name
            )
        _print_group_comparison(metrics_by_group)
        return metrics_by_group

    final_metrics = _run_single_group(args.data_dirs, args, device, "Combined")
    return final_metrics


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="State Decomposition Encoder - Verification"
    )
    
    # 数据
    parser.add_argument(
        '--data_dirs', type=str, nargs='+',
        default=[
            'diffuser/datasets/data/smac/3m/Good',
            'diffuser/datasets/data/smac/3m/Medium',
            'diffuser/datasets/data/smac/3m/Poor',
        ],
        help='Directories containing obs.npy and rewards.npy'
    )
    parser.add_argument(
        '--compare_groups', action='store_true', default=False,
        help='Train and evaluate each dataset group separately and compare metrics'
    )
    parser.add_argument(
        '--group_specs', type=str, nargs='*', default=None,
        help='Group spec like NAME=dir1,dir2. If omitted, each entry in --data_dirs is a group.'
    )
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Max samples to use (for faster training)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    
    # 模型
    parser.add_argument('--env_dim', type=int, default=None,
                        help='Dimension of z_env (None=auto: same as obs_dim for plugging into diffusion)')
    parser.add_argument('--inter_dim', type=int, default=16,
                        help='Dimension of z_inter (low-dim interaction signal)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of MLPs')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of MLP layers')
    parser.add_argument('--double_q', action='store_true', default=True,
                        help='Use double Q-network')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='Logging interval (epochs)')
    
    # 损失权重
    parser.add_argument('--reward_loss_weight', type=float, default=1.0,
                        help='Weight for reward prediction loss')
    parser.add_argument('--env_reconstruction_weight', type=float, default=1.0,
                        help='Weight for env reconstruction loss (Predictor(z_env) -> S)')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='Weight for temporal contrastive alignment loss when states are unavailable')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help='Temperature for temporal InfoNCE alignment')
    parser.add_argument('--mi_weight', type=float, default=0.1,
                        help='Weight for mutual information minimization (z_env <-> z_inter)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_verification(args)
