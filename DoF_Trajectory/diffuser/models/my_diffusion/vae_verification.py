"""
VAE Phase 1 独立训练脚本

验证 VAE 的分解能力：
    1. 重构质量 (L_rec → 0)
    2. 环境表征时序一致性 (L_NCE → 0)
    3. z_env ⊥ z_inter 解耦程度 (L_decouple → 0)

使用方式：
    python -m diffuser.models.my_diffusion.vae_verification \
        --data_dirs diffuser/datasets/data/mpe/simple_spread/Medium \
        --epochs 50 --batch_size 64
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
from tqdm import tqdm

# 动态加载本项目模块 (绕过 diffuser.models.__init__ 依赖链)
import importlib.util as _iu
_my_diffusion_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name: str, filepath: str):
    spec = _iu.spec_from_file_location(name, filepath)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vae_mod = _load_mod('vae', os.path.join(_my_diffusion_dir, 'vae.py'))

ObservationVAE = _vae_mod.ObservationVAE
VAEEncoder = _vae_mod.VAEEncoder
VAEDecoder = _vae_mod.VAEDecoder
compute_phase1_total_loss = _vae_mod.compute_phase1_total_loss
compute_reconstruction_loss = _vae_mod.compute_reconstruction_loss
compute_temporal_infonce_loss = _vae_mod.compute_temporal_infonce_loss
compute_decouple_loss = _vae_mod.compute_decouple_loss
freeze_vae = _vae_mod.freeze_vae


# ==============================================================================
# 数据集
# ==============================================================================

class VAEPhase1Dataset(Dataset):
    """
    Phase 1 训练数据集。

    提供连续时间步的 (obs_t, obs_{t+1}) 对，
    用于训练 VAE 的时序对比 (L_NCE) 和重构 (L_rec)。

    数据格式：
        obs:          [N, n_agents, obs_dim]
        path_lengths: [n_episodes]
    """

    def __init__(
        self,
        obs: np.ndarray,
        path_lengths: np.ndarray,
        max_samples: Optional[int] = None,
    ):
        self.obs = torch.from_numpy(obs).float()
        self.path_lengths = np.asarray(path_lengths, dtype=np.int64)
        self.pair_indices = self._build_pair_indices()
        if max_samples is not None and max_samples < len(self.pair_indices):
            idx = np.random.choice(len(self.pair_indices), max_samples, replace=False)
            self.pair_indices = self.pair_indices[idx]

    def _build_pair_indices(self):
        pairs = []
        offset = 0
        for pl in self.path_lengths:
            pl = int(pl)
            if pl < 2:
                offset += pl
                continue
            for t in range(pl - 1):
                pairs.append((offset + t, offset + t + 1))
            offset += pl
        return np.asarray(pairs, dtype=np.int64)

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        cur, nxt = self.pair_indices[idx]
        return {
            'obs_t': self.obs[cur],       # [n_agents, obs_dim]
            'obs_next': self.obs[nxt],    # [n_agents, obs_dim]
        }


# ==============================================================================
# Phase 1 训练器
# ==============================================================================

class VAEPhase1Trainer:
    """训练 VAE 的 Phase 1。"""

    def __init__(
        self,
        vae: ObservationVAE,
        device: str = 'cuda',
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        contrastive_temperature: float = 0.1,
        grad_clip: float = 1.0,
        weight_decay: float = 1e-5,
    ):
        self.vae = vae.to(device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.contrastive_temperature = contrastive_temperature
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(
            vae.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )

        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)

    def train_step(self, batch: dict) -> dict:
        self.vae.train()

        obs_t = batch['obs_t'].to(self.device)        # [B, N, D]
        obs_next = batch['obs_next'].to(self.device)   # [B, N, D]

        z_env_t, z_inter_t = self.vae.encode_all_agents(obs_t)        # [B,N,envD], [B,N,interD]
        z_env_next, _ = self.vae.encode_all_agents(obs_next)           # [B,N,envD]

        losses = compute_phase1_total_loss(
            decoder=self.vae.decoder,
            z_env_t=z_env_t,
            z_inter_t=z_inter_t,
            z_env_next=z_env_next,
            obs_t=obs_t,
            alpha=self.alpha,
            beta=self.beta,
            contrastive_temperature=self.contrastive_temperature,
        )

        self.optimizer.zero_grad()
        losses['loss_total'].backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_clip)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def eval_step(self, batch: dict) -> dict:
        self.vae.eval()

        obs_t = batch['obs_t'].to(self.device)
        obs_next = batch['obs_next'].to(self.device)

        z_env_t, z_inter_t = self.vae.encode_all_agents(obs_t)
        z_env_next, _ = self.vae.encode_all_agents(obs_next)

        losses = compute_phase1_total_loss(
            decoder=self.vae.decoder,
            z_env_t=z_env_t,
            z_inter_t=z_inter_t,
            z_env_next=z_env_next,
            obs_t=obs_t,
            alpha=self.alpha,
            beta=self.beta,
            contrastive_temperature=self.contrastive_temperature,
        )
        info = {k: v.item() for k, v in losses.items()}

        # 附加质量指标
        o_hat = self.vae.decode_all_agents(z_env_t, z_inter_t)
        mae = F.l1_loss(o_hat, obs_t).item()
        # 余弦相似度 (z_env vs z_inter) → 越接近 0 越好
        z_env_n = F.normalize(z_env_t.reshape(-1, self.vae.env_dim), dim=-1)
        z_inter_n = F.normalize(z_inter_t.reshape(-1, self.vae.inter_dim), dim=-1)
        # 处理 env_dim ≠ inter_dim 的情况：投影到较小维度
        D_env, D_inter = z_env_n.shape[-1], z_inter_n.shape[-1]
        if D_env != D_inter:
            D_min = min(D_env, D_inter)
            z_env_n = z_env_n[..., :D_min]
            z_inter_n = z_inter_n[..., :D_min]
        orthogonality = (z_env_n * z_inter_n).sum(dim=-1).abs().mean().item()
        info['recon_mae'] = mae
        info['orthogonality'] = orthogonality

        return info

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        log_interval: int = 5,
    ):
        for epoch in range(num_epochs):
            epoch_info = defaultdict(list)

            # --- epoch 训练进度条 ---
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for batch in pbar:
                info = self.train_step(batch)
                for k, v in info.items():
                    epoch_info[k].append(v)
                # 实时显示当前 batch 的损失
                pbar.set_postfix({
                    'L_rec': f"{info['loss_rec']:.4f}",
                    'L_nce': f"{info['loss_nce']:.4f}",
                    'L_dec': f"{info['loss_decouple']:.4f}",
                })

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
                    f"L_rec={val_avg['loss_rec']:.6f} | "
                    f"L_NCE={val_avg['loss_nce']:.6f} | "
                    f"L_decouple={val_avg['loss_decouple']:.6f} | "
                    f"MAE={val_avg['recon_mae']:.6f} | "
                    f"Orth={val_avg['orthogonality']:.6f} | "
                    f"LR={lr:.6f}"
                )

        return self.train_losses, self.val_losses


# ==============================================================================
# 数据加载
# ==============================================================================

def load_obs_data(data_dir: str):
    """加载 obs.npy 和 path_lengths.npy"""
    obs_path = os.path.join(data_dir, 'obs.npy')
    pl_path = os.path.join(data_dir, 'path_lengths.npy')

    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"obs.npy not found at {obs_path}")

    obs = np.load(obs_path)
    path_lengths = (
        np.load(pl_path)
        if os.path.exists(pl_path)
        else np.array([len(obs)])
    )

    print(f"Loaded: obs={obs.shape}, path_lengths={path_lengths.shape}")
    return obs, path_lengths


def create_dataloaders(
    obs: np.ndarray,
    path_lengths: np.ndarray,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    max_samples: int = None,
):
    dataset = VAEPhase1Dataset(obs, path_lengths, max_samples=max_samples)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


# ==============================================================================
# 主函数
# ==============================================================================

def run_phase1_training(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\n{'='*60}")
    print(f"VAE Phase 1 Training")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    all_obs, all_pl = [], []
    for data_dir in args.data_dirs:
        obs, pl = load_obs_data(data_dir)
        all_obs.append(obs)
        all_pl.append(pl)
    obs = np.concatenate(all_obs, axis=0)
    path_lengths = np.concatenate(all_pl, axis=0)

    n_agents = obs.shape[1]
    obs_dim = obs.shape[2]
    print(f"Dataset: {len(obs)} transitions, {n_agents} agents, obs_dim={obs_dim}")

    # 维度约束
    env_dim = args.env_dim if args.env_dim else max(4, obs_dim // 3)
    inter_dim = args.inter_dim if args.inter_dim else max(obs_dim, obs_dim - env_dim + 2)

    vae = ObservationVAE(
        obs_dim=obs_dim,
        env_dim=env_dim,
        inter_dim=inter_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        dropout=args.dropout,
    )
    print(f"\n{vae.summary}")
    print(f"Total params: {sum(p.numel() for p in vae.parameters()):,}")

    train_loader, val_loader = create_dataloaders(
        obs, path_lengths,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
    )

    trainer = VAEPhase1Trainer(
        vae=vae,
        device=device,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        contrastive_temperature=args.contrastive_temperature,
        grad_clip=args.grad_clip,
    )

    print(f"\n{'='*60}")
    print(f"Training {args.epochs} epochs...")
    print(f"  α (NCE weight)={args.alpha}, β (decouple weight)={args.beta}")
    print(f"{'='*60}\n")

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
    )

    # 最终评估
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    val_agg = defaultdict(list)
    for batch in val_loader:
        info = trainer.eval_step(batch)
        for k, v in info.items():
            val_agg[k].append(v)
    final = {k: np.mean(v) for k, v in val_agg.items()}
    print(f"  L_rec:       {final['loss_rec']:.6f}")
    print(f"  L_NCE:       {final['loss_nce']:.6f}")
    print(f"  L_decouple:  {final['loss_decouple']:.6f}")
    print(f"  Recon MAE:   {final['recon_mae']:.6f}")
    print(f"  Orthogonality:{final['orthogonality']:.6f}")

    # 保存模型
    if args.save_model:
        save_dir = args.save_dir or "outputs/vae_phase1"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "vae_phase1.pth")
        torch.save(
            {
                'vae_state_dict': vae.state_dict(),
                'hyper_params': {
                    'obs_dim': obs_dim,
                    'env_dim': vae.env_dim,
                    'inter_dim': vae.inter_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                },
                'final_metrics': final,
            },
            save_path,
        )
        print(f"\nSaved VAE to {save_path}")

    return vae, final


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Phase 1 Training")
    # Data
    parser.add_argument(
        '--data_dirs', type=str, nargs='+', required=True,
        help='Directories containing obs.npy and path_lengths.npy',
    )
    parser.add_argument('--max_samples', type=int, default=None)
    # Architecture
    parser.add_argument('--env_dim', type=int, default=None)
    parser.add_argument('--inter_dim', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='mish')
    parser.add_argument('--dropout', type=float, default=0.1)
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0, help='L_NCE weight')
    parser.add_argument('--beta', type=float, default=1.0, help='L_decouple weight')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1)
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=5)
    # Misc
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_dir', type=str, default='diffuser/models/my_diffusion/outputs/vae_phase1')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vae, metrics = run_phase1_training(args)
