"""
VAE Phase 2: 数据预编码脚本

将离线数据集中的所有观测 o_i 预编码为 z_env + z_inter,
并保存为新的 .npy 文件供 Phase 2 DoF 训练使用。

使用方式:
    python -m diffuser.models.my_diffusion.vae_preprocess \
        --vae_ckpt outputs/vae_phase1/vae_phase1.pth \
        --data_dir diffuser/datasets/data/mpe/simple_spread/Medium \
        --output_dir diffuser/datasets/data/mpe/simple_spread/Medium_vae
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

# 动态加载 VAE 模块
import importlib.util as _iu
_my_diffusion_dir = os.path.dirname(os.path.abspath(__file__))


def _load_mod(name: str, filepath: str):
    spec = _iu.spec_from_file_location(name, filepath)
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vae_mod = _load_mod('vae', os.path.join(_my_diffusion_dir, 'vae.py'))
ObservationVAE = _vae_mod.ObservationVAE


# ==============================================================================
# 预编码管道
# ==============================================================================

def pre_encode_dataset(
    vae: ObservationVAE,
    data_dir: str,
    output_dir: str,
    batch_size: int = 256,
    device: str = 'cuda',
    save_z_env: bool = True,
    save_z_inter: bool = True,
    save_reconstructed: bool = False,
) -> dict:
    """
    对数据集中的所有 obs 执行预编码。

    输入:data_dir/obs.npy
    输出:output_dir/z_env.npy, z_inter.npy (可选: obs_recon.npy)

    Args:
        vae:         已训练的 ObservationVAE (冻结)
        data_dir:    包含 obs.npy 的目录
        output_dir:  输出目录
        batch_size:  批处理大小
        device:      计算设备

    Returns:
        dict: 编码统计信息
    """
    obs_path = os.path.join(data_dir, 'obs.npy')
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"obs.npy not found at {obs_path}")

    obs = np.load(obs_path)  # [N, n_agents, obs_dim]
    print(f"Loaded obs: {obs.shape} ({obs.dtype})")

    vae.eval()
    vae.to(device)
    for p in vae.parameters():
        p.requires_grad = False

    N = obs.shape[0]
    obs_dim = obs.shape[-1]
    n_agents = obs.shape[1]

    z_env_all = np.zeros((N, n_agents, vae.env_dim), dtype=np.float32)
    z_inter_all = np.zeros((N, n_agents, vae.inter_dim), dtype=np.float32)
    obs_recon_all = np.zeros_like(obs) if save_reconstructed else None

    # 批编码
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            obs_batch = torch.from_numpy(obs[start:end]).float().to(device)

            z_env, z_inter, o_hat = vae(obs_batch.reshape(-1, vae.obs_dim))
            # 恢复形状
            z_env = z_env.reshape(end - start, n_agents, vae.env_dim)
            z_inter = z_inter.reshape(end - start, n_agents, vae.inter_dim)
            o_hat = o_hat.reshape(end - start, n_agents, obs_dim)

            z_env_all[start:end] = z_env.cpu().numpy()
            z_inter_all[start:end] = z_inter.cpu().numpy()
            if save_reconstructed:
                obs_recon_all[start:end] = o_hat.cpu().numpy()

            if (end) % (batch_size * 10) == 0 or end == N:
                print(f"  Encoded {end}/{N} samples")

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'z_env.npy'), z_env_all)
    np.save(os.path.join(output_dir, 'z_inter.npy'), z_inter_all)

    # 复制原始 action & rewards（不涉及 VAE）
    for fname in ['actions.npy', 'rewards.npy', 'terminals.npy', 'states.npy', 'path_lengths.npy']:
        src = os.path.join(data_dir, fname)
        if os.path.exists(src):
            import shutil
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)

    # 保存原始 obs 的副本
    np.save(os.path.join(output_dir, 'obs_original.npy'), obs)

    if save_reconstructed:
        np.save(os.path.join(output_dir, 'obs_reconstructed.npy'), obs_recon_all)

    # 计算重构质量
    recon_mse = np.mean((obs_recon_all - obs) ** 2)
    recon_mae = np.mean(np.abs(obs_recon_all - obs))

    stats = {
        'n_samples': N,
        'obs_dim': obs_dim,
        'n_agents': n_agents,
        'env_dim': vae.env_dim,
        'inter_dim': vae.inter_dim,
        'recon_mse': recon_mse,
        'recon_mae': recon_mae,
    }

    print(f"\nPre-encoding complete!")
    print(f"  Output: {output_dir}")
    print(f"  z_env shape:    {z_env_all.shape}")
    print(f"  z_inter shape:  {z_inter_all.shape}")
    print(f"  Recon MSE:      {recon_mse:.6f}")
    print(f"  Recon MAE:      {recon_mae:.6f}")

    return stats


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Pre-encode Dataset for Phase 2")
    parser.add_argument('--vae_ckpt', type=str, required=True,
                        help='Path to saved VAE checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing obs.npy')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir + _vae)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--save_reconstructed', action='store_true',
                        help='Also save reconstructed obs')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    # 加载 VAE
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    hp = ckpt['hyper_params']
    vae = ObservationVAE(
        obs_dim=hp['obs_dim'],
        env_dim=hp['env_dim'],
        inter_dim=hp['inter_dim'],
        hidden_dim=hp['hidden_dim'],
        num_layers=hp['num_layers'],
    )
    vae.load_state_dict(ckpt['vae_state_dict'])
    print(f"Loaded VAE from {args.vae_ckpt}")
    print(f"  {vae.summary}")
    if 'final_metrics' in ckpt:
        print(f"  Saved metrics: {ckpt['final_metrics']}")

    output_dir = args.output_dir or (args.data_dir.rstrip('/') + '_vae')

    stats = pre_encode_dataset(
        vae=vae,
        data_dir=args.data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        device=device,
        save_reconstructed=args.save_reconstructed,
    )

    # 保存统计信息
    import json
    with open(os.path.join(output_dir, 'pre_encoding_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print("\nDone!")


if __name__ == '__main__':
    main()
