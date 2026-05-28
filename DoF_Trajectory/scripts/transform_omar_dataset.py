import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party/og-marl"))

from og_marl.environments import smac, mamujoco
from og_marl.environments import simple_spread, simple_adversary


def save_to_tfrecord(
    all_observations, all_actions, all_rewards,
    all_discounts, all_logprobs, all_dones, all_next_observations,
    output_file, num_agents, length, desc="",
):
    """一次性写入所有记录，每条 shard 只打开一次。

    修复前：每条记录都打开 shard 文件（覆盖写入 → 只保留最后一条）
    修复后：每个 shard 打开一次，连续写满 200 条再关闭。
    """
    num_shards = (length + 199) // 200
    pbar = tqdm(total=length, desc=desc, unit="rec", ncols=80, leave=False)

    for shard_idx in range(num_shards):
        shard_start = shard_idx * 200
        shard_end = min(shard_start + 200, length)
        shard_path = f"{output_file}{shard_idx}.tfrecord"

        with tf.io.TFRecordWriter(
            shard_path,
            options=tf.io.TFRecordOptions(compression_type="GZIP"),
        ) as writer:
            for l in range(shard_start, shard_end):
                feature = {}
                for a in range(num_agents):
                    obs = all_observations[a][l]
                    act = all_actions[a][l]
                    rew = all_rewards[a][l]
                    next_obs = all_next_observations[a][l]
                    done = all_dones[a][l]

                    feature[f'agent_{a}_observations'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(obs)).numpy()]))
                    feature[f'agent_{a}_actions'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(act)).numpy()]))
                    feature[f'agent_{a}_rewards'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(rew)).numpy()]))
                    feature[f'agent_{a}_next_observations'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(next_obs)).numpy()]))
                    feature[f'agent_{a}_dones'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(done)).numpy()]))
                    feature[f'agent_{a}_discounts'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.float32(0.99))).numpy()]))
                    feature[f'agent_{a}_legal_actions'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.ones(5, "float32"))).numpy()]))

                feature['env_state'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.zeros(54, "float32"))).numpy()]))
                feature['episode_return'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.zeros(1, "float32"))).numpy()]))
                feature['zero_padding_mask'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.array(1, dtype=np.float32))).numpy()]))

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                pbar.update(1)

    pbar.close()


def process_seed(
    seed_subdir, dataset_dir, map_name, quality, env_name,
):
    """处理单个 seed 目录，写为 TFRecord shards。"""
    subdir_path = dataset_dir / seed_subdir
    filenames = sorted([str(f) for f in subdir_path.glob("*.npy")])

    all_observations, all_actions, all_rewards = [], [], []
    all_discounts, all_logprobs, all_dones, all_next_observations = [], [], [], []

    for filename in filenames:
        data = np.load(filename)
        name = os.path.basename(filename)
        if "acs" in name:
            all_actions.append(data)
        elif "dones" in name:
            all_dones.append(data)
        elif "next_obs" in name:
            all_next_observations.append(data)
        elif "obs" in name and "next" not in name:
            all_observations.append(data)
        elif "rews" in name:
            all_rewards.append(data)

    num_agents = len(all_actions)
    assert len(all_dones) == num_agents and len(all_next_observations) == num_agents
    assert len(all_observations) == num_agents and len(all_rewards) == num_agents

    length = all_actions[0].shape[0]
    for arr_list in (all_dones, all_next_observations, all_observations, all_rewards):
        assert arr_list[0].shape[0] == length

    prefix = f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/{seed_subdir}/"
    os.makedirs(prefix, exist_ok=True)
    output_file = prefix + f"{map_name}_"

    save_to_tfrecord(
        all_observations, all_actions, all_rewards,
        all_discounts, all_logprobs, all_dones, all_next_observations,
        output_file, num_agents, length,
        desc=f"{quality}/{seed_subdir}",
    )


def main(env_name="mpe", map_name="simple_spread", quality="expert", max_workers=4):
    dataset_dir = Path(f"diffuser/datasets/data/{env_name}/{map_name}/{quality}")
    print(f"Dataset: {dataset_dir}")

    seed_dirs = sorted([
        d.name for d in dataset_dir.iterdir()
        if d.is_dir()
    ])
    print(f"Seeds found: {seed_dirs}")

    if max_workers == 1:
        # 单进程（调试用）
        for seed_subdir in seed_dirs:
            process_seed(seed_subdir, dataset_dir, map_name, quality, env_name)
    else:
        # 多进程并行处理各 seed
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_seed, seed_subdir, dataset_dir, map_name, quality, env_name
                ): seed_subdir
                for seed_subdir in seed_dirs
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc=f"Seeds ({map_name}/{quality})", ncols=80,
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Seed {futures[future]} failed: {e}")

    print(f"Done: {map_name}/{quality}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="mpe")
    parser.add_argument("--map_name", type=str, default="simple_spread")
    parser.add_argument("--quality", type=str, default="expert")
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Number of parallel processes. 1 = single-process (debug).",
    )
    args = parser.parse_args()

    main(args.env_name, args.map_name, args.quality, args.workers)