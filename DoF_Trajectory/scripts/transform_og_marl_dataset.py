import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party/og-marl"))

from og_marl.environments import smac, mamujoco
from og_marl.environments import simple_spread, simple_adversary


def detect_compression_type(file_path: str) -> str:
    with open(file_path, "rb") as file_handle:
        magic = file_handle.read(2)
    if magic == b"\x1f\x8b":
        return "GZIP"
    if magic in (b"\x78\x01", b"\x78\x9c", b"\x78\xda"):
        return "ZLIB"
    return ""


def main(env_name: str, map_name: str, quality: str, compression: str):
    # 将tfrecord数据转换为numpy格式数据
    # 是否添加agent_id到obs
    add_agent_id_to_obs = True
    dataset_dir = PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality

    # 处理mpe环境的特殊数据格式
    if env_name == "mpe" :
        if map_name == "simple_adversary":
            dataset_dir = PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality
        elif map_name == "simple_spread":
            dataset_dir = PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality

    file_path = Path(dataset_dir)
    # 为每个子目录分配一个索引
    sub_dir_to_idx = {}
    idx = 0
    for subdir in os.listdir(file_path):
        if file_path.joinpath(subdir).is_dir():
            sub_dir_to_idx[subdir] = idx
            idx += 1

    def get_fname_idx(file_name):
        # 定义一个函数，根据文件名获取索引，用于排序
        # print(file_name)
        # print("\n")
        # print(sub_dir_to_idx)
        if env_name == "smac":
            # SMAC的数据格式是: .../{subdir}/log_{number}.tfrecord
            dir_idx = sub_dir_to_idx[file_name.split("/")[-2]] * 1000
            return dir_idx + int(file_name.split("log_")[-1].split(".")[0])
        elif env_name == "mamujoco":
            # print(file_name.split("/")[-3])
            dir_idx = sub_dir_to_idx[file_name.split("/")[-2]] * 1000
            return dir_idx + int(file_name.split("/")[-1].split("log_")[-1].split(".")[0]) 
        elif env_name == "mpe":
            return file_name
        else:
            raise ValueError(f"Unknown environment {env_name}")

    # 读取所有tfrecord文件，并按照索引排序
    filenames = [str(file_name) for file_name in file_path.glob("**/*.tfrecord")]
    filenames = sorted(filenames, key=get_fname_idx)
    if not filenames:
        raise ValueError(f"No .tfrecord files found under: {file_path}")

    # 初始化环境
    if env_name == "smac":
        env = smac.SMAC(map_name)
    elif env_name == "mamujoco":
        env = mamujoco.Mujoco(map_name)
    elif env_name == "mpe":
        if map_name == "simple_adversary":
            env = simple_adversary.SimpleAdversary()
        elif map_name == "simple_spread":
            env = simple_spread.SimpleSpread()
    
    else:
        raise ValueError(f"Unknown environment {env_name}")
    agents = env.agents # 获取所有智能体

    compression_map = {
        "gzip": "GZIP",
        "zlib": "ZLIB",
        "none": "",
    }
    compression = compression.lower()
    if compression == "auto":
        gzip_files, zlib_files, raw_files = [], [], []
        for file_name in filenames:
            file_compression = detect_compression_type(file_name)
            if file_compression == "GZIP":
                gzip_files.append(file_name)
            elif file_compression == "ZLIB":
                zlib_files.append(file_name)
            else:
                raw_files.append(file_name)

        datasets = []
        if gzip_files:
            datasets.append(
                tf.data.Dataset.from_tensor_slices(gzip_files).flat_map(
                    lambda x: tf.data.TFRecordDataset(
                        x, compression_type="GZIP"
                    ).map(env._decode_fn)
                )
            )
        if zlib_files:
            datasets.append(
                tf.data.Dataset.from_tensor_slices(zlib_files).flat_map(
                    lambda x: tf.data.TFRecordDataset(
                        x, compression_type="ZLIB"
                    ).map(env._decode_fn)
                )
            )
        if raw_files:
            datasets.append(
                tf.data.Dataset.from_tensor_slices(raw_files).flat_map(
                    lambda x: tf.data.TFRecordDataset(x, compression_type="").map(
                        env._decode_fn
                    )
                )
            )
        if not datasets:
            raise ValueError(f"No readable tfrecord files found under: {file_path}")
        raw_dataset = datasets[0]
        for extra_dataset in datasets[1:]:
            raw_dataset = raw_dataset.concatenate(extra_dataset)
    else:
        if compression not in compression_map:
            raise ValueError(
                "Unsupported compression. Use auto, gzip, zlib, or none."
            )
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        # 使用flat_map将每个文件的TFRecordDataset展平为一个整体的数据流
        raw_dataset = filename_dataset.flat_map(
            lambda x: tf.data.TFRecordDataset(
                x, compression_type=compression_map[compression]
            ).map(env._decode_fn)
        )

    period = 10 # 处理数据的时间步长

    # Split the dataset into multiple batches
    batch_size = 1024 # 每个batch的大小
    databatches = raw_dataset.batch(batch_size)

    (
        all_observations, # 所有智能体的观察
        all_actions, # 所有智能体的动作
        all_rewards, # 所有智能体的奖励
        all_discounts, # 所有智能体的折扣因子
        all_logprobs, # 所有智能体的动作概率（如果有的话）
    ) = ([], [], [], [], [])
    if env_name == "smac": # SMAC环境需要处理状态和合法动作
        all_states, all_legals = [], [] # 所有智能体的状态和合法动作
    all_path_lengths = [] # 所有路径的长度

    path_length = 0 # 当前路径的长度，用于累计每个智能体的观察时间步长
    (
        path_observations,
        path_actions,
        path_rewards,
        path_logprobs,
        path_discounts,
    ) = ([], [], [], [], [])
    if env_name == "smac":
        path_states, path_legals = [], []
    for databatch in tqdm(databatches): # 遍历每个batch的数据
        extras = databatch.extras # 获取额外的信息
        zero_padding_mask_batch = extras["zero_padding_mask"].numpy()
        if zero_padding_mask_batch.ndim == 1:
            # step-level records store one mask value per sample
            zero_padding_mask_batch = zero_padding_mask_batch[:, None]
        batch_size = zero_padding_mask_batch.shape[0] # 获取当前batch的大小
        if env_name == "smac":
            states = extras["s_t"] # SMAC环境的状态信息

        observations, actions, rewards, discounts, logprobs = ([], [], [], [], [])
        if env_name == "smac":
            legals = []
        for agent in agents: # 遍历每个智能体
            observations.append(databatch.observations[agent].observation.numpy())
            if env_name == "smac":
                legals.append(databatch.observations[agent].legal_actions.numpy())
            actions.append(databatch.actions[agent].numpy())
            rewards.append(databatch.rewards[agent].numpy())
            discounts.append(databatch.discounts[agent].numpy())
            if "logprobs" in extras: # 如果有动作概率
                logprobs.append(extras["logprobs"][agent].numpy())

        if observations[0].ndim == 2:
            # step-level data: [B, F] -> [B, 1, N, F]
            observations = np.stack(observations, axis=1)[:, None, :, :]
            if env_name == "smac":
                legals = np.stack(legals, axis=1)[:, None, :, :]
            actions = np.stack(actions, axis=1)[:, None, :, :]
            rewards = np.stack(rewards, axis=1)[:, None, :]
            discounts = np.stack(discounts, axis=1)[:, None, :]
        else:
            # sequence-level data: [B, T, F] -> [B, T, N, F]
            observations = np.stack(observations, axis=2)
            if env_name == "smac":
                legals = np.stack(legals, axis=2)
            actions = np.stack(actions, axis=2)
            rewards = np.stack(rewards, axis=-1)
            discounts = np.stack(discounts, axis=-1)
        if "logprobs" in extras:
            if logprobs[0].ndim == 2:
                logprobs = np.stack(logprobs, axis=1)[:, None, :, :]
            else:
                logprobs = np.stack(logprobs, axis=2)

        for idx in range(batch_size):
            zero_padding_mask = zero_padding_mask_batch[idx][:period]
            path_length += np.sum(zero_padding_mask, dtype=int)

            if env_name == "smac":
                path_states.append(states[idx, :period])
                path_legals.append(legals[idx, :period])
            path_observations.append(observations[idx, :period])
            path_actions.append(actions[idx, :period])
            path_rewards.append(rewards[idx, :period])
            path_discounts.append(discounts[idx, :period])
            if "logprobs" in extras:
                path_logprobs.append(logprobs[idx, :period])

            if (
                int(path_discounts[-1][-1, 0]) == 0
                or path_length >= env.max_episode_length
            ):
                path_observations = np.concatenate(path_observations, axis=0)
                if add_agent_id_to_obs:
                    T, N = path_observations.shape[:2]
                    agent_ids = []
                    for i in range(N):
                        agent_id = tf.one_hot(i, depth=N)
                        agent_ids.append(agent_id)
                    agent_ids = tf.stack(agent_ids, axis=0)

                    # Repeat along time dim
                    agent_ids = tf.stack([agent_ids] * T, axis=0)
                    agent_ids = agent_ids.numpy()

                    path_observations = np.concatenate(
                        [agent_ids, path_observations], axis=-1
                    )

                if env_name == "smac":
                    all_states.append(np.concatenate(path_states, axis=0)[:path_length])
                    all_legals.append(np.concatenate(path_legals, axis=0)[:path_length])
                all_observations.append(path_observations[:path_length])
                all_actions.append(np.concatenate(path_actions, axis=0)[:path_length])
                all_rewards.append(np.concatenate(path_rewards, axis=0)[:path_length])
                all_discounts.append(
                    np.concatenate(path_discounts, axis=0)[:path_length]
                )
                if "logprobs" in extras:
                    all_logprobs.append(
                        np.concatenate(path_logprobs, axis=0)[:path_length]
                    )
                all_path_lengths.append(path_length)

                (
                    path_observations,
                    path_actions,
                    path_rewards,
                    path_discounts,
                    path_logprobs,
                ) = ([], [], [], [], [])
                if env_name == "smac":
                    path_states, path_legals = [], []
                path_length = 0

    """ Concatenate Episodes """
    if env_name == "smac":
        all_states = np.concatenate(all_states, axis=0)
        all_legals = np.concatenate(all_legals, axis=0)
    all_observations = np.concatenate(all_observations, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_discounts = np.concatenate(all_discounts, axis=0)
    if "logprobs" in extras:
        all_logprobs = np.concatenate(all_logprobs, axis=0)
    all_path_lengths = np.array(all_path_lengths)

    """ Save Numpy Arrays """
    if env_name == "smac":
        np.save(
            PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"states.npy",
            all_states,
        )
        np.save(
            PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"legals.npy",
            all_legals,
        )
    np.save(
        PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"obs.npy",
        all_observations,
    )
    np.save(
        PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"actions.npy",
        all_actions,
    )
    np.save(
        PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"rewards.npy",
        all_rewards,
    )
    np.save(
        PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"discounts.npy",
        all_discounts,
    )
    if "logprobs" in extras:
        np.save(
            PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"logprobs.npy",
            all_logprobs,
        )
    np.save(
        PROJECT_ROOT/"diffuser"/"datasets"/"data"/env_name/map_name/quality/"path_lengths.npy",
        all_path_lengths,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="mpe")
    parser.add_argument("--map_name", type=str, default="simple_spread")
    parser.add_argument("--quality", type=str, default="medium")
    parser.add_argument(
        "--compression",
        type=str,
        default="auto",
        choices=["auto", "gzip", "zlib", "none"],
    )
    args = parser.parse_args()

    main(args.env_name, args.map_name, args.quality, args.compression)