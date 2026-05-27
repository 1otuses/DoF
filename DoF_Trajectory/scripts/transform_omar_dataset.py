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

def main(env_name: str="mpe", map_name: str="simple_spread", quality: str="expert"):
    add_agent_id_to_obs = True
    dataset_dir = Path(f"diffuser/datasets/data/{env_name}/{map_name}/{quality}")
    # 原始数据集格式为  env_name/map_name/quality/seed_x_data/(acs_x.npy, obs_x.npy, rews_x.npy, dones_x.npy, next_obs_x.npy)
    # 需要转换为 tfrecord 格式，格式如下：
    # 每条记录包含以下字段：agent_{i}_observations, agent_{i}_actions, agent_{i}_rewards, agent_{i}_next_observations, agent_{i}_dones, agent_{i}_discounts, agent_{i}_legal_actions, env_state

    file_path = Path(dataset_dir)
    print(file_path)
    sub_dir_to_idx = {}
    idx = 0
    for subdir in os.listdir(file_path):
        if file_path.joinpath(subdir).is_dir():
            sub_dir_to_idx[subdir] = idx
            idx += 1
    
    
    for subdir in os.listdir(file_path):
        subdir_path = file_path.joinpath(subdir)
        if not subdir_path.is_dir():
            continue
        filenames = [str(file_name) for file_name in subdir_path.glob("*.npy")]
        # filenames = sorted(filenames, key=get_fname_idx)
        filenames = sorted(filenames)
        print(filenames)

        (
            all_observations,
            all_actions,
            all_rewards,
            all_discounts,
            all_logprobs,
            all_dones,
            all_next_observations
        ) = ([], [], [], [], [], [], [])

        for filename in filenames:
            data = np.load(filename)
            if "acs" in filename:
                all_actions.append(data) # 存入所有agent的actions
            elif "dones" in filename:
                all_dones.append(data)
            elif "next_obs" in filename:
                all_next_observations.append(data)
            elif "obs" in filename:
                all_observations.append(data)
            elif "rews" in filename:
                all_rewards.append(data)
        
        num_agents = len(all_actions) # 获取agent数量
        assert len(all_dones) == num_agents
        assert len(all_next_observations) == num_agents
        assert len(all_observations) == num_agents
        assert len(all_rewards) == num_agents

        length = all_actions[0].shape[0] # 获取每个agent的actions长度
        assert all_dones[0].shape[0] == length
        assert all_next_observations[0].shape[0] == length
        assert all_observations[0].shape[0] == length
        assert all_rewards[0].shape[0] == length

        def save_to_tfrecord(all_observations, all_actions, all_rewards, all_discounts, all_logprobs, all_dones, all_next_observations, output_file):
            for l in range(length):
                shard_idx = l // 200
                shard_path = output_file + str(shard_idx) + ".tfrecord"
                with tf.io.TFRecordWriter(shard_path, options=tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
                    feature = {}
                    # 每 200 时间步生成一条记录
                    for a in range(num_agents):
                        obs = all_observations[a][l]
                        act = all_actions[a][l]
                        rew = all_rewards[a][l]
                        next_obs = all_next_observations[a][l]
                        done = all_dones[a][l]

                        feature[f'agent_{a}_observations'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(obs)).numpy()]))
                        feature[f'agent_{a}_actions'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(act)).numpy()]))
                        feature[f'agent_{a}_rewards'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(rew)).numpy()]))
                        feature[f'agent_{a}_next_observations'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(next_obs)).numpy()]))
                        feature[f'agent_{a}_dones'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(done)).numpy()]))
                        feature[f'agent_{a}_discounts'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.float32(0.99))).numpy()]))
                        feature[f'agent_{a}_legal_actions'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.ones(5, "float32"))).numpy()]))

                    feature[f'env_state'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.zeros(54, "float32"))).numpy()]))
                    feature[f'episode_return'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.zeros(1, "float32"))).numpy()]))
                    feature[f'zero_padding_mask'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(np.array(1, dtype=np.float32))).numpy()]))

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                if l % 200 == 0:
                    print(f"Saved {l} records to {shard_path}")

        prefix = f"tests/datasets/{map_name}/{quality}/{subdir}/"

        if not os.path.exists(prefix):
            os.makedirs(prefix)
            print(f"Directory '{prefix}' created.")
        else:
            print(f"Directory '{prefix}' already exists.")

        output_file = prefix + f"{map_name}_"
        # output_file = prefix + f"{map_name}.tfrecord"

        save_to_tfrecord(all_observations, all_actions, all_rewards, 
                         all_discounts, all_logprobs, all_dones, all_next_observations, 
                         output_file)
        
        # break


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="mpe")
    parser.add_argument("--map_name", type=str, default="simple_spread")
    parser.add_argument("--quality", type=str, default="expert")
    args = parser.parse_args()

    main(args.env_name, args.map_name, args.quality)