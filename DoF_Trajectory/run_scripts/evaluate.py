import argparse
import glob
import json
import os
import time

import numpy as np
import diffuser.utils as utils
import yaml
from diffuser.utils.launcher_util import build_config_from_dict
import time
from torch.utils.tensorboard import SummaryWriter


def evaluate(Config):
    # Overall flow:
    # 1) build evaluator once (lazy-init)
    # 2) for each checkpoint step, run evaluation
    # 3) wait for result json, then log to TensorBoard
    evaluator = None
    Config.condition_guidance_w = getattr(Config, "condition_guidance_w", None)

    tb_writer = SummaryWriter(os.path.join(Config.log_dir, "tensorboard"))

    for load_step in Config.load_steps:
        # 1) locate checkpoint
        ckpt_file_path = os.path.join(
            Config.log_dir, f"checkpoint/state_{load_step}.pt"
        )
        if not os.path.exists(ckpt_file_path):
            print(f"Checkpoint file {ckpt_file_path} not found. Skipping evaluation.")
            continue

        # 2) resolve results file path (optionally with DDIM + guidance tags)
        results_file_path = os.path.join(
            Config.log_dir,
            f"results/step_{load_step}-ep_{Config.num_eval}-ddim.json"
            if getattr(Config, "use_ddim_sample", False)
            else f"results/step_{load_step}-ep_{Config.num_eval}.json",
        )
        if Config.condition_guidance_w is not None:
            results_file_path = results_file_path.replace(
                ".json", f"-cg_{Config.condition_guidance_w}.json"
            )
        if not Config.overwrite and os.path.exists(results_file_path):
            print(
                f"Results file {results_file_path} already exist. Skipping evaluation."
            )
            # still log existing results to TensorBoard
            _log_results_to_tb(tb_writer, results_file_path)
            continue

        # 3) init evaluator once with config
        if evaluator is None:
            evaluator_config = utils.Config(Config.evaluator, verbose=True)
            evaluator = evaluator_config()
            evaluator.init(
                log_dir=Config.log_dir,
                num_eval=Config.num_eval,
                num_envs=getattr(Config, "num_envs", Config.num_eval),
                condition_guidance_w=Config.condition_guidance_w,
                use_ddim_sample=Config.use_ddim_sample,
                n_ddim_steps=Config.n_ddim_steps,
                use_consistency_models_sample=Config.use_consistency_models_sample,
                n_consistency_models_steps=Config.n_consistency_models_steps,
            )

        # 4) run evaluation for this checkpoint step
        evaluator.evaluate(load_step=load_step)

        # 5) wait for results to be written, then log to TensorBoard
        _wait_for_results(results_file_path)
        _log_results_to_tb(tb_writer, results_file_path)

    tb_writer.close()


def _wait_for_results(path, timeout=300, poll_interval=2):
    """Wait up to `timeout` seconds for the results json to appear."""
    elapsed = 0
    while not os.path.exists(path) and elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
    if not os.path.exists(path):
        print(f"Warning: results file {path} not found after waiting {timeout}s")


def _log_results_to_tb(writer, path):
    """Read a single results json and log its metrics to TensorBoard."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: could not read results {path}: {e}")
        return

    step_marker = path.split("step_")[-1].split("-")[0]
    try:
        load_step = int(step_marker)
    except ValueError:
        print(f"Warning: could not parse step from {path}")
        return

    for k in ("average_ep_reward", "std_ep_reward", "win_rate"):
        if k not in data:
            continue
        val = data[k]
        # val may be a per-agent list (e.g. [173.4, 173.4, 183.4])
        # TensorBoard add_scalar requires a scalar, so take mean across agents
        if isinstance(val, (list, tuple, np.ndarray)):
            for i, v in enumerate(val):
                writer.add_scalar(f"eval/{k}/agent_{i}", float(v), load_step)
            writer.add_scalar(f"eval/{k}/mean", float(np.mean(val)), load_step)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            writer.add_scalar(f"eval/{k}", float(val), load_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
    Config = build_config_from_dict(exp_specs)

    evaluate(Config)
