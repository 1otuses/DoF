import argparse
import glob
import json
import os

import numpy as np
import diffuser.utils as utils
import torch
import yaml
from diffuser.utils.launcher_util import (
    build_config_from_dict,
    discover_latest_checkpoint_path,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main(Config, RUN):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)
    dataset_extra_kwargs = dict()

    # configs that does not exist in old yaml files
    Config.discrete_action = getattr(Config, "discrete_action", False)
    Config.state_loss_weight = getattr(Config, "state_loss_weight", None)
    Config.opponent_loss_weight = getattr(Config, "opponent_loss_weight", None)
    Config.use_seed_dataset = getattr(Config, "use_seed_dataset", False)
    Config.residual_attn = getattr(Config, "residual_attn", True)
    Config.use_temporal_attention = getattr(Config, "use_temporal_attention", True)
    Config.env_ts_condition = getattr(Config, "env_ts_condition", False)
    Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
    Config.joint_inv = getattr(Config, "joint_inv", False)
    Config.use_zero_padding = getattr(Config, "use_zero_padding", True)
    Config.use_inv_dyn = getattr(Config, "use_inv_dyn", True)
    Config.pred_future_padding = getattr(Config, "pred_future_padding", False)
    # Config.use_learnable_agent_weights = getattr(
    #     Config, "use_learnable_agent_weights", False
    # )
    if not hasattr(Config, "agent_condition_type"):
        if Config.decentralized_execution:
            Config.agent_condition_type = "single"
        else:
            Config.agent_condition_type = "all"

    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    dataset_config = utils.Config(
        Config.loader,
        savepath="dataset_config.pkl", # 保存数据集配置
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        use_padding=Config.use_padding,
        use_action=Config.use_action,
        discrete_action=Config.discrete_action,
        max_path_length=Config.max_path_length,
        include_returns=Config.returns_condition,
        include_env_ts=Config.env_ts_condition,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        agent_share_parameters=utils.config.import_class(
            Config.model
        ).agent_share_parameters,
        use_seed_dataset=Config.use_seed_dataset,
        seed=Config.seed,
        use_inv_dyn=Config.use_inv_dyn,
        decentralized_execution=Config.decentralized_execution,
        use_zero_padding=Config.use_zero_padding,
        agent_condition_type=Config.agent_condition_type,
        pred_future_padding=Config.pred_future_padding,
        **dataset_extra_kwargs,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath="render_config.pkl", # 保存渲染器配置
        env_type=Config.env_type,
        env=Config.dataset,
    )
    data_encoder_config = utils.Config(
        getattr(Config, "data_encoder", "utils.IdentityEncoder"),
        savepath="data_encoder_config.pkl", # 保存数据编码器配置
    )

    dataset = dataset_config()
    renderer = render_config()
    data_encoder = data_encoder_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl", # 保存模型配置
        n_agents=Config.n_agents,
        # 如果采用models.TemporalUnet, 不接收n_agents
        # 其他models需要接收n_agents来决定是否共享参数
        horizon=Config.horizon + Config.history_horizon,
        history_horizon=Config.history_horizon,
        transition_dim=observation_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        env_ts_condition=Config.env_ts_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        max_path_length=Config.max_path_length,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl", # 保存扩散模型配置
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        discrete_action=Config.discrete_action,
        num_actions=getattr(dataset.env, "num_actions", 0),
        n_timesteps=Config.n_diffusion_steps,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        train_only_inv=Config.train_only_inv,
        share_inv=Config.share_inv,
        joint_inv=Config.joint_inv,
        # loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        state_loss_weight=Config.state_loss_weight,
        opponent_loss_weight=Config.opponent_loss_weight,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        data_encoder=data_encoder,
        use_learnable_agent_weights=Config.use_learnable_agent_weights,
        use_inv_dyn=Config.use_inv_dyn,
        device=Config.device,
        # ---- 信用引导参数 (CreditGuidedDiffusion 专用, GaussianDiffusion 忽略) ----
        use_credit_guide=getattr(Config, "use_credit_guide", False),
        credit_hidden_dim=getattr(Config, "credit_hidden_dim", 256),
        credit_router_mode=getattr(Config, "credit_router_mode", "minmax"),
        credit_lambda=getattr(Config, "credit_lambda", 0.01),
        cql_alpha=getattr(Config, "cql_alpha", 1.0),
        credit_condition_dropout=getattr(Config, "credit_condition_dropout", 0.2),
        cfg_guidance_w=getattr(Config, "cfg_guidance_w", 1.2),
        cfg_credit_w=getattr(Config, "cfg_credit_w", 0.5),
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl", # 保存训练器配置
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq, # 采样频率
        save_freq=Config.save_freq, # 保存频率
        log_freq=Config.log_freq, # 日志频率
        label_freq=int(Config.n_train_steps // Config.n_saves),
        eval_freq=Config.eval_freq, # 评估频率
        save_parallel=Config.save_parallel, # 并行保存
        bucket=logger.root,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    evaluator_config = utils.Config(
        Config.evaluator,
        savepath="evaluator_config.pkl", # 保存评估器配置
        verbose=False,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if Config.eval_freq > 0:
        evaluator = evaluator_config()
        evaluator.init(log_dir=os.path.join(logger.root, logger.prefix))
        trainer.set_evaluator(evaluator)

    if Config.continue_training:
        loadpath = discover_latest_checkpoint_path(
            os.path.join(trainer.bucket, logger.prefix, "checkpoint")
        )
        if loadpath is not None:
            state_dict = torch.load(loadpath, map_location=Config.device)
            logger.print(
                f"\nLoaded checkpoint from {loadpath} (step {state_dict['step']})\n",
                color="green",
            )
            trainer.step = state_dict["step"]
            trainer.model.load_state_dict(state_dict["model"])
            trainer.ema_model.load_state_dict(state_dict["ema"])

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(**batch)
    loss.backward()
    logger.print("✓")

    # -----------------------------------------------------------------------------#
    # ----------------------------- tensorboard & tqdm ----------------------------#
    # -----------------------------------------------------------------------------#

    tb_writer = SummaryWriter(os.path.join(logger.root, logger.prefix, "tensorboard"))

    # 检测是否为 CreditGuidedDiffusion 模型，用于进度条和 TensorBoard 显示
    is_credit_guided = (
        hasattr(diffusion, 'use_credit_guide') and
        diffusion.use_credit_guide and
        hasattr(diffusion, 'state_encoder')  # 只有真正启用了信用引导的模型才有 state_encoder
    )

    # 拦截 logger.log，将训练指标同时写入 TensorBoard
    _original_log = logger.log
    def _tb_log(step=None, loss=None, **metrics): # 记录训练指标
        if step is not None:
            if loss is not None:
                tb_writer.add_scalar("loss/train", loss, step)
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    # 信用引导模型特有的指标归类到 credit 子目录
                    credit_keys = ['credit_loss', 'q_tot_mean', 'c_mean',
                                   'diffuse_loss', 'cond_mode_uncond',
                                   'cond_mode_r', 'cond_mode_joint']
                    if is_credit_guided and key in credit_keys:
                        tb_writer.add_scalar(f"credit/{key}", float(value), step)
                    else:
                        tb_writer.add_scalar(f"loss/{key}", float(value), step)
        _original_log(step=step, loss=loss, **metrics)
    logger.log = _tb_log

    # 定义进度条显示的指标
    base_metrics = ["loss", "inv_loss", "inv_acc"]
    credit_metrics = ["credit_loss", "q_tot_mean", "c_mean", "diffuse_loss"]
    # 条件模式统计指标
    cond_mode_metrics = ["cond_mode_uncond", "cond_mode_r", "cond_mode_joint"]

    if is_credit_guided:
        display_metrics = base_metrics + credit_metrics + cond_mode_metrics
    else:
        display_metrics = base_metrics

    # 构建进度条显示函数
    def get_postfix_str(metrics_dict):
        """根据模型类型动态构建进度条后缀字符串"""
        parts = []
        for key in display_metrics:
            value = metrics_dict.get(key, None)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                if key in ["loss", "credit_loss", "diffuse_loss", "inv_loss", "c_mean"]:
                    parts.append(f"{key}={value:.4f}")
                else:
                    parts.append(f"{key}={value:.3f}")
        return parts

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ----------------------------------#
    # -----------------------------------------------------------------------------#

    def log_eval_results_to_tb(step):
        """读取并记录评估结果到 TensorBoard"""
        results_dir = os.path.join(trainer.bucket, logger.prefix, "results")
        if not os.path.isdir(results_dir):
            return
        # 查找当前 step 对应的评估结果
        pattern = f"step_{step}-"
        for fname in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
            if pattern not in fname:
                continue
            try:
                with open(fname) as f:
                    data = json.load(f)
                for k in ("average_ep_reward", "std_ep_reward", "win_rate"):
                    if k not in data:
                        continue
                    val = data[k]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        for ai, v in enumerate(val):
                            tb_writer.add_scalar(f"eval/{k}/agent_{ai}", float(v), step)
                        tb_writer.add_scalar(f"eval/{k}/mean", float(np.mean(val)), step)
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        tb_writer.add_scalar(f"eval/{k}", float(val), step)
            except (json.JSONDecodeError, ValueError, OSError):
                pass

    steps_remaining = Config.n_train_steps - trainer.step

    with tqdm(total=steps_remaining, desc="Training", initial=trainer.step) as pbar:
        for _ in range(steps_remaining):
            trainer.train(n_train_steps=1)

            # 动态更新进度条后缀
            postfix_parts = get_postfix_str(trainer._last_metrics)
            pbar.set_postfix_str(" | ".join(postfix_parts) if postfix_parts else "")
            pbar.update(1)

    trainer.finish_training()

    # 记录评估结果到 TensorBoard（遍历所有评估结果）
    results_dir = os.path.join(trainer.bucket, logger.prefix, "results")
    if os.path.isdir(results_dir):
        for fname in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
            try:
                with open(fname) as f:
                    data = json.load(f)
                step_marker = fname.split("step_")[-1].split("-")[0]
                load_step = int(step_marker)
                for k in ("average_ep_reward", "std_ep_reward", "win_rate"):
                    if k not in data:
                        continue
                    val = data[k]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        for ai, v in enumerate(val):
                            tb_writer.add_scalar(f"eval/{k}/agent_{ai}", float(v), load_step)
                        tb_writer.add_scalar(
                            f"eval/{k}/mean", float(np.mean(val)), load_step
                        )
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        tb_writer.add_scalar(f"eval/{k}", float(val), load_step)
            except (json.JSONDecodeError, ValueError, OSError):
                pass

    tb_writer.close()
    logger.log = _original_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    from ml_logger import RUN, logger

    Config = build_config_from_dict(exp_specs)

    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_name = Config.job_name.format(**vars(Config))
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=exp_specs["exp_name"],
        job_name=job_name + f"/{Config.seed}",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_root = os.path.join(project_root, "diffuser")
    os.makedirs(log_root, exist_ok=True)
    logger.configure(RUN.prefix, root=log_root)
    # logger.remove('*.pkl')
    logger.remove("traceback.err")
    # logger.remove("parameters.pkl")  # keep for evaluator
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text(
        """
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """,
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )
    logger.save_yaml(exp_specs, "exp_specs.yml")

    main(Config, RUN)
