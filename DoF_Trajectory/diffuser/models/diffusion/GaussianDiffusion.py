from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # DDPM噪声调度器（前向加噪+反向去噪）
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # DDIM采样调度器（加速采样，通常15~50步）
from diffusers.schedulers.scheduling_consistency_models import CMStochasticIterativeScheduler  # 一致性模型调度器（单步/少步采样）


import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning  # Losses: 损失函数集合（'l2', 'state_l2'等）;  apply_conditioning: 将条件约束（已知观测）替换到张量中


class QMixNet(nn.Module):
    """
    QMix混合网络 —— 多智能体值分解网络。
    将各智能体的独立动作值通过 hypernetwork 生成的权重进行单调混合，
    得到全局联合动作值 Q_tot,使得 argmax 在各智能体间保持单调性(abs(weights)实现单调性约束)
    """
    def __init__(self, state_dim: int, n_agents: int, action_dim: int):

        super(QMixNet, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        
        # hyper_w: 以全局状态为输入,生成 n_agents * action_dim 维的混合权重
        # 输出的 weights 将被重塑为 [batch_size, n_agents, action_dim]
        self.hyper_w = nn.Linear(state_dim, n_agents * action_dim)
        # hyper_b: 以全局状态为输入,生成 action_dim 维的偏置
        self.hyper_b = nn.Linear(state_dim, action_dim)

    def forward(self, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将多智能体动作混合成联合动作值。
        Args:
            actions: [batch_size, n_agents * action_dim] 联合动作
            states:  [batch_size, state_dim] 全局状态
        Returns:
            mixed_actions: [batch_size, 1, action_dim] 混合后的联合动作表示
        """
        batch_size = actions.shape[0]
        # w: [batch_size, n_agents, action_dim] — hypernetwork生成的权重，取绝对值保证单调性
        w = torch.abs(self.hyper_w(states)).view(batch_size, self.n_agents, self.action_dim)
        # b: [batch_size, 1, action_dim] — hypernetwork生成的偏置
        b = self.hyper_b(states).view(batch_size, 1, self.action_dim)
        
        # batch矩阵乘法: [batch_size,1,n_agents*action_dim] @ [batch_size,n_agents*action_dim,action_dim] -> [batch_size,1,action_dim]
        mixed_actions = torch.bmm(actions.view(batch_size, 1, -1), w).squeeze(1) + b
        return mixed_actions


class GaussianDiffusion(nn.Module):
    """
    高斯扩散模型 —— 多智能体轨迹生成的核心模块。
    功能：
    1. 前向扩散过程：对智能体的观测/动作序列逐步添加高斯噪声。
    2. 反向去噪过程：训练一个神经网络预测噪声，从纯噪声逐步还原出有意义的轨迹。
    3. 可选的逆动力学模型(Inverse Dynamics)：从观测序列中重建动作。
    4. 支持 classifier-free guidance(无分类器引导)和 returns guided 采样。
    B = batch_size, T = horizon(规划长度), H = history_horizon(历史窗口)
    N = n_agents(智能体数), O = observation_dim(观测维度), A = action_dim(动作维度)
    完整轨迹长度 = T + H
    """
    def __init__(
        self,
        model,                                    # 核心去噪U-Net/Transformer模型，用于预测噪声ε_θ或原始数据x_0
        n_agents: int,                            # N
        horizon: int,                             # T
        history_horizon: int,                     # H
        observation_dim: int,                     # O
        action_dim: int,                          # A
        use_inv_dyn: bool = True,
        discrete_action: bool = False,
        num_actions: int = 0,                     # 离散动作空间大小
        n_timesteps: int = 1000,                  # 扩散过程总时间步数
        clip_denoised: bool = False,
        predict_epsilon: bool = True,             # True: 预测噪声 ε; False: 预测原始数据 x_0
        action_weight: float = 1.0,               # 不使用逆动力学时，动作部分loss的额外权重
        hidden_dim: int = 256,
        loss_discount: float = 1.0,
        loss_weights: np.ndarray = None,          # 自定义loss权重矩阵（覆盖自动计算）
        state_loss_weight: float = None,          # （预留）状态部分loss的额外权重
        opponent_loss_weight: float = None,       # 对手观测部分loss的衰减权重（用于部分可观场景）
        returns_condition: bool = False,          # 是否使用 returns 进行条件生成（Classifier-Free Guidance）
        condition_guidance_w: float = 1.2,        # Classifier-Free Guidance 的引导强度 w
        returns_loss_guided: bool = False,        # 是否使用 returns 引导的loss（通过价值函数引导采样）
        loss_guidence_w: float = 0.1,             # returns 引导loss的权重系数
        value_diffusion_model: nn.Module = None,  # 预训练的 value diffusion 模型（用于 returns_loss_guided）
        train_only_inv: bool = False,
        share_inv: bool = True,
        joint_inv: bool = False,
        data_encoder: utils.Encoder = utils.IdentityEncoder(),  # 数据编码器（默认为恒等映射）
        use_learnable_agent_weights=False,        # 为每个智能体学习可训练权重，用于加权聚合多智能体输出
        use_qmix_combiner=False,                  # 使用 QMix 网络对多智能体输出进行单调混合
        use_data_agent_weights=False,             # 学习数据级别的智能体权重（用于非平衡数据场景）
        **kwargs,
    ):
        assert action_dim > 0
        assert (
            not returns_condition or not returns_loss_guided
        ), "不能同时使用 returns conditioning 和 returns loss guidance"
        # CFG通过条件引导和Q值引导实现两种不同的returns引导机制，通常不同时使用

        super().__init__()
        # ========== 基础维度信息 ==========
        self.n_agents = n_agents
        self.horizon = horizon # T
        self.history_horizon = history_horizon # H
        self.observation_dim = observation_dim # O
        self.action_dim = action_dim # A
        self.state_loss_weight = state_loss_weight
        self.opponent_loss_weight = opponent_loss_weight
        self.discrete_action = discrete_action
        self.num_actions = num_actions # N
        self.transition_dim = observation_dim + action_dim  # 转移维度 (O+A)
        
        # ========== 网络模块 ==========
        self.model = model
        self.use_inv_dyn = use_inv_dyn
        self.train_only_inv = train_only_inv
        self.share_inv = share_inv
        self.joint_inv = joint_inv
        self.data_encoder = data_encoder # 数据编码器
        self.use_learnable_agent_weights = use_learnable_agent_weights
        self.use_qmix_combiner = use_qmix_combiner
        self.use_data_agent_weights = use_data_agent_weights
        self.agent_models = nn.ModuleList([model for _ in range(n_agents)])
        if self.use_qmix_combiner:
            self.qmix_net = QMixNet(observation_dim, n_agents, action_dim)

        if self.use_learnable_agent_weights:
            self.agent_weights = nn.Parameter(torch.ones(n_agents))

        if self.use_data_agent_weights:
            self.data_agent_weights = nn.Parameter(torch.ones(n_agents))

        if self.use_inv_dyn:
            self.inv_model = self._build_inv_model(
                hidden_dim,
                output_dim=action_dim if not discrete_action else num_actions,
            )

        # ========== 条件生成设置 ==========
        self.returns_condition = returns_condition # returns_condition与return_loss_guided冲突
        self.condition_guidance_w = condition_guidance_w
        self.returns_loss_guided = returns_loss_guided
        self.loss_guidence_w = loss_guidence_w
        self.value_diffusion_model = value_diffusion_model # 与returns_loss_guided配套
        if self.value_diffusion_model is not None:
            self.value_diffusion_model.requires_grad_(False)  # 冻结值函数模型参数

        # ========== 扩散过程参数 ==========
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # DDPM调度器：管理beta/noise schedule，提供add_noise和step方法
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",      # 余弦beta schedule，对高分辨率更友好
        )
        self.use_ddim_sample = False                # 是否使用DDIM加速采样
        self.use_consistency_models_sample = False  # 是否使用一致性模型采样

        # ========== 损失函数设置 ==========
        loss_weights = self.get_loss_weights(loss_discount, action_weight)
        loss_type = "state_l2" if self.use_inv_dyn else "l2"
        # state_l2: 只对观测部分算MSE(动作由逆动力学重建)
        # l2: 对整个transition(观测+动作)算MSE
        self.loss_fn = Losses[loss_type](loss_weights)

    def _build_inv_model(self, hidden_dim: int, output_dim: int):
        """
        构建逆动力学模型(Inverse Dynamics Model)
        逆动力学的任务：给定当前观测 o_t 和下一时刻观测 o_{t+1}，预测 t 时刻的动作 a_t
        即: a_t ≈ inv_model( [o_t, o_{t+1}] )

        有三种构建模式：
        1. joint_inv: 所有智能体共用一个网络,输入是所有智能体的观测拼接,输出所有智能体的拼接动作
        2. share_inv: 所有智能体共享同一个网络,但每个智能体独立前向传播
        3. independent_inv: 每个智能体独立的网络参数
        """
        if self.joint_inv:
            # 联合逆动力学：输入 [B*T, A*2*O]，输出 [B*T, A*U]
            print("\n USE JOINT INV \n")
            inv_model = nn.Sequential(
                nn.Linear(self.n_agents * (2 * self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * output_dim),
            )

        elif self.share_inv:
            # 共享逆动力学：输入 [B*T*A, 2*O]，输出 [B*T*A, U]
            print("\n USE SHARED INV \n")
            inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            # 独立逆动力学：每个智能体一个独立网络
            # 离散动作时最后一层加Softmax输出动作概率分布
            print("\n USE INDEPENDENT INV \n")
            inv_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1) if self.discrete_action else nn.Identity(),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return inv_model
    
    def set_ddim_scheduler(self, n_ddim_steps: int = 15):
        """
        切换到DDIM采样调度器(加速采样)
        DDIM(Denoising Diffusion Implicit Models)允许使用更少的采样步数生成质量相近的结果
        
        Args:
            n_ddim_steps: DDIM采样的步数(通常15~50,远小于DDPM的1000步)
        """
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.ddim_noise_scheduler.set_timesteps(n_ddim_steps)
        self.use_ddim_sample = True
    
    def set_consistency_models_scheduler(self, n_consistency_model_steps: int = 15):
        """
        切换到一致性模型采样调度器(一步或少步采样)
        一致性模型通过将任意噪声水平映射到数据分布，实现单步或少步生成
        
        Args:
            n_consistency_model_steps: 一致性模型采样的步数
        """
        self.consistency_models_scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps = self.n_timesteps,
            sigma_min = 0.002,   # 最小噪声水平
            sigma_max = 80,      # 最大噪声水平
            rho = 7.0            # 调度曲线形状参数
        )
        self.consistency_models_scheduler.set_timesteps(n_consistency_model_steps)
        self.use_consistency_models_sample = True

    def get_loss_weights(self, discount: float, action_weight: Optional[float] = None):
        """
        计算轨迹中各时间步和各维度的loss权重系数。
        权重策略：
        - 时间维度：越远的未来步权重越低，按照 discount^t 指数衰减
        - 智能体维度：所有智能体权重相同
        - 特征维度：使用逆动力学时只对观测部分加权，否则对观测+动作整体加权
        - 不使用逆动力学时，可以在第一个规划步的动作维度上额外加权重
        Args:
            discount: 时间折扣因子(gamma),每个时间步乘以 discount^t
            action_weight: 不使用逆动力学时，动作维度的额外权重
        Returns:
            loss_weights: [T+H, A, O] 或 [T+H, A, O+U] 的权重矩阵
                          (T=horizon, H=history_horizon, A=n_agents, O=observation_dim, U=action_dim)
        """
        if self.use_inv_dyn:
            # 扩散模型只预测观测序列(动作由逆动力学重建)
            dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)
        else:
            # 扩散模型直接预测观测+动作的完整transition:O+U
            dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # 时间折扣权重：discount^t，并归一化使均值=1
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        # 历史窗口部分的loss权重设为0(仅用作conditioning，不参与loss计算)
        discounts = torch.cat([torch.zeros(self.history_horizon), discounts])
        # 外积：[T+H] x [O] -> [T+H, O]
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        # 扩展到智能体维度：[T+H, A, O]
        loss_weights = loss_weights.unsqueeze(1).expand(-1, self.n_agents, -1).clone()

        # 不使用逆动力学时，给动作维度额外的权重(通常action_weight>1以强调动作预测)
        if not self.use_inv_dyn:
            loss_weights[self.history_horizon, :, : self.action_dim] = action_weight
        return loss_weights

    def get_model_output( # 获取模型输出
        self,
        x: torch.Tensor,                           # [B, T+H, A, O] 当前带噪输入
        t: torch.Tensor,                           # [B] 当前时间步
        returns: Optional[torch.Tensor] = None,    # [B, 1] 条件returns值(可选)
        env_ts: Optional[torch.Tensor] = None,     # [B, T+H] 环境时间步编码(可选)
        attention_masks: Optional[torch.Tensor] = None,  # attention mask(可选)
        states: Optional[torch.Tensor] = None,     # 全局状态(可选，用于QMix)
    ):
        # (仅在首次调用时打印一次维度信息，避免干扰进度条)
        print("get_model_output: \n " \
        "x.shape={}, \n " \
        "t.shape={}, \n " \
        "returns.shape={}, \n " \
        "env_ts.shape={}, \n " \
        "attention_masks.shape={}, \n " \
        "states.shape={}".format(
            x.shape, t.shape, returns.shape if returns is not None else None,
            env_ts.shape if env_ts is not None else None,
            attention_masks.shape if attention_masks is not None else None,
            states.shape if states is not None else None,
        ))
        """
        获取模型输出(predicted epsilon/x0),支持可选的条件生成
        当 self.returns_condition=True 时，使用 classifier-free guidance (CFG):
        - epsilon_cond: 有条件预测(给定returns)
        - epsilon_uncond: 无条件预测(通过use_dropout=True丢弃条件)
        - 最终输出: epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
          其中w=condition_guidance_w控制条件引导强度
        当 use_learnable_agent_weights=True 时:
        - 对每个智能体的输出进行加权平均(可学习的智能体重要性权重)
        """
        if self.returns_condition:
            
            per_epsilon_con = []
            for i, per_model in enumerate(self.agent_models):
                per_epsilon = per_model(
                    x[:, :, i, :], # 每个智能体独立前向传播，输入是该智能体的观测序列 
                    t,
                    returns = returns[:, :, i],
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    use_dropout=False,
                )
                per_epsilon_con.append(per_epsilon)
            epsilon_cond = torch.stack(per_epsilon_con, dim=2)

            # ---- 有条件预测 ----
            # epsilon_cond = self.model(
            #     x, t,
            #     returns=returns,
            #     env_timestep=env_ts,
            #     attention_masks=attention_masks,
            #     use_dropout=False,  # 不使用dropout，保留条件信息
            # )
            # 可学习智能体加权（可选）
            if self.use_learnable_agent_weights:

                weighted_epsilon_cond = epsilon_cond * self.agent_weights.view(1, 1, -1, 1)
                epsilon_cond = weighted_epsilon_cond / self.agent_weights.sum()

            per_epsilons_uncon = []
            for i, per_model in enumerate(self.agent_models):
                per_epsilon_un = per_model(
                    x[:,:,i,:],
                    t,
                    returns = returns[:,:,i],
                    env_timestep = env_ts,
                    attention_masks = attention_masks,
                    use_dropout=True,
                    )
                per_epsilons_uncon.append(per_epsilon_un)

            epsilon_uncond = torch.stack(per_epsilons_uncon, dim=2)


            # ---- 无条件预测 ----
            # epsilon_uncond = self.model(
            #     x, t,
            #     returns=returns,
            #     env_timestep=env_ts,
            #     attention_masks=attention_masks,
            #     use_dropout=True,  # 使用dropout，丢弃条件信息近似无条件
            # )
            if self.use_learnable_agent_weights:

                weighted_epsilon_uncond = epsilon_uncond * self.agent_weights.view(1, 1, -1, 1)
                epsilon_uncond = weighted_epsilon_uncond / self.agent_weights.sum()

            # Classifier-Free Guidance：有条件-无条件 外推
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            per_epsilons = []
            for i, per_model in enumerate(self.agent_models):
                per_epsilon = per_model(
                    x[:,:,i,:],
                    t,
                    returns = returns[:,:,i],
                    env_timestep = env_ts,
                    attention_masks = attention_masks,
                    use_dropout=False,
                )
                per_epsilons.append(per_epsilon)
            epsilons = per_epsilons
            epsilon =  torch.stack(epsilons, dim=2)
            # 不使用条件生成，直接模型前向传播
            # epsilon = self.model(
            #     x, t,
            #     returns=returns,
            #     env_timestep=env_ts,
            #     attention_masks=attention_masks,
            #     use_dropout=False,
            # )

        return epsilon

    @torch.no_grad()
    def conditional_sample( # 条件采样
        self,
        cond: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        horizon: int = None,
        attention_masks: Optional[torch.Tensor] = None,
        verbose: bool = True,
        return_diffusion: bool = False,
    ):
        """
        条件采样：从纯噪声开始，逐步去噪生成智能体的观测序列
        采样过程：
        1. 从高斯噪声 N(0, 0.5*I) 初始化 x_T
        2. 对每个时间步 t = T-1, T-2, ..., 0:
           a. 应用条件约束(将已知的conditioning部分替换回去)
           b. 通过模型预测噪声 epsilon
           c. 用 scheduler.step 进行一次去噪:x_{t-1} = step(x_t, epsilon)
        3. 返回最终生成的观测序列 x_0
        Args:
            cond: 条件字典，包含已知的观测片段等信息
            returns: 条件returns值(用于CFG)
            env_ts: 环境时间步编码(可选)
            horizon: 生成的轨迹总长度
            attention_masks: 注意力掩码
            verbose: 是否显示进度条
            return_diffusion: 是否返回完整的扩散过程
        Returns:
            x: 生成的观测序列 [B, T+H, N, O]
            (可选) diffusion: 完整的扩散过程 [B, n_steps+1, T+H, N, O]
        """
        batch_size = cond["x"].shape[0]
        horizon = horizon or self.horizon + self.history_horizon
        
        shape = (batch_size, horizon, self.n_agents, self.observation_dim)

        device = list(cond.values())[0].device
        
        # 选择调度器
        if self.use_ddim_sample:
            scheduler = self.ddim_noise_scheduler
        else:
            scheduler = self.noise_scheduler

        if self.use_consistency_models_sample:
            scheduler = self.consistency_models_scheduler
        else:
            scheduler = self.noise_scheduler
            
        # 从各向同性高斯噪声初始化 x_T
        # 使用 0.5 缩放使初始噪声的方差接近数据分布方差
        x = 0.5 * torch.randn(shape, device=device)  # [B, T+H, N, O]

        if return_diffusion:
            diffusion = [x]  # 记录扩散过程的所有中间结果

        # 按时间步从 T-1 到 0 迭代去噪
        timesteps = scheduler.timesteps

        progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()
        for t in timesteps:
            # 应用条件约束(将已知的conditioning片段替换到 x 中)
            x = apply_conditioning(x, cond, action_dim=self.action_dim)
            x = self.data_encoder(x)

            # 当前时间步t，构造batch维度的时间张量
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long) # [B]
            # 模型预测噪声
            model_output = self.get_model_output(
                x, ts, returns, env_ts, attention_masks
            )
            print("model_output: ", model_output.shape)
            
            # scheduler.step: 根据预测的噪声和当前x_t，计算x_{t-1}
            x = scheduler.step(model_output, t, x).prev_sample

            if verbose:
                progress.update({"t": t})
            if return_diffusion:
                diffusion.append(x)

        # 最终再次应用条件约束
        x = apply_conditioning(x, cond, action_dim=self.action_dim)
        x = self.data_encoder(x)

        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)  # [B, n_steps+1, T+H, N, O]
        else:
            return x  # [B, T+H, N, O]

    def p_losses(
        self,
        x_start: torch.Tensor,                      # [B, T+H, N, O] 原始无噪声观测序列
        cond: Dict[str, torch.Tensor],              # 条件字典
        t: torch.Tensor,                            # [B] 随机采样的时间步
        loss_masks: torch.Tensor,                   # [B, T+H, N, 1] loss掩码（padding位置=0）
        attention_masks: Optional[torch.Tensor] = None,  # attention掩码
        returns: Optional[torch.Tensor] = None,     # 条件returns
        env_ts: Optional[torch.Tensor] = None,      # 环境时间步
        states: Optional[torch.Tensor] = None,      # 全局状态(用于QMix)
    ):
        """
        计算扩散模型的损失(per-step loss)
        流程：
        1. 生成高斯噪声 noise,加到 x_start 上得到 x_noisy
        2. 通过模型预测噪声 epsilon_theta(x_noisy, t)
        3. 计算 ||epsilon - epsilon_theta||^2 的加权MSE
        扩展功能：
        - use_learnable_agent_weights: 可学习的智能体权重
        - use_qmix_combiner: QMix单调混合多智能体输出
        - opponent_loss_weight: 对手观测部分loss降权
        - returns_loss_guided: 额外的returns引导loss
        """
        noise = torch.randn_like(x_start)  # [B, T+H, N, O] 采样高斯噪声

        # 前向扩散：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        # 对加噪后的结果应用条件约束
        x_noisy = apply_conditioning(x_noisy, cond, action_dim=self.action_dim)
        x_noisy = self.data_encoder(x_noisy)

        per_epsilons = []
        for i, per_model in enumerate(self.agent_models): # 遍历每个智能体的模型 适用models: TemporalUnet
            per_epsilon = per_model(
                x_noisy[:, :, i, : ],
                t,
                returns = returns[:, : , i],
                env_timestep = env_ts,
                attention_masks = attention_masks,
            )
            per_epsilons.append(per_epsilon)
        epsilon = torch.stack(per_epsilons, dim = 2)

        # 模型预测噪声(或原始数据)
        # epsilon = self.model(
        #     x_noisy,
        #     t,
        #     returns=returns,
        #     env_timestep=env_ts,
        #     attention_masks=attention_masks,
        # )

        # 可学习智能体权重：加权聚合各智能体的输出
        if self.use_learnable_agent_weights:
            weighted_x_recon = epsilon * self.agent_weights.view(1, 1, -1, 1)
            epsilon = weighted_x_recon / self.agent_weights.sum()
        
        # QMix组合器：将多智能体输出通过QMix网络混合
        if self.use_qmix_combiner:
            batch_size, seq_len, n_agents, action_dim = epsilon.shape
            epsilon = epsilon.view(batch_size, seq_len, n_agents * action_dim)
            
            # 如果没有提供全局状态,从x_start中提取观测部分作为状态
            if states is None:
                states = x_start[:, :, :, :self.observation_dim].reshape(batch_size, seq_len, -1)
            else:
                states = states.reshape(batch_size, seq_len, -1)
            
            epsilon = self.qmix_net(epsilon, states)
            epsilon = epsilon.view(batch_size, seq_len, n_agents, action_dim)

        if not self.predict_epsilon:
            # 如果预测x0而不是epsilon,需要应用条件约束和编码
            epsilon = apply_conditioning(epsilon, cond, action_dim=self.action_dim)
            epsilon = self.data_encoder(epsilon)

        assert noise.shape == epsilon.shape

        # 计算loss
        if self.predict_epsilon:
            loss, info = self.loss_fn(epsilon, noise)   # MSE(预测噪声, 真实噪声)
        else:
            loss, info = self.loss_fn(epsilon, x_start)  # MSE(预测x0, 真实x0)

        # 对手观测loss衰减：对非当前智能体的观测部分施加更低的权重
        if "agent_idx" in cond.keys() and self.opponent_loss_weight is not None:
            opponent_loss_weight = torch.ones_like(loss) * self.opponent_loss_weight
            indices = (
                cond["agent_idx"]
                .to(torch.long)[..., None]
                .repeat(
                    1, opponent_loss_weight.shape[1], 1, opponent_loss_weight.shape[-1]
                )
            )
            opponent_loss_weight.scatter_(dim=2, index=indices, value=1)
            loss = loss * opponent_loss_weight

        # 应用loss mask（忽略padding部分），先按样本和智能体求平均
        loss = (
            (loss * loss_masks).mean(dim=[1, 2]) / loss_masks.mean(dim=[1, 2])
        ).mean()

        # Returns引导的loss（可选）：通过价值函数引导生成高回报轨迹
        if self.returns_loss_guided:
            returns_loss = self.r_losses(x_noisy, t, epsilon, cond)
            info["returns_loss"] = returns_loss
            loss = loss + returns_loss * self.loss_guidence_w

        return loss, info

    def r_losses(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, cond: Dict):
        """
        Returns引导的损失函数(Return Loss Guidance)
        思路:使用一个预训练的价值扩散模型(value_diffusion_model)来估计
        当前去噪轨迹的价值，然后通过梯度下降最大化该价值。
        具体步骤:
        1. 从当前噪声估计 x_0(预测的原始数据)
        2. 用 q_posterior 计算 x_{t-1} 的分布参数
        3. 从该分布采样 x_{t-1}
        4. 用价值扩散模型评估 x_{t-1} 的价值
        5. 返回负价值作为loss(即最大化价值)
        Args:
            x_t: [B, T+H, N, O] 当前时间步的带噪观测
            t: [B] 当前时间步
            noise: [B, T+H, N, O] 模型预测的噪声
            cond: 条件字典
            
        Returns:
            loss: 标量，负的价值预测均值
        """
        b = x_t.shape[0]
        t = t.detach().to(torch.int64)
        
        # 从噪声预测 x_0（去噪后的原始数据）
        x_recon = self.predict_start_from_noise(x_t, t, noise)

        # 裁剪到[-1,1]范围
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        # 计算后验分布 q(x_{t-1} | x_t, x_0) 的参数
        model_mean, _, model_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )

        # 从后验分布采样 x_{t-1}
        noise = 0.5 * torch.randn_like(x_t)
        # t=0时方差为0(即x_0没有噪声)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        x_t_minus_1 = (
            model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        )
        x_t_minus_1 = apply_conditioning(x_t_minus_1, cond, action_dim=self.action_dim)
        x_t_minus_1 = self.data_encoder(x_t_minus_1)

        # 用预训练的价值扩散模型评估 x_{t-1} 的价值
        value_pred = self.value_diffusion_model(x_t_minus_1, t)

        # 返回负价值(最大化价值 = 最小化负价值)
        return -1.0 * value_pred.mean()

    def compute_inv_loss( # 计算逆动力学损失
        self,
        x: torch.Tensor,                                 # [B, T+H, N, O+A] 完整transition(obs+action)
        loss_masks: torch.Tensor,                         
        legal_actions: Optional[torch.Tensor] = None,     # [B, T+H, N, num_actions] 合法动作掩码(离散动作)
        use_data_agent_weights: bool = False,             # 是否使用数据级智能体权重
    ):
        """
        给定 o_t 和 o_{t+1}，预测 a_t
        学习函数 f: (o_t, o_{t+1}) -> a_t
        
        数据准备：
        - x_t = o_t (当前观测)
        - a_t = 真实动作
        - x_{t+1} = o_{t+1} (下一时刻观测)
        - x_comb_t = [o_t, o_{t+1}] (拼接后的输入)
        
        支持三种逆动力学模式：
        - joint_inv: 所有智能体联合输入输出
        - share_inv: 共享网络参数但独立推理
        - independent_inv: 每个智能体独立网络
        
        支持离散和连续动作：
        - 离散动作: CrossEntropyLoss 交叉熵损失
        - 连续动作: MSELoss 均方误差损失
        
        Args:
            x: 包含观测和动作的完整transition(obs+action)
            loss_masks: 指定哪些位置参与loss计算
            legal_actions: 离散动作的合法动作掩码(非法动作位置通过-1e10屏蔽)
            use_data_agent_weights: 是否按数据级智能体权重聚合
            
        Returns:
            inv_loss: 逆动力学损失标量
            info: 包含额外信息(如 inv_acc 逆动力学准确率)
        """
        info = {}
        
        # 从完整transition中提取观测和动作：x = [a_t || o_t] 的拼接
        # x的结构假设：前半部分是action_dim维动作，后半部分是observation_dim维观测
        x_t = x[:, :-1, :, self.action_dim :]             # o_t: [B, T+H-1, N, O]  [:-1]裁掉最后一步
        a_t = x[:, :-1, :, : self.action_dim]             # a_t: [B, T+H-1, N, A]
        x_t_1 = x[:, 1:, :, self.action_dim :]            # o_{t+1}: [B, T+H-1, N, O]  [1:]裁掉第一步
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)        # [o_t, o_{t+1}]: [B, T+H-1, N, 2*O]
        
        # loss mask 平移一位（因为预测的是 t 时刻的动作，用 t+1 时刻的mask）
        masks_t = loss_masks[:, 1:]
        if legal_actions is not None:
            legal_actions_t = legal_actions[:, :-1].reshape(
                -1, *legal_actions.shape[2:]
            )
        
        if use_data_agent_weights:
            # 按数据级智能体权重聚合：将所有智能体的数据求和
            x_comb_t = x_comb_t.sum(dim=2, keepdim=True) # 在智能体维度上求和 [B, T+H-1, 1, 2*O]
            a_t = a_t.sum(dim=2, keepdim=True)
            masks_t = masks_t.sum(dim=2, keepdim=True)
            if legal_actions is not None:
                legal_actions_t = legal_actions_t.sum(dim=2, keepdim=True)
            
            x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[-1]) # [B*(T+H-1), 2*O] 合并batch和time维度
            a_t = a_t.reshape(-1, a_t.shape[-1])
            masks_t = masks_t.reshape(-1)
            if legal_actions is not None:
                legal_actions_t = legal_actions[:, :-1].sum(dim=2, keepdim=True)   
        else:
            # 按智能体维度独立处理
            # [B*(T+H-1), A, 2*O] -> 合并batch和time维度
            x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[2], 2 * self.observation_dim)
            a_t = a_t.reshape(-1, a_t.shape[2], self.action_dim)
            masks_t = masks_t.reshape(-1, masks_t.shape[2])
            if legal_actions is not None:
                legal_actions_t = legal_actions[:, :-1].reshape(-1, *legal_actions.shape[2:])


        if self.joint_inv or self.share_inv:
            # ---- 联合或共享逆动力学 ----
            if self.joint_inv:
                # 联合模式：所有智能体输入拼成一个大向量
                pred_a_t = self.inv_model(
                    x_comb_t.reshape(x_comb_t.shape[0], -1)  # [B*(T+H-1), A*2*O]
                ).reshape(x_comb_t.shape[0], x_comb_t.shape[1], -1)  # [B*(T+H-1), A, U]
            else:
                # 共享模式：每个智能体独立推理
                pred_a_t = self.inv_model(x_comb_t)  # [B*(T+H-1), A, U]

            # 离散动作：屏蔽非法动作(在logits上加-1e10)
            if legal_actions is not None:
                pred_a_t[legal_actions_t == 0] = -1e10
            if self.discrete_action:
                # CrossEntropyLoss for discrete action
                inv_loss = (
                    F.cross_entropy(
                        pred_a_t.reshape(-1, pred_a_t.shape[-1]),
                        a_t.reshape(-1).long(),
                        reduction="none",
                    )
                    * masks_t.reshape(-1)
                ).mean() / masks_t.mean()
                # 计算逆动力学准确率(仅用于监控)
                inv_acc = (
                    (pred_a_t.argmax(dim=-1, keepdim=True) == a_t)
                    .to(dtype=float)
                    .squeeze(-1)
                    * masks_t
                ).mean() / masks_t.mean()
                info["inv_acc"] = inv_acc
            else:
                # MSELoss for continuous action
                inv_loss = (
                    F.mse_loss(pred_a_t, a_t, reduction="none") * masks_t.unsqueeze(-1)
                ).mean() / masks_t.mean()

        else:
            # ---- 独立逆动力学：每个智能体有自己的网络 ----
            inv_loss = 0.0
            for i in range(self.n_agents):
                pred_a_t = self.inv_model[i](x_comb_t[:, i])  # 第i个智能体的预测
                if self.discrete_action:
                    inv_loss += (
                        F.cross_entropy(
                            pred_a_t, a_t[:, i].reshape(-1).long(), reduction="none"
                        )
                        * masks_t[:, i]
                    ).mean() / masks_t[:, i].mean()
                else:
                    inv_loss += (
                        F.mse_loss(pred_a_t, a_t[:, i]) * masks_t[:, i].unsqueeze(-1)
                    ).mean() / masks_t[:, i].mean()

        return inv_loss, info

    def loss( # 主损失函数
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        legal_actions: Optional[torch.Tensor] = None,
    ):
        """
        主损失函数，组合扩散损失和逆动力学损失
        总损失 = 0.5 * 扩散损失 + 0.5 * 逆动力学损失(使用逆动力学时)
        总损失 = 扩散损失(不使用逆动力学时)
        
        扩散损失(p_losses):
        - 对观测序列加噪,让模型预测噪声,计算MSE
        - 如果 returns_condition=True,使用classifier-free guidance
        
        逆动力学损失(compute_inv_loss):
        - 从 o_t, o_{t+1} 预测 a_t
        - 支持离散动作(CrossEntropy)和连续动作(MSE)
        
        Args:
            x: 当 use_inv_dyn=True 时为 [B, T+H, A, O+U]（包含动作）
               当 use_inv_dyn=False 时为 [B, T+H, A, O]（仅观测，扩散直接预测全部）
            cond: 条件信息字典
            loss_masks: 指示哪些位置参与loss计算
            attention_masks: 注意力掩码
            returns: 回报条件
            env_ts: 环境时间步信息
            states: 全局状态
            legal_actions: 合法动作（离散动作场景）
            
        Returns:
            loss: 标量损失值
            info: 包含各组件损失的字典（用于日志记录）
        """
        if self.train_only_inv:
            # 仅训练逆动力学：冻结扩散模型，只训练逆动力学
            assert self.use_inv_dyn, "If train_only_inv, must use inv_dyn"
            info = {}
        else:
            batch_size = len(x)
            # 为batch中每个样本随机采样一个时间步 t ∈ [0, num_train_timesteps)
            t = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=x.device,
            ).long()

            if self.use_inv_dyn:
                # 使用逆动力学时，扩散模型只预测观测部分(动作由逆动力学重建)
                diffuse_loss, info = self.p_losses(
                    x[..., self.action_dim :],  # 只取观测部分 [B, T+H, A, O]
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                    states,
                )
            else:
                # 不使用逆动力学时，扩散模型直接预测完整transition（观测+动作）
                diffuse_loss, info = self.p_losses(
                    x,
                    cond,
                    t,
                    loss_masks,
                    attention_masks,
                    returns,
                    env_ts,
                    states,
                )

        if self.use_inv_dyn:
            # 计算逆动力学损失
            inv_loss, inv_info = self.compute_inv_loss(x, loss_masks, legal_actions, self.use_data_agent_weights)
            info = {**info, **inv_info}
            info["inv_loss"] = inv_loss

            if self.train_only_inv:
                return inv_loss, info

            # 总损失 = 0.5 * 扩散损失 + 0.5 * 逆动力学损失
            loss = (1 / 2) * (diffuse_loss + inv_loss)
        else:
            loss = diffuse_loss

        return loss, info

    def forward(self, cond, *args, **kwargs):
        """
        前向传播(推理时使用):从条件cond生成轨迹
        等价于调用 conditional_sample
        """
        return self.conditional_sample(cond=cond, *args, **kwargs)