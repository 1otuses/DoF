from typing import List

import numpy as np
import scipy.interpolate as interpolate

# PointMass环境专用的需要归一化的key列表：观测、动作、下一观测、增量
POINTMASS_KEYS = ["observations", "actions", "next_observations", "deltas"]



class DatasetNormalizer:
    """
    数据集归一化管理器 —— 对所有数据字段的归一化/反归一化进行统一调度
    核心能力：
    1. 根据配置选择不同的归一化策略(GaussianNormalizer, LimitsNormalizer, CDFNormalizer等)
    2. 区分"每个智能体独立归一化" vs "全局/共享参数归一化"
    3. 多智能体场景下，默认对每个智能体分别统计归一化参数
    4. 全局特征(如states)所有智能体共享同一套归一化参数
    """
    def __init__(
        self,
        dataset,                                # 原始数据集字典 {key: [n_episodes, max_len, N, dim]}
        normalizer,                             # 归一化器类名（字符串）或类对象
        global_feats: List[str] = ["states"],   # 全局特征key列表（共享归一化参数，不按agent分开）
        agent_share_parameters=False,           # 是否所有智能体共享同一套归一化参数
        path_lengths=None,                      # 每条episode的实际长度列表（用于flatten）
    ):
        # 将 [n_episodes, max_len, N, dim] 展平为 [总样本数, N, dim]
        dataset = flatten(dataset, path_lengths)

        # 从数据中推断维度信息
        self.n_agents = dataset["observations"].shape[1]       # 智能体数量 N
        self.observation_dim = dataset["observations"].shape[-1]  # 观测维度 O
        self.action_dim = (
            dataset["actions"].shape[-1] if "actions" in dataset.keys() else 0
        ) # A
        self.global_feats = global_feats
        self.agent_share_parameters = agent_share_parameters

        # 如果传入的是字符串，eval解析为对应的类
        if type(normalizer) is str:
            normalizer = eval(normalizer)

        # 为数据集中每个key创建对应的归一化器
        self.normalizers = {}
        for key, val in dataset.items():
            try:
                if key in global_feats or self.agent_share_parameters:
                    # 全局特征/共享参数: 所有智能体数据合并在一起拟合归一化参数
                    # val形如 [N, A, dim] -> reshape为 [N*A, dim]
                    self.normalizers[key] = normalizer(val.reshape(-1, val.shape[-1]))
                else:
                    # 每个智能体独立归一化: 为每个agent单独创建一个归一化器
                    # val[:, i] 取第i个智能体的所有数据: [N, dim]
                    self.normalizers[key] = [
                        normalizer(val[:, i]) for i in range(val.shape[1])
                    ]
            except Exception:
                print(f"[ utils/normalization ] Skipping {key} | {normalizer}")

    def __repr__(self):
        string = ""
        for key, normalizer in self.normalizers.items():
            string += f"{key}: {normalizer}]\n"
        return string

    def __call__(self, *args, **kwargs):
        """实例直接调用等价于normalize"""
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        """
        对指定key的数据进行归一化
        x shape 可以是 [..., N, dim] 或 [..., dim](取决于key是否全局特征)
        """
        if key in self.global_feats or self.agent_share_parameters:
            # 全局特征: 所有智能体一起归一化
            return self.normalizers[key].normalize(x)
        else:
            # 每个智能体独立归一化: 对每个agent分别调用对应的normalizer
            return np.stack(
                [
                    self.normalizers[key][i].normalize(x[..., i, :])
                    for i in range(x.shape[-2])
                ],
                axis=-2,
            )

    def unnormalize(self, x, key):
        """
        对指定key的数据进行反归一化(从归一化空间回到原始空间)
        """
        if key in self.global_feats or self.agent_share_parameters:
            return self.normalizers[key].unnormalize(x)
        else:
            return np.stack(
                [
                    self.normalizers[key][i].unnormalize(x[..., i, :])
                    for i in range(x.shape[-2])
                ],
                axis=-2,
            )


def flatten(dataset, path_lengths):
    """
    将分episode存储的数据集展平为样本级别
    
    输入: { key: [n_episodes, max_path_length, N, dim] }
    输出: { key: [总样本数, N, dim] }  总样本数 = sum(path_lengths)
    
    每个episode只取其实际长度 path_lengths[i] 部分(去掉padding)
    """
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        # 将每个episode截断到实际长度后拼接
        flattened[key] = np.concatenate(
            [x[:length] for x, length in zip(xs, path_lengths)], axis=0
        )
    return flattened

class PointMassDatasetNormalizer(DatasetNormalizer):
    """
    PointMass环境的专用归一化管理器
    与DatasetNormalizer的区别:不区分智能体维度,直接对所有数据做全局归一化
    适用于单智能体或point mass简单环境
    """
    def __init__(self, preprocess_fns, dataset, normalizer, keys=POINTMASS_KEYS):
        reshaped = {}
        for key, val in dataset.items():
            dim = val.shape[-1]
            reshaped[key] = val.reshape(-1, dim)  # 直接展平所有维度

        self.observation_dim = reshaped["observations"].shape[1]
        self.action_dim = reshaped["actions"].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        # 只对指定的keys创建归一化器
        self.normalizers = {key: normalizer(reshaped[key]) for key in keys}


class Normalizer:
    """
    归一化器基类
    
    所有具体归一化策略都继承此类，需实现 normalize 和 unnormalize 方法
    基类中计算了数据每个维度的最小值/最大值(所有子类共用的统计量)
    """
    def __init__(self, X):
        """
        Args:
            X: np.ndarray, shape [N, dim] 用于拟合归一化参数的数据
        """
        X = X.astype(np.float32)
        self.mins = X.min(axis=0)   # 每个维度的最小值 [dim]
        self.maxs = X.max(axis=0)   # 每个维度的最大值 [dim]

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    """
    调试用的恒等归一化器 —— 输入输出完全一致,不做任何变换
    用于测试流水线或数据已经归一化好的场景
    """

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    """
    高斯归一化器 —— 将数据映射到零均值、单位方差 (z-score标准化)
    
    公式: x_norm = (x - mean) / std
          x_orig = x_norm * std + mean
    
    适用于数据分布接近高斯分布的情况
    """

    def __init__(self, X, *args, **kwargs):
        super().__init__(X=X, *args, **kwargs)
        self.means = X.mean(axis=0)  # 每个维度的均值 [dim]
        self.stds = X.std(axis=0)    # 每个维度的标准差 [dim]
        self.z = 1                   # 缩放因子（可调整，默认1）

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    """
            f"""means: {np.round(self.means, 2)}\n    """
            f"""stds: {np.round(self.z * self.stds, 2)}\n"""
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    """
    极值归一化器 —— 将数据从 [xmin, xmax] 线性映射到 [-1, 1]
    公式: x_norm = 2 * (x - xmin) / (xmax - xmin) - 1
    适合已知数据边界（如动作范围）的场景
    注意：若 xmax = xmin,除零问题由 1e-20 处理
    """

    def normalize(self, x):
        # 第一步: [xmin, xmax] -> [0, 1]
        x = (x - self.mins) / (self.maxs - self.mins + 1e-20)
        # 第二步: [0, 1] -> [-1, 1]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        """
        反归一化: [-1, 1] -> [xmin, xmax]
        Args:
            x: 在 [-1, 1] 范围内的归一化数据
            eps: 容差，超出此范围时自动裁剪到 [-1, 1]
        """
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)

        # 第一步: [-1, 1] -> [0, 1]
        x = (x + 1) / 2.0
        # 第二步: [0, 1] -> [xmin, xmax]
        return x * (self.maxs - self.mins) + self.mins


class SafeLimitsNormalizer(LimitsNormalizer):
    """
    安全版极值归一化器 —— 与LimitsNormalizer相同,但能处理某个维度全为常数的数据
    当某个维度的最大值=最小值时，自动扩展范围 [-eps, +eps] 避免除零
    """

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(
                    f"""
                    [ utils/normalization ] Constant data in dimension {i} | """
                    f"""max = min = {self.maxs[i]}"""
                )
                self.mins -= eps   # 最小值减1
                self.maxs += eps   # 最大值加1


class CDFNormalizer(Normalizer):
    """
    CDF(累积分布函数)归一化器 —— 通过边际CDF将每个维度的数据转化为均匀分布
    核心思想:
    1. 对每个维度独立计算经验CDF
    2. 用CDF将原始数据映射到 [0, 1] 均匀分布
    3. 再映射到 [-1, 1] 范围
    优点:可以处理任意复杂分布的数据,将数据"拉直"为均匀分布,
          有助于扩散模型学习(模型输入分布更规整)
    缺点:反归一化依赖插值,对于超出训练数据范围的值只能clip
    
    注意:每个维度独立归一化,会破坏维度间的相关性结构
    """

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = X.shape[1]                      # 数据维度
        # 为每个维度创建一个1D CDF归一化器
        self.cdfs = [CDFNormalizer1d(X[:, i]) for i in range(self.dim)]

    def __repr__(self):
        return f"[ CDFNormalizer ] dim: {self.mins.size}\n" + "    |    ".join(
            f"{i:3d}: {cdf}" for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        """
        对输入x的每个维度分别执行 fn_name 操作(normalize或unnormalize)
        Args:
            fn_name: "normalize" 或 "unnormalize"
            x: 任意shape的数据,最后一维是特征维度
        Returns:
            与x相同shape的输出
        """
        shape = x.shape
        print(x.shape, self.dim)

        x = x.reshape(-1, shape[-1])  # [N, dim]
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs[:shape[-1]]):
            fn = getattr(cdf, fn_name)  # 获取CDFNormalizer1d的方法
            out[:, i] = fn(x[:, i])     # 对第i维单独操作
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap("normalize", x)

    def unnormalize(self, x):
        return self.wrap("unnormalize", x)


class CDFNormalizer1d:
    """
    一维CDF归一化器 —— 对单个维度进行CDF变换
    流程：
    - 计算经验CDF(通过数据排序和累积概率)
    - normalize: 原始值 -> CDF概率值 -> [-1, 1]
    - unnormalize: [-1, 1] -> CDF概率值 -> 原始值（通过插值反查）
    
    使用 scipy.interpolate.interp1d 实现前向和反向映射
    """

    def __init__(self, X):
        """
        Args:
            X: np.ndarray, shape [N] 一维数据
        """
        assert X.ndim == 1
        X = X.astype(np.float32)
        if X.max() == X.min():
            # 常数维度：不做任何变换
            self.constant = True
        else:
            self.constant = False
            # 计算经验CDF：得到分位数和对应的累积概率
            quantiles, cumprob = empirical_cdf(X)
            # 前向映射：分位数 -> 累积概率（插值）
            self.fn = interpolate.interp1d(quantiles, cumprob)
            # 反向映射：累积概率 -> 分位数（插值）
            self.inv = interpolate.interp1d(cumprob, quantiles)

            self.xmin, self.xmax = quantiles.min(), quantiles.max()  # 原始数据范围
            self.ymin, self.ymax = cumprob.min(), cumprob.max()      # CDF概率范围
        # self._warned = False  # 每个实例只打印一次 out-of-range 警告

    def __repr__(self):
        return f"[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}"

    def normalize(self, x):
        """
        归一化: 原始值 -> CDF概率 -> [-1, 1]
        Args:
            x: 原始数据值
        Returns:
            映射到 [-1, 1] 范围的值
        """
        if self.constant:
            return x

        # 裁剪到训练数据范围（避免外推）
        x = np.clip(x, self.xmin, self.xmax)
        # 通过CDF插值得到累积概率 [0, 1]
        y = self.fn(x)
        # 映射到 [-1, 1]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        """
        反归一化: [-1, 1] -> CDF概率 -> 原始值
        Args:
            x: [-1, 1] 范围的归一化值
            eps: 容差
        Returns:
            反归一化后的原始数据值
        """
        if self.constant:
            return x

        # [-1, 1] -> [0, 1] 恢复为CDF概率值
        x = (x + 1) / 2.0

        # 只在第一次越界时打印警告
        # if not self._warned and ((x < self.ymin - eps).any() or (x > self.ymax + eps).any()):
        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f"""[ dataset/normalization ] Warning: out of range in unnormalize: """
                f"""[{x.min()}, {x.max()}] | """
                f"""x : [{self.xmin}, {self.xmax}] | """
                f"""y: [{self.ymin}, {self.ymax}]"""
            )
            # self._warned = True

        # 裁剪到CDF概率范围内(避免外推警告)
        x = np.clip(x, self.ymin, self.ymax)

        # 通过逆插值还原为原始数据值
        y = self.inv(x)
        return y


def empirical_cdf(sample):
    """
    计算一维数据的经验累积分布函数(Empirical CDF)
    原理：对样本排序，每个唯一值的累积概率 = 小于等于该值的样本数 / 总样本数。
    Args:
        sample: np.ndarray, shape [N] 一维数据
    Returns:
        quantiles: 唯一值(升序排列)
        cumprob:   每个唯一值对应的累积概率 [0, 1]
    """
    # np.unique 返回排序后的唯一值及其出现次数
    quantiles, counts = np.unique(sample, return_counts=True)

    # 累积概率 = cumsum(counts) / N
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


def atleast_2d(x):
    """
    确保输入至少是2维数组。如果输入是1维,则增加一个维度
    用于CDFNormalizer初始化时统一处理输入shape
    Args:
        x: np.ndarray
    Returns:
        至少2维的数组
    """
    if x.ndim < 2:
        x = x[:, None]  # [N] -> [N, 1]
    return x
