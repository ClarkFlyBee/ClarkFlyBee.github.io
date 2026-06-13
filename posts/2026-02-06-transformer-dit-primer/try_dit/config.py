from dataclasses import dataclass

@dataclass
class DiTConfig:
    """DiT 模型架构配置"""
    image_size: int = 28    # MNIST 尺寸
    patch_size: int = 4     # 4×4 patches -> 7×7=49 tokens
    in_channels: int = 1    # MNIST 灰度图
    hidden_size: int = 256  # Transformer 隐藏维度
    depth: int = 4          # Transformer 层数
    num_heads: int = 4      # 注意力头数
    mlp_ratio: float = 4.0  # MLP 隐藏层倍数
    num_classes: int = 10   # MNIST 类别数
    dropout: float = 0.1    # Dropout 概率

    @property
    def num_patches(self) -> int:
        """计算 patch 数量"""
        return (self.image_size // self.patch_size) ** 2
    
@dataclass
class DiffusionConfig:
    """扩散过程配置"""
    timesteps: int = 1000       # 总扩散步数
    beta_start: float = 1e-4    # 起始噪声强度
    beta_end: float = 0.02      # 终止噪声强度
    schedule: str = "linear"    # beta 调度方式：linear/cosine

    # 采样配置
    sample_steps: int = 1000    # 采样步数（可小于 timesteps 用于加速）
    cfg_scale: float = 2.0      # Classifier-free guidance 强度
    cfg_dropout: float = 0.1    # 训练时条件丢弃概率

    device: str = "cpu"

@dataclass
class TrainConfig:
    """训练配置"""
    # 数据
    batch_size: int = 64
    num_workers: int = 4    # 数据加载线程

    # 优化器
    learning_rate: float = 1e-4 
    weight_decay: float = 0.03
    epochs: int = 50

    # 日志与保存
    log_every: int = 100    # 每 N batch 打印日志
    sample_every: int = 5   # 每 N epoch 生成样本
    save_every: int = 10    # 每 N epoch 保存

    # 路径
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"

    # 设备与恢复
    device: str = "cpu"
    resume: bool = False        # 是否从 checkpoint 恢复
    checkpoint_path: str = ""   # 指定恢复路径

# 组合配置（方便传递）
@dataclass
class Config:
    model: DiTConfig = DiTConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    train: TrainConfig = TrainConfig()