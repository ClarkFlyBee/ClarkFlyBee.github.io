+++
date = '2026-02-06T20:09:38+08:00'
draft = true
title = 'Transformer + 扩散模型：DiT 入门'
ShowToc = false
math = true
+++

> 本文内容基于论文 [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)。
> 
> 但文章内容并不是论文阅读笔记，而是对 DiT 的入门介绍。未来有机会可以写一个详细的阅读笔记。

## 扩散模型介绍

扩散模型（***Denoising Diffusion Probabilistic Models***，DDPMs）在图像/音频/视频生成方面取得了显著的成果。

本文中，采用离散时间（潜变量模型）（discrete-time (lantent variable model)）的视角，事实上有多种关于扩散模型的观点，可以都去了解一下。

> 随机微分方程（SDE）、得分匹配（Score Matching）、朗之万动力学（Langevin Dynamics）、变分推断视角（Variational Inference）……

### 什么是扩散模型

## DiT 架构详解

## 代码实践

这里，我们用 MNIST 数据集（手写数字）来做一个简单的小案例，用扩散模型生成指定数字的**单通道灰度图**。

---

下面的 “**代码详细解释**” 中，我会把我自己在代码中不懂或困惑的地方进行详细说明，可能会有些冗余。

读者可按需阅读或跳过。

---

### config.py

集中管理所有超参数，避免魔法数字散落在代码中

```python
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
```

#### 代码细节解释

- Patch 与 Batch 区分
  
  - **Patch**：一次性送进模型的**一组样本**
  
  - **Batch**：将一张大图切成**小方块**，每个方块是一个 token
  
  - 对于我们的例子：
    
    ```python
    class DiTConfig:
        """DiT 模型架构配置"""
        ……
        patch_size: int = 4     # 4×4 patches -> 7×7=49 tokens
    
    class TrainConfig:
        """训练配置"""
        # 数据
        batch_size: int = 64
    ```
    
    - 一张图：28×28；Patch 大小：4×4；切成 7×7=49 个patches
    
    - Batch=64，一次处理 64 张图

- **Beta** 控制每一步加多少噪声
  
  ```python
  class DiffusionConfig:
      """扩散过程配置"""
      ……
      schedule: str = "linear"    # beta 调度方式：linear/cosine
  ```
  
  - **Linear**：直线增长，均匀加噪；简单，但后期可能加噪过快
  
  - **Cosine**：余弦曲线，先慢后快；更平滑，训练更稳定

- ***采样***  可以理解为 **生成图片的过程**
  
  - 训练时：从真实图像出发，前向加噪学习
  
  - 采样时：从纯噪声出发，反向去噪生成
  
  - 因为每步都从概率分布（高斯分布）中**随机抽取**数值，不是确定性计算

- **权重衰减**（***Weight Decay***）是防止模型过拟合的正则化手段
  
  - 普通梯度下降：参数=参数-学习率×梯度
  
  - 带上权重衰减：参数=参数-学习率×梯度 - **学习率×weight_decay×参数**
    
    ```python
    class TrainConfig:
        """训练配置"""
        ……
        # 优化器
        ……
        weight_decay: float = 0.03
    ```
  
  - 强迫模型参数保持小数值，避免过于依赖“一家独大”的参数，提升模型**泛化能力**

- `@dataclass` 自动为类生成样板代码（`__init__`、`__repr__`、`__eq__`），减少重复代码

### utils.py

纯工具函数，无状态，可被任何模块导入

```python
import math
import torch
from typing import Optional
import matplotlib.pyplot as plt
import os

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    正弦位置编码，将标量时间步嵌入到 dim 维向量。

    基于 Transformer 原始论文的位置编码，但用于连续时间步。

    Args:
        timesteps: [B] 或 [B, 1] 的时间步张量
        dim: 输入嵌入维度
        max_period: 正弦/余弦的最大周期

    Returns:
        embedding: [B, dim] 时间嵌入 
    """

    # 确保是 1D
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    if timesteps.ndim > 1:
        timesteps = timesteps.view(-1)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)

    args = timesteps[:, None].float() * freqs[None, :]                  # [B, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # [B, dim]

    # 奇数维度时补零
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding

def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    将图像分割为 patches 并展平为序列。

    Args:
        x: [B, C, H, W] 图像
        patch_size: patch 边长

    Returns: 
        patches: [B, N, C*patch_size^2] 其中 N = (H//p)*(W//p)
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "图像尺寸必须能被 patch_size 整除"

    # 使用 unfold 提取滑动窗口，然后 reshape
    # [B, C, H//p, p, W//p, p] -> [B, H//p, W//p, C, p, p]
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()

    # [B, N, C*p*p]
    patches = x.view(B, -1, C * patch_size * patch_size)
    return patches

def unpatchify(patches: torch.Tensor, patch_size: int, channels: int, img_size: int) -> torch.Tensor:
    """
    将 patch 序列还原为图像。

    Args: 
        patches: [B, N, C*patch_size^2]
        patch_size: patch 边长
        channels: 图像通道数
        img_size: 图像边长

    Returns:
        x: [B, C, H, W]
    """
    B = patches.shape[0]
    H = W = img_size // patch_size
    C = channels

    # [B, N, C*p*p] -> [B, H, W, C, p, p]
    x = patches.view(B, H, W, C, patch_size, patch_size)

    # [B, C, H, p, W, p] -> [B, C, H*p, W*p]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H * patch_size, W * patch_size)

    return x

def normalize_image(x: torch.Tensor, 
                    mean: Optional[tuple] = None, 
                    std: Optional[tuple] = None) -> torch.Tensor:
    """
    图像归一化到 [-1, 1] 或标准化。

    Args:
        x: [B, C, H, W], 值域 [0, 1]
        mean/std: 若为 None，直接缩放到 [-1, 1]

    Returns:
        归一化后的图像
    """

    if mean is None:
        return x * 2.0 - 1.0
    else:
        # 标准标准化
        mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
        return (x - mean) / std

def denormalize_image(x: torch.Tensor, 
                      mean: Optional[tuple] = None, 
                      std: Optional[tuple] = None) -> torch.Tensor:
    """
    反归一化，将图像还原到 [0, 1] 用于可视化
    """
    if mean is None:
        return (x + 1.0) / 2.0
    else:
        mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
        return x * std + mean

def save_image_grid(images: torch.Tensor, 
                    path: str, 
                    nrow: int = 10, 
                    normalize: bool = True):
    """
    保存图像网格。

    Args:
        images: [N, C, H, W] 或 [N, H, W] 图像张量
        path: 保存路径
        nrow: 每行图像数
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if images.ndim == 3:
        images = images.unsqueeze(1)    # 添加通道维

    if normalize: 
        images = denormalize_image(images)

    images = images.clamp(0, 1).cpu().numpy()

    N = images.shape[0]
    ncol = math.ceil(N / nrow)

    fig, axes = plt.subplot(ncol, nrow, figsize=(nrow * 1.5, ncol * 1.5))
    if ncol == 1:
        axes = axes.reshape(1, -1)
    if nrow == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(N):
        i, j = idx // nrow, idx % nrow
        ax = axes[i, j]
        img = images[idx]

        # 单通道灰度图
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap='gray')
        else:
            ax.imshow(img.transpose(1, 2, 0))
        ax.axis('off')

    for idx in range(N, ncol * nrow):
        i, j = idx // nrow, idx % nrow
        axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Save image grid to {path}")

def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_beta_schedule(schedule: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """
    获取 beta 噪声调度

    Args:
        schedule: "linear" 或 "cosine"
        timesteps: 总步数
        beta_start/end: 起始/终止值

    Returns:
        betas: [timesteps]
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == "cosine":
        # Improved DDPM 的余弦调度
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
```

#### 代码细节解释

这里挑选**一部分我感到困惑的内容**来详细说明。

##### 正弦位置编码 `timestep_embedding()`

把整数时间步（如 0, 1, 2, ..., 999）编码为高维向量（如 256 维），让神经网络“感知”当前处于去噪的哪个阶段。

比如第100步与第900步的去噪策略完全不同，因此我们需要把时间步变成一个“特征向量”来告诉模型“现在处于扩散过程的哪个位置”。

这里我们编码的思想参考了 Transformer 的原论文，但用于**连续时间**而不是序列位置。

对于每个时间步，我们会根据一定的规则把它映射到一个向量上。

向量中各个维度映射规则如下所示：

设 `dim=D`，则 `half=D//2`。

1. **频率公式**
   
   $\omega_i=\exp(-\ln(\text{max\_period})\cdot\frac{i}{\text{half}})=\frac{1}{\text{max\_period}^{i/\text{half}}}$
   
   其中，$i \in [0, 1, \dots,\text{half}-1]$
   
   ```python
       half = dim // 2
       freqs = torch.exp(
           -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
       ).to(timesteps.device)
   ```

> 代码中为什么要用 **exp-log 形式**而不是分数？
> 
> 1. 数值更加稳定，分数形式会有边界问题；exp-log 形式全程连续可导，无边界问题。
> 
> 2. 计算效率更高，幂运算通常比 `exp` 更慢。

2. **维度映射通式**
   
   对于输出向量的第 $d$ 维（$d \in [0, 1, \dots, D-1]$）：
   
   $\text { embedding }[d]=\left\{\begin{array}{ll}
    \cos \left(t \cdot \omega_{d}\right) & \text { if } d<\text { half } \\
    \sin \left(t \cdot \omega_{d-\text { half }}\right) & \text { if } d \geq \text { half }
    \end{array}\right.$ 
   
   ```python
       args = timesteps[:, None].float() * freqs[None, :]                  # [B, half]
       embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # [B, dim]
   ```

以 `dim=4` 为例：

| 维度  | 计算公式                     |
| --- | ------------------------ |
| 0   | $\cos(t \cdot \omega_0)$ |
| 1   | $\cos(t \cdot \omega_1)$ |
| 2   | $\sin(t \cdot \omega_0)$ |
| 3   | $\sin(t \cdot \omega_1)$ |

>  值得一提的是，当 dim 为奇数时，我们让最后一个维度不参与编码，即补零。
> 
> 以 `dim=7` 为例：
> 
> | 维度    | 计算公式                     |
> |:-----:|:------------------------ |
> | 0     | $\cos(t \cdot \omega_0)$ |
> | 1     | $\cos(t \cdot \omega_1)$ |
> | 2     | $\cos(t \cdot \omega_2)$ |
> | 3     | $\sin(t \cdot \omega_0)$ |
> | 4     | $\sin(t \cdot \omega_1)$ |
> | 5     | $\sin(t \cdot \omega_2)$ |
> | **6** | **0**                    |
> 
> ```python
>     # 奇数维度时补零
>     if dim % 2:
>         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
> 
> ```

##### `patchify()` 与 `unpatchify()`

这两个函数实现了**像素网络**与**Token序列**之间的转换。

具体细节没什么好说的，我们借此总结一些 Pytorch 的常见方法：

- `unflod()` - 滑动窗口提取
  
  在指定维度上以固定步长提取滑动窗口
  
  ```python
  x.unfold(dimension, size, step)
  ```
  
  ```python
      x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
  ```
  
  在 H, W 维度上展开，窗口大小=步长= `patch_size`
  
  举个例子，把 28×28 图像切成 4×4 的小块，每块独立成维。

- `reshape()` vs `view()`
  
  `reshape()` = `view()` + `contiguous()`
  
  - `view()` 速度更快，但要求内存连续（确定连续时推荐使用）
  
  - `reshape()` 自动处理非连续，必要时拷贝（不确定连续性时）
  
  `view()` 用于改变张量形状，不改变数据内容
  
  ```python
      # [B, N, C*p*p]
      patches = x.view(B, -1, C * patch_size * patch_size)
  ```
  
  > `-1` 让 Pytorch 自动计算该维度大小，避免手动算 N。

- `permute()` - 维度重排
  
  按指定顺序重新排列张量维度。
  
  ```python
      # [B, C, H//p, p, W//p, p] -> [B, H//p, W//p, C, p, p]
      x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
  ```

- `contiguous()` - 内存连续性保证
  
  确保张量在内存中连续存储，使 `view()` 能正常工作。
  
  - `permute()`、`unflod()` 等操作会改变内存布局，导致张量**不连续**
  
  - `view()` 要求连续内存，否则报错

##### 统计模型可训练参数量 `count_parameters()`

还是一样，我们来看看这个函数中 Pytorch 的相关方法与属性

- `numel()` - 元素计数
  
  返回张量中**元素的总个数**（number of elements）

- `parameters()` - 参数迭代器
  
  返回模型**所有可学习参数**的生成器，用于遍历

- `requires_grad` - 梯度追踪标志
  
  标记张量**是否需要计算梯度**（是否参与训练）
  
  | 值       | 含义           | 场景           |
  |:------- |:------------ |:------------ |
  | `True`  | 需要梯度，反向传播时更新 | 可学习参数        |
  | `False` | 冻结，不更新       | 预训练模型冻结、推理模式 |

##### Beta 噪声调度 `get_beta_schedule()`

这个函数生成扩散过程中每一步的噪声强度 $\beta_t$ ，控制“每一步加多少噪声”

函数返回 `betas`：`[timesteps]` 形状的一维张量，每个元素是 0~1 之间的浮点数

`betas[t]` = 第 t 步的噪声方差

这里我们着重看看**余弦调度**的实现：

###### 余弦调度

余弦调度的目标是：

> 让信号衰减速度遵循余弦曲线的平方，前期慢，后期快，形成平滑过渡

1. 定义累计信号比例
   
   定义从原始图像到第 $t$ 步**保留的信号比例**为：
   
   $$
   \overline{\alpha}_t=\cos^2(\frac{t}{T}\cdot\frac{\pi}{2})
   $$
   
   
   
   - $t=0$ ：$\cos^2(0)=1$ ，完全保留信号
   
   - $t=T$：$\cos^2(1)=0$，完全变成噪声
   
   - 中间平缓过渡，曲线形状先缓后陡

2. 引入偏移量防止数值问题
   
   为了防止 $t=0$ 时导数过大，我们**引入偏移量** $s$（代码中取`s=0.008`）：
   
   $$
   \overline{\alpha}_t=\cos^2(\frac{\frac{t}{T}+s}{1+s}\cdot\frac{\pi}{2})/C
   $$
   
   
   
   其中 $C$ 是归一化常数，确保 $\overline{\alpha}_0=1
   
   ```python
   alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
   alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
   ```

3. 从累计信号反推单步噪声
   
   已知
   
   $$
   \overline{\alpha}_t=\prod^t_{i=1}(1-\beta_i)
   $$
   
   > 这是扩散模型的核心，是 $x_0$ 无需迭代，可以直接得出任意 $x_t$ 的理论来源。具体推导见前文（待补充）
   
   则：
   
   $$
   \beta_i=1-\frac{\overline{\alpha}_t}{\overline{\alpha}_{t-1}}
   $$
   
   直观理解一下，每一步的噪声强度 = 1- 这一步相对于上一步保留了多少额外信号
   
   ```python
   betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
   ```

4. 裁剪保护
   
   ```python
   return torch.clip(betas, 0.0001, 0.9999)
   ```
   
   防止极端值导致数值不稳定

### diffusion.py



---

## 参考

- [[2212.09748] Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

- [[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

- 
