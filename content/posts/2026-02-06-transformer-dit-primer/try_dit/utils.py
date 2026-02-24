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