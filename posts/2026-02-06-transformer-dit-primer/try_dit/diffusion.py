import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from tqdm import tqdm

from utils import get_beta_schedule, timestep_embedding
from config import DiffusionConfig, TrainConfig

class Diffusion:
    """
    DDPM 扩散过程：封装前向加噪、训练损失计算、反向采样
    """

    def __init__(
            self,
            config: DiffusionConfig,
            device: Optional[str] = None
    ):
        self.config = config
        self.device = device if device else config.device

        timesteps = config.timesteps
        beta_start = config.beta_start
        beta_end = config.beta_end
        schedule = config.schedule

        # 获取 beta 调度
        self.betas = get_beta_schedule(schedule, timesteps, beta_start, beta_end).to(self.device)

        # 计算 alpha 相关变量
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 预计算平方根等中间量，避免重复计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)

        # 数值稳定性裁剪
        self.posterior_variance = torch.clip(self.posterior_variance, min=1e-20)

    def q_sample(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散：从 x_0 采样得到 x_t（重参数化技巧）
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 提取对应时间步的系数
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
    
    def predict_start_from_noise(
            self, 
            x_t: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor
    ) -> torch.Tensor:
        """
        从预测的噪声反推 x_0
        """
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # x_0 = (x_t - √(1-ᾱ_t)*ε) / √ᾱ_t
        return (x_t - sqrt_one_minus_alpha_cumprod_t * noise) / sqrt_alpha_cumprod_t
    
    def q_posterior_mean_variance(
            self,
            x_start: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算真实后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)

        # 后验均值系数
        coef_x0 = torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t)
        coef_xt = torch.sqrt(self.alphas[t].view(-1, 1, 1, 1)) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

        posterior_mean = coef_x0 * x_start + coef_xt * x_t
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)

        return posterior_mean, posterior_variance
    
    def p_losses(
            self,
            model: torch.nn.Module,
            x_start: torch.Tensor,
            t: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算训练损失（预测噪声的 MSE）
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 前向加噪得到 x_t
        x_t = self.q_sample(x_start, t, noise)

        # 模型预测噪声
        noise_pred = model(x_t, t, y)

        # MSE 损失
        loss = F.mse_loss(noise_pred, noise)

        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: int,
        y: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        单步去噪：从 x_t 采样 x_{t-1}
        """
        # 使用配置默认值或传入值
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale

        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size, ), t, device=self.device, dtype=torch.long)

        # Classifier-free guidance
        if cfg_scale > 1.0 and y is not None:
            # 无条件预测
            noise_uncond = model(x_t, t_batch, None)
            # 有条件预测
            noise_cond = model(x_t, t_batch, y)
            # 外推
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x_t, t_batch, y)

        # 计算预测的 x_0
        x_0_pred = self.predict_start_from_noise(x_t, t_batch, noise_pred)

        # 使用配置中的裁剪范围（可扩展）
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        # 后验均值
        alpha_cumprod_t = self.alphas_cumprod[t].item()
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].item()
        beta_t = self.betas[t].item()

        coef_x0 = math.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t)
        coef_xt = math.sqrt(self.alphas[t].item()) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

        mean = coef_x0 * x_0_pred + coef_xt * x_t

        # 采样方差
        if t > 0:
            variance = self.posterior_variance[t].item()
            noise = torch.randn_like(x_t)
            x_prev = mean + math.sqrt(variance) * noise
        else:
            x_prev = mean    # t=0 时不加噪声

        return x_prev
    
    @torch.no_grad()
    def sample(
            self,
            model: torch.nn.Module,
            shape: Tuple[int, int, int, int],
            y: Optional[torch.Tensor] = None,
            cfg_scale: Optional[float] = None,
            progress: bool = True
    ) -> torch.Tensor:
        """
        完整采样：从纯噪声生成图像
        """
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale
        batch_size, C, H, W = shape

        # 从纯噪声开始
        x = torch.randn(shape, device=self.device)

        # 设置条件（随机类别如果未提供）
        if y is None:
            # 从配置获取类别数，避免硬编码
            num_classes = getattr(model.config, 'num_classes', 10)
            y = torch.randint(0, num_classes, (batch_size,), device=self.device)
        
        # 迭代去噪
        timesteps = range(self.config.timesteps -1, -1, -1)
        if progress: 
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t in timesteps:
            x = self.p_sample(model, x, t, y, cfg_scale)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(
       self,
       model: torch.nn.Module,
       shape: Tuple[int, int, int, int],
       y: Optional[torch.Tensor] = None,
       steps: Optional[int] = None,
       eta: Optional[float] = None,
       cfg_scale: Optional[float] = None,
       progress: bool = True 
    ) -> torch.Tensor:
        """
        DDIM 加速采样（确定性，步数更少）
        """
        # 使用配置或传入值
        steps = steps if steps is not None else self.config.sample_steps
        eta = eta if eta is not None else 0.0   # 默认确定性
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale

        batch_size = shape[0]

        # 均匀选取时间步
        c = self.config.timesteps // steps
        timesteps = list(range(0, self.config.timesteps, c))[:steps]
        timesteps = list(reversed(timesteps)) + [0]

        x = torch.randn(shape, device=self.device)
        if y is None:
            num_classes = getattr(model.config, 'num_classes', 10)
            y = torch.randint(0, num_classes, (batch_size,), device=self.device)

        iterator = tqdm(timesteps[:-1], desc="DDIM Sampling") if progress else timesteps[:-1]

        for i, t in enumerate(iterator):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # CFG
            if cfg_scale > 1.0:
                noise_uncond = model(x, t_batch, None)
                noise_cond = model(x, t_batch, y)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(x, t_batch, y)
            
            # 预测 x_0
            alpha_cumprod_t = self.alphas_cumprod[t]

            if i + 1 < len(timesteps) - 1:  # 还有下一步
                alpha_cumprod_prev = self.alphas_cumprod[timesteps[i+1]]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=self.device) 

            x_0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            # DDIM 方向
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev))

            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma_t**2) * noise_pred

            x = torch.sqrt(alpha_cumprod_prev) * x_0_pred + dir_xt            
            
            if eta > 0 and i < len(timesteps) - 1:
                x = x + sigma_t * torch.randn_like(x)
            
        return x