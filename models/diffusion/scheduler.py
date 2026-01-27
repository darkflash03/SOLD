import torch
import torch.nn as nn
import numpy as np
def cosine_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

class NoiseScheduler(nn.Module):
    def __init__(self, timesteps):
        super(NoiseScheduler, self).__init__()
        self.timesteps = torch.tensor(timesteps)
        betas = cosine_beta_schedule_discrete(self.timesteps)
        alphas = 1 - torch.clamp(betas, min=0, max=0.9999)
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        alpha_bars = torch.exp(log_alpha_bar)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * (1 - alphas[i])
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas', sigmas)