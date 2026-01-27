import torch
import torch.nn as nn
from models.diffusion.scheduler import NoiseScheduler
import math

class LatentTransition(nn.Module):
    def __init__(self, num_steps):
        super(LatentTransition, self).__init__()
        self.noise_scheduler = NoiseScheduler(num_steps)
        self.num_steps = num_steps

    @torch.no_grad()
    def add_noise(self, latent_t0, t):
        alpha_bar = self.noise_scheduler.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        e_rand = torch.randn_like(latent_t0)
        latent_noisy = c0 * latent_t0 + c1 * e_rand
        return latent_noisy

    @torch.no_grad()
    def ddpm_step(self, latent_t, pred_latent_t0, t):
        alpha_t = self.noise_scheduler.alphas[t].clamp_min(
            self.noise_scheduler.alphas[-2]
        )
        alpha_bar_t = self.noise_scheduler.alpha_bars[t]
        alpha_bar_s = self.noise_scheduler.alpha_bars[t - 1]
        sigma = self.noise_scheduler.sigmas[t].view(-1, 1)

        z = torch.randn_like(latent_t)
        latent_s = (torch.sqrt(alpha_t) * (1 - alpha_bar_s) * latent_t + torch.sqrt(alpha_bar_s) * (
                    1 - alpha_t) * pred_latent_t0) / (1 - alpha_bar_t + 1e-8) + sigma * z
        return latent_s

    @torch.no_grad()
    def ddim_step(self, latent_t, pred_latent_t0, t, s, eta):
        alpha_bar_t = self.noise_scheduler.alpha_bars[t]
        alpha_bar_s = self.noise_scheduler.alpha_bars[s]
        sigma_t = eta * torch.sqrt((1 - alpha_bar_s) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_s))
        pred_noise = (latent_t - torch.sqrt(alpha_bar_t) * pred_latent_t0) / torch.sqrt(1 - alpha_bar_t + 1e-8)
        latent_s = (torch.sqrt(alpha_bar_s) * pred_latent_t0 +
                    torch.sqrt(1 - alpha_bar_s - sigma_t ** 2) * pred_noise +
                    sigma_t * torch.randn_like(latent_t))

        return latent_s

    def step_with_logprob(self, prev_latent_t, pred_latent_t0, t, latent_t=None):
        alpha_t = self.noise_scheduler.alphas[t].clamp_min(self.noise_scheduler.alphas[-2]).view(-1, 1, 1)
        alpha_bar_t = self.noise_scheduler.alpha_bars[t].view(-1, 1, 1)
        alpha_bar_s = self.noise_scheduler.alpha_bars[t - 1].view(-1, 1, 1)
        sigma = self.noise_scheduler.sigmas[t].view(-1, 1, 1)

        pred_mean = (torch.sqrt(alpha_t) * (1 - alpha_bar_s) * prev_latent_t + torch.sqrt(alpha_bar_s) * (
                1 - alpha_t) * pred_latent_t0) / (1 - alpha_bar_t + 1e-8)

        if latent_t is None:
            z = torch.randn_like(prev_latent_t)
            latent_t = pred_mean + sigma * z

        log_prob = (
                -0.5 * ((latent_t - pred_mean) ** 2 / (sigma ** 2 + 1e-8))
                - torch.log(sigma + 1e-8)
                - 0.5 * torch.log(2 * torch.as_tensor(math.pi))
        )
        log_prob = log_prob.mean(dim=(-2, -1))
        return latent_t, log_prob

    def ddim_step_with_logprob(self, prev_latent_t, pred_latent_t0, t, s, eta):
        alpha_bar_t = self.noise_scheduler.alpha_bars[t].view(-1, 1, 1)
        alpha_bar_s = self.noise_scheduler.alpha_bars[s].view(-1, 1, 1)
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_s) / (1 - alpha_bar_t + 1e-8) * (1 - alpha_bar_t / (alpha_bar_s + 1e-8))
        )

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
        sqrt_one_minus_alpha_bar_t = torch.sqrt((1 - alpha_bar_t).clamp(min=1e-8))
        pred_epsilon = (prev_latent_t - sqrt_alpha_bar_t * pred_latent_t0) / sqrt_one_minus_alpha_bar_t

        pred_direction = torch.sqrt((1 - alpha_bar_s - sigma_t ** 2).clamp(min=1e-8)) * pred_epsilon
        pred_mean = torch.sqrt(alpha_bar_s) * pred_latent_t0 + pred_direction
        z = torch.randn_like(prev_latent_t)
        latent_s = pred_mean + sigma_t * z

        log_prob = (
                -0.5 * ((latent_s.detach() - pred_mean) ** 2 / (sigma_t ** 2 + 1e-8))
                - torch.log(sigma_t + 1e-8)
                - 0.5 * torch.log(2 * torch.as_tensor(math.pi))
        )
        log_prob = log_prob.mean(dim=(-2, -1))
        return latent_s, log_prob