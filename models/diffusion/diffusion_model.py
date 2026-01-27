import torch
import torch.nn as nn
from models.diffusion.latent_transition import LatentTransition

class DiffusionModel(nn.Module):
    def __init__(self, model, latent_transition_config):
        super(DiffusionModel, self).__init__()
        self.model = model
        self.latent_transition = LatentTransition(**latent_transition_config)
        self.num_steps = self.latent_transition.num_steps

    def add_noise(self, data, t):
        batch_size = t.shape[0]
        length = data.x.shape[0] // batch_size
        latent = data.x.reshape(batch_size, length, -1)
        noisy_X = self.latent_transition.add_noise(latent, t)
        noisy_data = data.clone()
        noisy_data.x = noisy_X.reshape(-1, latent.shape[-1])
        return noisy_data

    def forward(self, data):
        t = torch.randint(0, self.latent_transition.num_steps + 1, size=(data.batch[-1] + 1, 1), device=data.x.device)
        noisy_data = self.add_noise(data, t)
        pred_latent = self.model(noisy_data, t)
        return pred_latent, t

    @torch.no_grad()
    def ddpm_sample(self, data):
        latent_t = torch.randn_like(data.x, device=data.x.device)
        for t in range(self.num_steps, 0, -1):
            sample_t = torch.tensor([[t]], dtype=torch.int32).to(data.x.device)
            noisy_data = data.clone()
            noisy_data.x = latent_t
            pred_latent = self.model(noisy_data, sample_t)
            latent_t = self.latent_transition.ddpm_step(latent_t, pred_latent, t)
        return latent_t

    @torch.no_grad()
    def ddim_sample(self, data, num_steps=None, eta=1.0):
        if num_steps is None:
            num_steps = self.num_steps

        indices = torch.linspace(0, self.num_steps, steps=num_steps + 1, device=data.x.device).long()
        timesteps = self.num_steps - indices
        timesteps = torch.clamp(timesteps, min=0)
        timesteps = timesteps[:-1]
        latent_t = torch.randn_like(data.x, device=data.x.device)

        for i in range(len(timesteps)):
            t = timesteps[i]
            s = 0 if i == len(timesteps) - 1 else timesteps[i + 1]  # 最后一步到 t=0
            sample_t = torch.tensor([[t]], dtype=torch.int32, device=data.x.device)
            noisy_data = data.clone()
            noisy_data.x = latent_t
            pred_latent_t0 = self.model(noisy_data, sample_t)
            latent_t = self.latent_transition.ddim_step(latent_t, pred_latent_t0, t, s, eta)

        return latent_t

    @torch.no_grad()
    def sample_with_logprob(self, data):
        batch_size = data.batch[-1] + 1
        length = data.x.shape[0] // batch_size
        latent_dim = data.x.shape[-1]

        # Initialize with proper shapes
        latent_init = torch.randn(batch_size * length, latent_dim, device=data.x.device)
        traj = {self.num_steps: latent_init.view(batch_size, length, -1)}
        logprob_traj = {}

        for t in range(self.num_steps, 0, -1):
            # Create time tensor for full batch
            sample_t = torch.full((batch_size, 1), t, dtype=torch.int32, device=data.x.device)

            # Get current latent and reshape for model
            latent_t = traj[t].view(-1, data.x.shape[-1])
            noisy_data = data.clone()
            noisy_data.x = latent_t

            # Get model prediction
            pred_latent = self.model(noisy_data, sample_t)

            # Reshape tensors for transition step
            latent_t_reshaped = latent_t.view(batch_size, length, -1)
            pred_latent_reshaped = pred_latent.view(batch_size, length, -1)

            # Compute transition and log probability
            prev_latent, logprob = self.latent_transition.step_with_logprob(
                latent_t_reshaped,
                pred_latent_reshaped,
                t
            )

            # Store results
            logprob_traj[t - 1] = logprob
            traj[t - 1] = prev_latent
        return traj, logprob_traj

    @torch.no_grad()
    def ddim_sample_with_logprob(self, data, num_steps=None, eta=1.0):
        if num_steps is None:
            num_steps = self.num_steps

        batch_size = data.batch[-1] + 1
        length = data.x.shape[0] // batch_size
        latent_dim = data.x.shape[-1]
        indices = torch.linspace(0, self.num_steps, steps=num_steps + 1, device=data.x.device).long()
        timesteps = self.num_steps - indices
        timesteps = list(torch.clamp(timesteps, min=0).cpu().numpy())
        latent_init = torch.randn(batch_size * length, latent_dim, device=data.x.device)
        traj = {timesteps[0]: latent_init.view(batch_size, length, -1)}
        logprob_traj = {}

        # ddim reverse sampling
        for i in range(len(timesteps)):
            t = timesteps[i]
            s = 0 if i == len(timesteps) - 1 else timesteps[i + 1]
            sample_t = torch.full((batch_size, 1), t, dtype=torch.int32, device=data.x.device)
            latent_t = traj[t].view(-1, latent_dim)
            noisy_data = data.clone()
            noisy_data.x = latent_t

            pred_latent = self.model(noisy_data, sample_t)
            latent_t_reshaped = latent_t.view(batch_size, length, -1)
            pred_latent_reshaped = pred_latent.view(batch_size, length, -1)

            # Compute transition and log probability
            prev_latent, logprob = self.latent_transition.ddim_step_with_logprob(
                latent_t_reshaped,
                pred_latent_reshaped,
                t,
                s,
                eta
            )

            # Store results
            logprob_traj[s] = logprob
            traj[s] = prev_latent
        return traj, logprob_traj

    def step_with_logprob(self, data, prev_latent_t, t, latent_t=None):
        batch_size = data.batch[-1] + 1
        length = data.x.shape[0] // batch_size
        latent_dim = data.x.shape[-1]

        # Reshape inputs for model
        noisy_data = data.clone()
        noisy_data.x = prev_latent_t.contiguous().view(batch_size * length, latent_dim)

        # Create proper time tensor for batch
        if t.size(0) == 1:
            sample_t = torch.full((batch_size, 1), t, dtype=torch.int32, device=data.x.device)
        else:
            sample_t = t

        # Get model prediction
        pred_latent = self.model(noisy_data, sample_t)

        # Reshape for transition step
        prev_latent_reshaped = prev_latent_t.view(batch_size, length, -1)
        pred_latent_reshaped = pred_latent.view(batch_size, length, -1)
        latent_t_reshaped = None if latent_t is None else latent_t.view(batch_size, length, -1)

        # Compute transition and log probability
        prev_latent, logprob = self.latent_transition.step_with_logprob(
            prev_latent_reshaped,
            pred_latent_reshaped,
            t,
            latent_t=latent_t_reshaped
        )

        return prev_latent, logprob

    def ddim_step_with_logprob(self, data, prev_latent_t, t, s, eta=1.0, latent_t=None):
        batch_size = data.batch[-1] + 1
        length = data.x.shape[0] // batch_size
        latent_dim = data.x.shape[-1]

        noisy_data = data.clone()
        noisy_data.x = prev_latent_t.contiguous().view(batch_size * length, latent_dim)

        if t.size(0) == 1:
            sample_t = torch.full((batch_size, 1), t, dtype=torch.int32, device=data.x.device)
        else:
            sample_t = t

        pred_latent = self.model(noisy_data, sample_t)
        prev_latent_reshaped = prev_latent_t.view(batch_size, length, -1)
        pred_latent_reshaped = pred_latent.view(batch_size, length, -1)
        latent_t_reshaped = None if latent_t is None else latent_t.view(batch_size, length, -1)

        # Compute transition and log probability
        prev_latent, logprob = self.latent_transition.ddim_step_with_logprob(
            latent_t_reshaped,
            pred_latent_reshaped,
            t,
            s,
            eta
        )
        return prev_latent, logprob