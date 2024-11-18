import warnings
from typing import Tuple

import torch
import torch.nn as nn

from quadrotor_diffusion.utils.nn.schedulers import cosine_beta_schedule
from quadrotor_diffusion.models.losses import MSELoss
from quadrotor_diffusion.models.temporal import Unet1D
from quadrotor_diffusion.utils.nn.args import Unet1DArgs, DiffusionWrapperArgs
from quadrotor_diffusion.utils.logging import dataclass_to_table, iprint as print

# TODO(shreepa): Probably should fix this at some point
# Suppress FutureWarning for this specific issue
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


class DiffusionWrapper(nn.Module):
    def __init__(
        self,
        args: Tuple[DiffusionWrapperArgs, Unet1DArgs],
    ):
        """
        Wrapper that noises sample and computes the loss for a denoising diffusion model
        """
        super().__init__()

        diffusion_args: DiffusionWrapperArgs = args[0]
        unet_args: Unet1DArgs = args[1]
        assert isinstance(diffusion_args, DiffusionWrapperArgs), "diffusion_args must be of type DiffusionWrapperArgs"
        assert isinstance(unet_args, Unet1DArgs), "unet_args must be of type Unet1DArgs"

        self.args = args
        self.diffusion_args = diffusion_args
        self.unet_args = unet_args

        predict_epsilon = diffusion_args.predict_epsilon
        n_timesteps = diffusion_args.n_timesteps
        loss = diffusion_args.loss

        # Model to either predict epsilon or x_t directly
        self.model = Unet1D(unet_args)

        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon

        # Compute Scheduler Conflicts, betas is an array of dimension 1
        betas = cosine_beta_schedule(n_timesteps)
        alpha = 1. - betas
        alpha_bar = torch.cumprod(alpha, axis=0)
        self.register_buffer('alpha_bar', alpha_bar)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        posterior_variance = betas * \
            (1. - alpha_bar_prev) / (1. - alpha_bar)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))

        if loss == "MSELoss":
            self.loss = MSELoss()
        else:
            raise NotImplementedError(f"{loss} loss module is not supported")

        num_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"{num_params:.2f} million parameters")

    def compute_loss(self, x_0) -> torch.Tensor:
        """
        Does a forward pass and computes the loss

        Parameters:
        - x_0: clean trajectories [batch_size x horizon x states]

        Return:
        - loss: Result from loss function
        """

        batch_size = len(x_0)

        # Sample noisy trajectories
        with torch.no_grad():
            # Select random timesteps for each sample in batch
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device).long()
            epsilon = torch.randn_like(x_0)

            # Unsqueeze alpha_bar so it goes from [1] to [1 x 1 x 1] allowing for broadcasting
            alpha_t = self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
            x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * epsilon

        model_output = self.model(x_t, t)
        # TODO(shreepa): add ability to train inpainting by fixing specific points in traj to inpaitned vals

        target = epsilon if self.predict_epsilon else x_0
        loss = self.loss(model_output, target)

        return loss

    @torch.no_grad()
    def sample_unguided(self, batch_size: int, horizon: int, device: str) -> torch.Tensor:
        """
        Samples trajectories from pure noise using the reverse diffusion process

        Parameters:
            batch_size: number of trajectories to generate
            horizon: length of each trajectory
            device: device for pytorch tensors

        Returns:
            x: Generated trajectories of shape [batch_size x horizon x traj_dim]
        """
        # Start from pure noise
        x = torch.randn((batch_size, horizon, self.unet_args.traj_dim), device=device)

        # Iteratively denoise the samples
        for t in reversed(range(0, self.n_timesteps)):
            time_t = torch.ones(batch_size, device=device).long() * t

            # Get model prediction (either epsilon or x_0)
            model_output = self.model(x, time_t)

            alpha_t = self.alpha_bar[t]
            alpha_t_prev = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

            if self.predict_epsilon:
                # If model predicts noise, use it to compute x_0
                pred_epsilon = model_output
                pred_x_0 = (x - torch.sqrt(1 - alpha_t) * pred_epsilon) / torch.sqrt(alpha_t)
            else:
                # Model directly predicts x_0
                pred_x_0 = model_output

            # Compute posterior mean and variance
            posterior_mean = (
                torch.sqrt(alpha_t_prev) *
                (x - torch.sqrt(1 - alpha_t) * model_output / torch.sqrt(alpha_t))
            ) if self.predict_epsilon else pred_x_0

            # Add noise if not the final step
            if t > 0:
                noise = torch.randn_like(x)
                posterior_variance = self.posterior_variance[t]
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = posterior_mean

        return x

    def __str__(self):
        return dataclass_to_table(self.diffusion_args) + "\n" + dataclass_to_table(self.unet_args) + "\n"
