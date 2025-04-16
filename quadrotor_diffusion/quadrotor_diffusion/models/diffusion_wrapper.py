import warnings
import random
import enum
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quadrotor_diffusion.utils.nn.schedulers import cosine_beta_schedule
from quadrotor_diffusion.models.losses import MSELoss, L1Loss, WeightedL1Loss
from quadrotor_diffusion.models.temporal import Unet1D
from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.nn.args import (
    DiffusionWrapperArgs,
    Unet1DArgs,
    LatentDiffusionWrapperArgs,
)
from quadrotor_diffusion.utils.quad_logging import dataclass_to_table, iprint as print

# TODO(shreepa): Probably should fix this at some point
# Suppress FutureWarning for this specific issue
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


class SamplerType(enum.Enum):
    DDPM = "ddpm"
    DDIM = "ddim"
    CONSISTENCY = "consistency"


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

        self.register_buffer('betas', betas)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        posterior_variance = betas * \
            (1. - alpha_bar_prev) / (1. - alpha_bar)

        self.register_buffer('posterior_variance', posterior_variance)

        if loss == "MSELoss":
            self.loss = MSELoss()
        elif loss == "L1Loss":
            self.loss = L1Loss()
        elif loss == "WeightedL1Loss":
            self.loss = WeightedL1Loss(self.args[0].loss_params)
        else:
            raise NotImplementedError(f"{loss} loss module is not supported")

    def compute_loss(self, x_0, **kwargs) -> torch.Tensor:
        """
        Does a forward pass and computes the loss

        Parameters:
        - x_0: clean trajectories [batch_size x horizon x states]
        - kwargs: Capture any other arguments that may not be used

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

        c = None if "conditioning" not in kwargs else kwargs["conditioning"]
        model_output = self.model(x_t, t, c)

        target = epsilon if self.predict_epsilon else x_0
        loss = self.loss(model_output, target)

        return loss

    @torch.no_grad()
    def sample(self, batch_size: int, horizon: int, device: str, conditioning: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Samples trajectories from pure noise using the reverse diffusion process

        Parameters:
            batch_size: number of trajectories to generate
            horizon: length of each trajectory
            device: device for pytorch tensors
            conditioning: If the model is trained to use conditioning pass in the conditioning to the model. Default none = no passed on data
                          If provided first tuple should be conditioning second should be none null
        Returns:
            x: Generated trajectories of shape [batch_size x horizon x traj_dim]
        """
        # Start from pure noise
        x_t = torch.randn((batch_size, horizon, self.unet_args.traj_dim), device=device)

        # Iteratively denoise the samples
        for t in reversed(range(self.n_timesteps)):
            time_t = torch.ones(batch_size, device=device).long() * t
            beta_t = self.betas[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_t_prev = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

            if conditioning is not None:
                x_t_tiled = x_t.repeat(2, 1, 1)
                time_t_tiled = time_t.repeat(2)
                conditioning_0 = conditioning[0][0].repeat(2, 1, 1)
                conditioning_1 = torch.cat((conditioning[0][1], conditioning[1][1]))

                eps = self.model(x_t_tiled, time_t_tiled, (conditioning_0, conditioning_1))
                eps_c = eps[:eps.shape[0]//2, :, :]
                eps_null = eps[eps.shape[0]//2:, :, :]

                w = 0
                model_output = (1 + w) * eps_c - w * eps_null
            else:
                model_output = self.model(x_t, time_t, None)

            # If predicting epsilon reconstruct \hat{x_0} from predicted epsilon
            if self.predict_epsilon:
                x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)

            # Otherwise just use predicted \hat{x_0}
            else:
                x_0_hat = model_output

            # Compute posterior mean
            c1 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1.0 - alpha_bar_t)
            c2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_t_prev)) / (1.0 - alpha_bar_t)
            posterior_mean = c1 * x_0_hat + c2 * x_t

            # Add noise if not the final step
            if t > 0:
                noise = torch.randn_like(x_t)
                posterior_variance = self.posterior_variance[t]
                x_t = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x_t = posterior_mean

        return x_t.detach()

    @torch.no_grad()
    def sample_ddim(self,
                    batch_size: int,
                    horizon: int,
                    device: str,
                    S: int,
                    conditioning: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
                    ) -> torch.Tensor:
        """
        Sample with DDIM

        Args:
            batch_size (int): Number of samples to generate
            horizon (int): Length of the trajectory
            device (str): device
            S (int): Number of steps to take
            conditioning: If the model is trained to use conditioning pass in the conditioning to the model. Default none = no passed on data
                          If provided first tuple should be conditioning second should be none null

        Returns:
            torch.Tensor: Trajectories
        """

        # Start from pure noise
        x_t = torch.randn((batch_size, horizon, self.unet_args.traj_dim), device=device)

        # https://arxiv.org/pdf/2305.08891

        # Leading (from stable diffusion) <Performs best on this dataset>
        timesteps = np.arange(0, self.n_timesteps - 1, int(self.n_timesteps / S))
        timesteps = np.append(np.array([-1]), timesteps)

        # Linspace iDDPM
        # timesteps = np.round(np.linspace(0, self.n_timesteps - 1, S)).astype(int)
        # timesteps = np.append(np.array([-1]), timesteps)

        # Trailing DPM
        # timesteps = np.round(np.flip(np.arange(self.n_timesteps - 1, -1, -self.n_timesteps / S))).astype(int)
        # timesteps = np.append(np.array([-1]), timesteps)

        # Iteratively denoise the samples
        for t, t_prev in zip(reversed(timesteps), reversed(timesteps[:-1])):
            time_t = torch.ones(batch_size, device=device).long() * t
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_t_prev = self.alpha_bar[t_prev] if t > 0 else torch.tensor(1.0, device=device)

            if conditioning is not None:
                x_t_tiled = x_t.repeat(2, 1, 1)
                time_t_tiled = time_t.repeat(2)
                conditioning_0 = conditioning[0][0].repeat(2, 1, 1)
                conditioning_1 = torch.cat((conditioning[0][1], conditioning[1][1]))

                eps = self.model(x_t_tiled, time_t_tiled, (conditioning_0, conditioning_1))
                eps_c = eps[:eps.shape[0]//2, :, :]
                eps_null = eps[eps.shape[0]//2:, :, :]

                w = 0
                model_output = (1 + w) * eps_c - w * eps_null
            else:
                model_output = self.model(x_t, time_t, None)

            # If predicting epsilon reconstruct \hat{x_0} from predicted epsilon
            if self.predict_epsilon:
                x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)

            # Otherwise just use predicted \hat{x_0}
            else:
                raise NotImplementedError("whoops didn't implement ddim for x_theta model")

            # Compute posterior mean
            c1 = torch.sqrt(alpha_bar_t_prev)
            c2 = torch.sqrt(1 - alpha_bar_t_prev)
            x_t = c1 * x_0_hat + c2 * model_output

        return x_t.detach()


class LatentDiffusionWrapper(nn.Module):
    def __init__(
        self,
        args: Tuple[
            LatentDiffusionWrapperArgs,
            Unet1DArgs,
        ],
    ):
        super().__init__()

        self.args = args
        self.diffusion = DiffusionWrapper((
            DiffusionWrapperArgs(
                predict_epsilon=args[0].predict_epsilon,
                loss=args[0].loss,
                loss_params=args[0].loss_params,
                n_timesteps=args[0].n_timesteps
            ),
            self.args[1],
        ))

        self.null_token_local: torch.Tensor
        self.register_buffer("null_token_local", 5 * torch.ones(args[1].conditioning[0]).reshape((1, 1, -1)))

        self.null_token_global: torch.Tensor
        self.register_buffer("null_token_global", 5 * torch.ones(args[1].conditioning[1]).reshape((1, 1, -1)))

        self.encoder: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None
        self.decoder: Callable[[torch.Tensor], torch.Tensor] = None

    def compute_loss(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Compute loss for the given trajectory
        Args:
            batch:
                - x_0 (torch.Tensor): [B, Horizon, 3]
                - conditioning (torch.Tensor): [B, 7, 4]
            kwargs (dict): Capture any additional arguments

        Returns:
            torch.Tensor: Loss value
        """
        assert self.encoder is not None, "Encoder has not been set"

        # [B, Horizon // 4, VAE_latent_dim]
        latent_trajectory, _ = self.encoder(batch["x_0"])

        batch_size = latent_trajectory.shape[0]
        mask = torch.rand(batch_size, device=latent_trajectory.device) < self.args[0].dropout
        batch["global_conditioning"][mask] = self.null_token_global.expand((
            -1, batch["global_conditioning"].shape[1], -1
        ))
        # mask = torch.rand(batch_size, device=latent_trajectory.device) < self.args[0].dropout
        # batch["local_conditioning"][mask] = self.null_token_local.expand((
        #     -1, batch["local_conditioning"].shape[1], -1
        # ))

        horizon_downsample_factor = 2 ** (len(self.args[1].channel_mults)-1)
        horizon_modulo = latent_trajectory.shape[-2] % horizon_downsample_factor
        assert horizon_modulo == 0, f"Invalid input data not divisible has module: {horizon_modulo}"

        return self.diffusion.compute_loss(latent_trajectory, conditioning=(batch["local_conditioning"], batch["global_conditioning"]))

    @torch.no_grad()
    def sample(self,
               batch_size: int,
               horizon: int,
               vae_downsample: int,
               device: str,
               local_conditioning: torch.Tensor,
               global_conditioning: torch.Tensor,
               sampler: SamplerType,
               ) -> torch.Tensor:
        """
        1. Samples latents from pure noise using the reverse diffusion process.
        2. Decodes latents using the VAE decoder.

        Parameters:
            batch_size: number of trajectories to generate
            horizon: length of each trajectory. must be divisible by 4 (VAE) * 8 (diffusion) = 32
            vae_downsample: What factor VAE downsample the trajectory
            device: device for pytorch tensors
            local_conditioning: Set of waypoints given as [batch_size x 6  x 3] where the center is the initial state
            global_conditioning: Set of waypoints given as [batch_size x 4  x 4]
            sampler: SamplerType
        Returns:
            x: Generated trajectories of shape [batch_size x horizon x traj_dim]
        """
        assert self.decoder is not None, "Need to attach a decoder"

        diffusion_downsample = 2 ** (len(self.args[1].channel_mults) - 1)
        assert horizon % (vae_downsample * diffusion_downsample) == 0

        latent_horizon = horizon // vae_downsample

        conditioning = (local_conditioning, global_conditioning)
        null_conditioning = (
            self.null_token_local.expand((batch_size, local_conditioning.shape[1], -1)),
            self.null_token_global.expand((batch_size, global_conditioning.shape[1], -1))
        )

        if sampler == SamplerType.DDPM:
            latent_trajectories = self.diffusion.sample(
                batch_size, latent_horizon, device, conditioning=(conditioning, null_conditioning)
            )
        elif sampler == SamplerType.DDIM:
            latent_trajectories = self.diffusion.sample_ddim(
                batch_size, latent_horizon, device, 4, conditioning=(conditioning, null_conditioning)
            )
        else:
            raise ValueError(f"Sampler {sampler} not implemented.")

        trajectories = self.decoder(latent_trajectories)

        return trajectories
