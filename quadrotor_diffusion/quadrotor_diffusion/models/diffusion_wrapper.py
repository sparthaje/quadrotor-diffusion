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
    GAMMA_CTM = "gamma_ctm"


class DiffusionWrapper(nn.Module):
    def __init__(
        self,
        args: Tuple[DiffusionWrapperArgs, Unet1DArgs],
    ):
        """
        Wrapper that noises sample and computes the loss for a denoising diffusion model (discrete diffusion steps)
        """
        super().__init__()

        diffusion_args: DiffusionWrapperArgs = args[0]
        unet_args: Unet1DArgs = args[1]
        assert isinstance(diffusion_args, DiffusionWrapperArgs), "diffusion_args must be of type DiffusionWrapperArgs"
        assert isinstance(unet_args, Unet1DArgs), "unet_args must be of type Unet1DArgs"

        self.args = args
        self.diffusion_args = diffusion_args
        self.unet_args = unet_args

        n_timesteps = diffusion_args.n_timesteps
        loss = diffusion_args.loss

        # Model to either predict epsilon or x_t directly
        self.model = Unet1D(unet_args)

        self.n_timesteps = n_timesteps

        # Compute Scheduler Conflicts, betas is an array of dimension 1
        clip = not diffusion_args.predict == "v"
        betas = cosine_beta_schedule(n_timesteps, clip=clip)

        alpha = 1. - betas
        alpha_bar = torch.cumprod(alpha, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        posterior_variance = betas * \
            (1. - alpha_bar_prev) / (1. - alpha_bar)

        self.register_buffer('posterior_variance', posterior_variance)

        self.null_token_local: torch.Tensor
        self.register_buffer("null_token_local", 5 * torch.ones(args[1].conditioning[0]).reshape((1, 1, -1)))

        self.null_token_global: torch.Tensor
        self.register_buffer("null_token_global", 5 * torch.ones(args[1].conditioning[1]).reshape((1, 1, -1)))

        self.encoder: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None
        self.decoder: Callable[[torch.Tensor], torch.Tensor] = None

        if loss == "MSELoss":
            self.loss = MSELoss()
        elif loss == "L1Loss":
            self.loss = L1Loss()
        elif loss == "WeightedL1Loss":
            self.loss = WeightedL1Loss(self.args[0].loss_params)
        else:
            raise NotImplementedError(f"{loss} loss module is not supported")

    def compute_loss(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Compute loss for the given trajectory
        Args:
            batch:
                - x_0 (torch.Tensor): [B, Horizon, 3]
                - global_conditioning (torch.Tensor): [B, 4, 4]
                - local_conditioning (torch.Tensor): [B, 6, 3]
            kwargs (dict): Capture any additional arguments

        Returns:
            torch.Tensor: Loss value
        """

        x_0 = batch["x_0"]
        if self.encoder:
            x_0, _ = self.encoder(x_0)

        batch_size = x_0.shape[0]
        mask = torch.rand(batch_size, device=x_0.device) < self.args[0].dropout
        batch["global_conditioning"][mask] = self.null_token_global.expand((
            -1, batch["global_conditioning"].shape[1], -1
        ))

        horizon_downsample_factor = 2 ** (len(self.args[1].channel_mults)-1)
        horizon_modulo = x_0.shape[-2] % horizon_downsample_factor
        assert horizon_modulo == 0, f"Invalid input data not divisible has module: {horizon_modulo}"

        # Sample noisy trajectories
        with torch.no_grad():
            # Select random timesteps for each sample in batch
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device).long()
            epsilon = torch.randn_like(x_0)

            # Unsqueeze alpha_bar so it goes from [1] to [1 x 1 x 1] allowing for broadcasting
            alpha_t = self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
            x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * epsilon

        model_output = self.model(x_t, t, (batch["local_conditioning"], batch["global_conditioning"]))

        target = None
        if self.args[0].predict == "epsilon":
            target = epsilon
        elif self.args[0].predict == "x":
            target = x_0
        elif self.args[0].predict == "v":
            target = torch.sqrt(alpha_t) * epsilon - torch.sqrt(1 - alpha_t) * x_0

        return self.loss(model_output, target)

    def model_out_cfg(self, x_t: torch.Tensor, time_t: torch.Tensor, local_conditioning: torch.Tensor, global_conditioning: torch.Tensor, w: float):
        """
        eps = (1 + w) eps_conditioned - w * eps_null

        Args:
            x_t (torch.Tensor): _description_
            time_t (torch.Tensor): _description_
            local_conditioning (torch.Tensor): _description_
            global_conditioning (torch.Tensor): _description_
        """
        # w = 0 means no CFG so just return model output
        if w < 1e-2:
            return self.model(x_t, time_t, (local_conditioning, global_conditioning))

        x_t_tiled = x_t.repeat(2, 1, 1)
        time_t_tiled = time_t.repeat(2)

        c_local = local_conditioning.repeat(2, 1, 1)
        c_global = torch.cat((
            global_conditioning,
            self.null_token_global.expand((global_conditioning.shape[0], global_conditioning.shape[1], -1))
        ))

        eps = self.model(x_t_tiled, time_t_tiled, (c_local, c_global))
        eps_c = eps[:eps.shape[0]//2, :, :]
        eps_null = eps[eps.shape[0]//2:, :, :]

        return (1 + w) * eps_c - w * eps_null

    @torch.no_grad()
    def sample_ddpm(self,
                    batch_size: int,
                    horizon: int,
                    device: str,
                    local_conditioning: torch.Tensor,
                    global_conditioning: torch.Tensor,
                    w: float) -> torch.Tensor:
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

            model_output = self.model_out_cfg(x_t, time_t, local_conditioning, global_conditioning, w)

            # If predicting epsilon reconstruct \hat{x_0} from predicted epsilon
            if self.args[0].predict == "epsilon":
                x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)

            elif self.args[0].predict == "v":
                x_0_hat = torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1 - alpha_bar_t) * model_output

            # Otherwise just use predicted \hat{x_0}
            elif self.args[0].predict == "x":
                x_0_hat = model_output

            else:
                raise ValueError("Predict should be epsilon, x or v")

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
                    local_conditioning: torch.Tensor,
                    global_conditioning: torch.Tensor,
                    w: float,
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

        # Just predict x_0_hat and call it a day
        if S == 1:
            t = 50
            time_t = torch.ones(batch_size, device=device).long() * t
            alpha_bar_t = self.alpha_bar[t]

            # Step 3: Get model output
            model_output = self.model_out_cfg(x_t, time_t, local_conditioning, global_conditioning, w)

            # Step 4: Reconstruct x_0 based on prediction type
            if self.args[0].predict == "epsilon":
                x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
            elif self.args[0].predict == "v":
                x_0_hat = torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1 - alpha_bar_t) * model_output
            elif self.args[0].predict == "x":
                x_0_hat = model_output
            else:
                raise NotImplementedError("Unknown prediction type")

            return x_0_hat.detach()

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

            model_output = self.model_out_cfg(x_t, time_t, local_conditioning, global_conditioning, w)

            # Predict \hat{x_0}
            if self.args[0].predict == "epsilon":
                epsilon_hat = model_output
                x_0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
            elif self.args[0].predict == "v":
                epsilon_hat = torch.sqrt(alpha_bar_t) * model_output + torch.sqrt(1 - alpha_bar_t) * x_t
                x_0_hat = torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1 - alpha_bar_t) * model_output
            elif self.args[0].predict == "x":
                x_0_hat = model_output
                epsilon_hat = (x_t - torch.sqrt(alpha_bar_t) * x_0_hat) / torch.sqrt(1 - alpha_bar_t)

            # Otherwise just use predicted \hat{x_0}
            else:
                raise NotImplementedError("whoops didn't implement ddim")

            # Compute posterior mean
            c1 = torch.sqrt(alpha_bar_t_prev)
            c2 = torch.sqrt(1 - alpha_bar_t_prev)
            x_t = c1 * x_0_hat + c2 * epsilon_hat

        return x_t.detach()

    @torch.no_grad()
    def sample(self,
               batch_size: int,
               horizon: int,
               device: str,
               local_conditioning: torch.Tensor,
               global_conditioning: torch.Tensor,
               sampler: tuple[SamplerType, int],
               decoder_downsample: int = 1,
               w: float = 0.0
               ) -> torch.Tensor:
        """
        1. Samples latents from pure noise using the reverse diffusion process.
        2. Decodes latents using the VAE decoder.

        Parameters:
            batch_size: number of trajectories to generate
            horizon: length of each trajectory. must be divisible by 4 (VAE) * 8 (diffusion) = 32
            device: device for pytorch tensors
            local_conditioning: Set of waypoints given as [batch_size x 6  x 3] where the center is the initial state
            global_conditioning: Set of waypoints given as [batch_size x 4  x 4]
            sampler: SamplerType and number of steps
            decoder_downsample: What factor VAE downsample the trajectory (1 == no decoder)
            w: Classifier free guidance weight
        Returns:
            x: Generated trajectories of shape [batch_size x horizon x traj_dim]
        """

        diffusion_downsample = 2 ** (len(self.args[1].channel_mults) - 1)
        assert horizon % (decoder_downsample * diffusion_downsample) == 0

        sample_horizon = horizon // decoder_downsample

        if sampler[0] == SamplerType.DDPM:
            trajectories = self.sample_ddpm(
                batch_size, sample_horizon, device, local_conditioning, global_conditioning, w
            )
        elif sampler[0] == SamplerType.DDIM:
            trajectories = self.sample_ddim(
                batch_size, sample_horizon, device, sampler[1], local_conditioning, global_conditioning, w
            )
        else:
            raise ValueError(f"Sampler {sampler} not implemented.")

        if self.decoder:
            trajectories = self.decoder(trajectories)

        return trajectories


class ConsistencyTrajectoryWrapper(nn.Module):
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
        # assert isinstance(diffusion_args, DiffusionWrapperArgs), "diffusion_args must be of type DiffusionWrapperArgs"
        assert isinstance(unet_args, Unet1DArgs), "unet_args must be of type Unet1DArgs"

        self.args = args
        self.diffusion_args = diffusion_args
        self.unet_args = unet_args

        assert self.diffusion_args.predict == "g"
        assert self.unet_args.context_mlp == "time=t,s"

        loss = diffusion_args.loss

        # Model to either predict epsilon or x_t directly
        self.model = Unet1D(unet_args)

        self.null_token_local: torch.Tensor
        self.register_buffer("null_token_local", 5 * torch.ones(args[1].conditioning[0]).reshape((1, 1, -1)))

        self.null_token_global: torch.Tensor
        self.register_buffer("null_token_global", 5 * torch.ones(args[1].conditioning[1]).reshape((1, 1, -1)))

        self.encoder: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None
        self.decoder: Callable[[torch.Tensor], torch.Tensor] = None

        self.T = 1.0

        if loss == "MSELoss":
            self.loss = MSELoss()
        elif loss == "L1Loss":
            self.loss = L1Loss()
        elif loss == "WeightedL1Loss":
            self.loss = WeightedL1Loss(self.args[0].loss_params)
        else:
            raise NotImplementedError(f"{loss} loss module is not supported")

    def G(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor, conditioning):
        ts = torch.stack((t, s), dim=1)

        return (s / t).unsqueeze(-1).unsqueeze(-1) * x_t + (1.0 - s / t).unsqueeze(-1).unsqueeze(-1) * self.model(x_t, ts, conditioning)

    def compute_loss(*args, **kwargs):
        """
        Method isn't used just provides loss titles
        """

        return {
            "loss": 0.0,
            "ctm": 0.0,
            "dsm": 0.0,
        }

    @torch.no_grad()
    def sample_gamma(
        self,
        batch_size,
        horizon,
        local_conditioning,
        global_conditioning,
        device: str,
        N=1,
        gamma: float = 0.1,
    ):
        x_t = torch.randn((batch_size, horizon, self.unet_args.traj_dim), device=device)

        def get_t_tensor(t): return torch.full((batch_size,), float(t), device=device)

        if N == 1:
            return self.model(x_t, get_t_tensor(1.0).unsqueeze(1).repeat(1, 2), (local_conditioning, global_conditioning))

        T = 1.
        timesteps = (T * (torch.arange(N+1) / N) + 1e-9).flip(0)

        for n in range(N):
            t_n = get_t_tensor(timesteps[n])
            t_next = get_t_tensor(timesteps[n + 1])
            t_tilde = np.sqrt(1.0 - gamma**2) * t_next

            # Denoise jump  t_n → t̃_{n+1}
            x_tilde = self.G(x_t, t_n, t_tilde, (local_conditioning, global_conditioning))

            if gamma == 0.0:
                x_t = x_tilde
            else:
                eps = torch.randn_like(x_t)
                x_t = x_tilde + gamma * torch.sqrt(t_next).view(-1, 1, 1) * eps

        return x_t

    @torch.no_grad()
    def sample(self,
               batch_size: int,
               horizon: int,
               device: str,
               local_conditioning: torch.Tensor,
               global_conditioning: torch.Tensor,
               sampler: tuple[SamplerType, int],
               decoder_downsample: int = 1,
               w: float = 0.0
               ) -> torch.Tensor:
        """
        1. Samples latents from pure noise using the reverse diffusion process.
        2. Decodes latents using the VAE decoder.

        Parameters:
            batch_size: number of trajectories to generate
            horizon: length of each trajectory. must be divisible by 4 (VAE) * 8 (diffusion) = 32
            device: device for pytorch tensors
            local_conditioning: Set of waypoints given as [batch_size x 6  x 3] where the center is the initial state
            global_conditioning: Set of waypoints given as [batch_size x 4  x 4]
            sampler: SamplerType and number of steps
            decoder_downsample: What factor VAE downsample the trajectory (1 == no decoder)
            w: Classifier free guidance weight
        Returns:
            x: Generated trajectories of shape [batch_size x horizon x traj_dim]
        """

        diffusion_downsample = 2 ** (len(self.args[1].channel_mults) - 1)
        assert horizon % (decoder_downsample * diffusion_downsample) == 0

        sample_horizon = horizon // decoder_downsample

        trajectories = self.sample_gamma(
            batch_size, sample_horizon, local_conditioning, global_conditioning, device, sampler[1]
        )
        # if sampler[0] == SamplerType.GAMMA_CTM:
        #     pass
        # else:
        #     raise ValueError(f"Sampler {sampler} not implemented.")

        if self.decoder:
            trajectories = self.decoder(trajectories)

        return trajectories
