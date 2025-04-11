import copy

import tqdm
import torch
import torch.nn as nn

import quadrotor_diffusion.utils.nn.ema as ema
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper


class LcmDistillationTrainer(Trainer):

    def get_loss_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        LCM Distillation loss
        https://arxiv.org/pdf/2310.04378

        Args:
            batch (dict[str, torch.Tensor]):
                - x_0 (torch.Tensor): [B, Horizon, 3]
                - global_conditioning (torch.Tensor): [B, N_gates, 4]
                - local_conditioning (torch.Tensor): [B, N_states, 3]

        Returns:
            dict[str, torch.Tensor]: losses
        """
        self.model: LatentDiffusionWrapper = self.model
        assert self.model.encoder is not None

        W_MAX = 1
        W_MIN = 0.2
        K = 2

        # Generate B random values between 0.2 and 1 on same device as batch["x_0"]
        B = batch["x_0"].shape[0]
        w = torch.rand(B).to(batch["x_0"].device) * (W_MAX - W_MIN) + W_MIN

        N = self.model.args[0].n_timesteps - 1
        # Generate B random integers between 1 and N-K on same device as batch["x_0"]
        n = torch.randint(1, N-K, size=(B,), device=batch["x_0"].device)

        # Encode latent trajectories
        z_0 = self.model.encoder(batch["x_0"])

        epsilon = torch.randn_like(z_0)

        # Unsqueeze alpha_bar so it goes from [1] to [1 x 1 x 1] allowing for broadcasting
        alpha_nk = self.model.alpha_bar[n+K].unsqueeze(-1).unsqueeze(-1)
        z_nk = torch.sqrt(alpha_nk) * z_0 + torch.sqrt(1 - alpha_nk) * epsilon

        c = None
        c_null = None

        def psi(z, nk, n, c) -> torch.Tensor:
            """DDIM ODE Solver"""
            pass

        z_n_psi = z_nk + (1 + w) * psi(z_nk, n+K, n, c) - w * psi(z_nk, n+K, n, c_null)

        def f(z, w, c_local, c_global, t, model) -> torch.Tensor:
            alpha_t = self.model.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
            epsilon = model_output
            return c_skip * z + \
                c_out * (z - torch.sqrt(1 - alpha_t) * epsilon) / torch.sqrt(alpha_t)

        out_one = f(z_nk, w, c, n+K, self.model)
        out_two = f(z_n_psi, w, c, n, self.ema_model)

        return loss
