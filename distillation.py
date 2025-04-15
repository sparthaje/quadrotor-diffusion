import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper
from quadrotor_diffusion.models.losses import MSELoss
from quadrotor_diffusion.utils.nn.args import TrainerArgs


class LcmDistillationTrainer(Trainer):
    def __init__(
        self,
        args: TrainerArgs,
        model: nn.Module,
        dataset: Dataset,
    ):
        super().__init__(args, model, dataset)
        self.distillation_loss = MSELoss().to(self.args.device)

        self.ema_model: LatentDiffusionWrapper = copy.deepcopy(self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model)
        self.ema_model.to(args.device)

    def f(
        z: torch.Tensor,
        w: float,
        c_local: torch.Tensor,
        c_global: torch.Tensor,
        t: torch.Tensor,
        model: LatentDiffusionWrapper,
        device: str
    ) -> torch.Tensor:
        alpha_t = model.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
        epsilon = model(z, t, (c_local, c_global))

        # This isn't really a real value for my data but this function is just here to serve as a boundary condition
        sigma_data = 0.5
        # Scaling that affects smoothness of boundary condition larger = sharper
        s = 10.0
        c_skip = (sigma_data ** 2) / (torch.pow(s * t, 2) + sigma_data ** 2)
        c_out = (s * t) / torch.sqrt(torch.pow(s * t, 2) + sigma_data ** 2)

        z_0_hat = (z - torch.sqrt(1 - alpha_t) * epsilon) / torch.sqrt(alpha_t)

        # This boundary condition enforces that at t=0 we get the initial latent back in a smooth differential way
        return c_skip * z + c_out * z_0_hat

    def ddim_solver(
        self,
        z_nk: torch.Tensor,
        t_nk: torch.Tensor,
        t_n: torch.Tensor,
        c_local: torch.Tensor,
        c_global: torch.Tensor,
        w: float,
    ) -> torch.Tensor:
        alpha_n = self.model.alpha_bar[t_n].unsqueeze(-1).unsqueeze(-1)
        sigma_n = torch.sqrt(1 - alpha_n)
        alpha_n = torch.sqrt(alpha_n)

        alpha_nk = self.model.alpha_bar[t_nk].unsqueeze(-1).unsqueeze(-1)
        sigma_nk = torch.sqrt(1 - alpha_nk)
        alpha_nk = torch.sqrt(alpha_nk)

        c1 = alpha_n / alpha_nk - 1
        c2 = (sigma_nk * alpha_n) / (alpha_nk * sigma_n) - 1
        c2 = sigma_n * c2

        z_tiled = z_nk.repeat(2, 1, 1)
        time_t = torch.ones(z.shape[0], device=device).long() * t
        time_t_tiled = time_t.repeat(2)

        conditioning_0 = c_local.repeat(2, 1, 1)
        conditioning_1 = torch.cat(
            c_global,
            model.null_token_global.expand((
                c_global.shape[0], c_global.shape[1], -1
            ))
        )
        model_output = model(z_tiled, time_t_tiled, (conditioning_0, conditioning_1))
        eps_c = model_output[:model_output.shape[0]//2, :, :]
        eps_null = model_output[model_output.shape[0]//2:, :, :]

        psi_c = c1 * z_nk - c2 * eps_c
        psi_null = c1 * z_nk - c2 * eps_null

        z_n_psi = z_nk + (1 + w) * psi_c - w * psi_null
        return z_n_psi

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

        z_n_psi = z_nk + (1 + w) * psi(z_nk, n+K, n, c) - w * psi(z_nk, n+K, n, c_null)

        out_one = LcmDistillationTrainer.f(z_nk, w, c, n+K, self.model, self.args.device)
        out_two = LcmDistillationTrainer.f(z_n_psi, w, c, n, self.ema_model, self.args.device)

        return self.distillation_loss(
            out_one, out_two
        )
