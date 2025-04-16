import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper
from quadrotor_diffusion.models.losses import MSELoss
from quadrotor_diffusion.utils.nn.args import TrainerArgs, LatentConsistencyArgs


class LcmDistillationTrainer(Trainer):
    def __init__(
        self,
        args: TrainerArgs,
        consistency_args: LatentConsistencyArgs,
        model: nn.Module,
        dataset: Dataset,
    ):
        # Add it to model args for logging purposes
        model.args.append(consistency_args)
        super().__init__(args, model, dataset)

        self.distillation_loss = MSELoss().to(self.args.device)

        self.ema_model: LatentDiffusionWrapper = copy.deepcopy(self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model)
        self.ema_model.to(args.device)
        self.consistency_args = consistency_args

    def ddim_solver(
        self,
        z_nk: torch.Tensor,
        t_nk: torch.Tensor,
        t_n: torch.Tensor,
        c_local: torch.Tensor,
        c_global: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        alpha_n = self.model.diffusion.alpha_bar[t_n].unsqueeze(-1).unsqueeze(-1)
        sigma_n = torch.sqrt(1 - alpha_n)
        sqrt_alpha_n = torch.sqrt(alpha_n)

        alpha_nk = self.model.diffusion.alpha_bar[t_nk].unsqueeze(-1).unsqueeze(-1)
        sigma_nk = torch.sqrt(1 - alpha_nk)
        sqrt_alpha_nk = torch.sqrt(alpha_nk)

        c1 = sqrt_alpha_n / sqrt_alpha_nk - 1
        c2 = sigma_n * ((sigma_nk * sqrt_alpha_n) / (sqrt_alpha_nk * sigma_n) - 1)

        z_nk_tiled = z_nk.repeat(2, 1, 1)
        time_nk_tiled = t_nk.repeat(2)

        conditioning_0 = c_local.repeat(2, 1, 1)
        conditioning_1 = torch.cat((
            c_global,
            self.model.null_token_global.expand((
                c_global.shape[0], c_global.shape[1], -1
            ))
        ))
        model_output = self.model.diffusion.model(z_nk_tiled, time_nk_tiled, (conditioning_0, conditioning_1))
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

        W_MAX = self.consistency_args.w_max
        W_MIN = self.consistency_args.w_min
        K = self.consistency_args.k

        # Generate B random values between 0.2 and 1 on same device as batch["x_0"]
        B = batch["x_0"].shape[0]
        w = torch.rand(B).to(batch["x_0"].device) * (W_MAX - W_MIN) + W_MIN
        w = w.unsqueeze(-1).unsqueeze(-1)

        N = self.model.args[0].n_timesteps - 1
        # Generate B random integers between 1 and N-K on same device as batch["x_0"]
        t_n = torch.randint(1, N-K, size=(B,), device=batch["x_0"].device)

        # Encode latent trajectories
        z_0, _ = self.model.encoder(batch["x_0"])

        epsilon = torch.randn_like(z_0, device=batch["x_0"].device)

        # Unsqueeze alpha_bar so it goes from [1] to [1 x 1 x 1] allowing for broadcasting
        alpha_nk = self.model.diffusion.alpha_bar[t_n+K].unsqueeze(-1).unsqueeze(-1)
        z_nk = torch.sqrt(alpha_nk) * z_0 + torch.sqrt(1 - alpha_nk) * epsilon
        z_n_psi = self.ddim_solver(
            z_nk=z_nk,
            t_nk=t_n + K,
            t_n=t_n,
            c_local=batch["local_conditioning"],
            c_global=batch["global_conditioning"],
            w=w,
        )

        out_student = self.model.diffusion.consistency_step(
            z_n=z_nk,
            c_local=batch["local_conditioning"],
            c_global=batch["global_conditioning"],
            t_n=t_n + K,
            w=w,
        )

        out_teacher = self.ema_model.diffusion.consistency_step(
            z_n=z_n_psi,
            c_local=batch["local_conditioning"],
            c_global=batch["global_conditioning"],
            t_n=t_n,
            w=w,
        )

        return self.distillation_loss(
            out_student, out_teacher
        )
