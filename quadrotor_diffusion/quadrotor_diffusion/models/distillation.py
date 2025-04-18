import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import ConsistencyTrajectoryWrapper
from quadrotor_diffusion.models.losses import L1Loss
from quadrotor_diffusion.utils.nn.args import TrainerArgs, LatentConsistencyArgs
from quadrotor_diffusion.utils.nn.schedulers import alpha_bar_cont, sample_times


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


class ConsistencyTrajectoryTrainer(Trainer):
    def __init__(
        self,
        args: TrainerArgs,
        model: nn.Module,
        dataset: Dataset,
    ):
        # Add it to model args for logging purposes
        super().__init__(args, model, dataset)

        self.ctm_loss = L1Loss().to(self.args.device)
        self.dsm_loss = L1Loss().to(self.args.device)

        self.model: ConsistencyTrajectoryWrapper
        self.ema_model: ConsistencyTrajectoryWrapper = copy.deepcopy(self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model)
        self.ema_model.to(args.device)

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

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

        batch_size = len(batch["x_0"])
        mask = torch.rand(batch_size, device=self.args.device) < self.model.args[0].dropout
        batch["global_conditioning"][mask] = self.model.null_token_global.expand((
            -1, batch["global_conditioning"].shape[1], -1
        ))
        conditioning = (
            batch["local_conditioning"], batch["global_conditioning"]
        )

        t_n = sample_times(batch_size, device=self.args.device)  # top of jump
        t_s = sample_times(batch_size, T=t_n, device=self.args.device)  # start of student jump
        t_u = torch.empty_like(t_n)
        u_mask = torch.rand_like(t_n) < 0.5  # 50‑50 pick exact s or random in [s,t)
        t_u[u_mask] = t_s[u_mask]
        t_u[~u_mask] = sample_times((~u_mask).sum().item(), T=t_n[~u_mask],
                                    eps=t_s[~u_mask].min().item(), device=self.args.device)

        x_0 = batch["x_0"]

        with torch.no_grad():
            if self.model.encoder:
                # Discard logvar component
                x_0, _ = self.model.encoder(x_0)

            epsilon = torch.randn_like(x_0)
            alpha_t = alpha_bar_cont(t_n.view(-1, 1, 1)).to(x_0.dtype)
            x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1.0 - alpha_t) * epsilon

        with torch.no_grad():
            x_u = self.ema_model.model(x_t, t_n.unsqueeze(1).repeat(1, 2), conditioning)
        x_u_euler = (t_u / t_n).view(-1, 1, 1) * x_t + (1.0 - (t_u / t_n)).view(-1, 1, 1) * x_u

        x_target = self.ema_model.G(
            x_t=self.ema_model.G(
                x_t=x_u_euler,
                t=t_u,
                s=t_s,
                conditioning=conditioning,
            ),
            t=t_s,
            s=torch.zeros_like(t_s),
            conditioning=conditioning
        )

        x_estimate = self.ema_model.G(
            self.model.G(
                x_t=x_t,
                t=t_n,
                s=t_s,
                conditioning=conditioning,
            ),
            t=t_s,
            s=torch.zeros_like(t_s),
            conditioning=conditioning
        )

        x_0_hat = self.model.model(x_t, t_n.unsqueeze(1).repeat(1, 2), conditioning)

        ctm_losses = self.ctm_loss(x_estimate, x_target)
        dsm_losses = self.dsm_loss(x_0, x_0_hat)

        g_ctm = torch.autograd.grad(ctm_losses["loss"], self.model.parameters(), retain_graph=True, allow_unused=True)
        g_dsm = torch.autograd.grad(dsm_losses["loss"], self.model.parameters(), retain_graph=True, allow_unused=True)
        def norm(gs): return torch.norm(torch.stack([p.norm() for p in gs if p is not None]))

        scale_dsm = (norm(g_ctm) / (norm(g_dsm) + 1e-6)).detach()
        scale_ctm = (norm(g_dsm) / (norm(g_ctm) + 1e-6)).detach()
        if self.epoch < 50:
            scale_ctm = 0.0
            scale_dsm = 1.0

        loss = {
            "loss": scale_ctm * ctm_losses["loss"] + scale_dsm * dsm_losses["loss"],
            "ctm": ctm_losses["loss"].detach(),
            "dsm": dsm_losses["loss"].detach(),
        }

        return loss
