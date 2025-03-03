from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from quadrotor_diffusion.utils.nn.args import Unet1DArgs
from quadrotor_diffusion.models.temporal import Unet1D
from quadrotor_diffusion.models.losses import SymmetricLoss
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.guide_functions.baseline import compute_trajectory_alignment


class LatentClassifierWrapper(nn.Module):
    def __init__(self, args: tuple[Unet1DArgs]):
        """
        Wrapper that uses a UNET to produce a classifier for p(c|z) where c is a course and z is a latent

        Args:
            args (tuple[Unet1DArgs]): UNET args
        """
        super().__init__()

        assert args[0].context_mlp == "waypoints"
        self.args = args

        self.model = Unet1D(args[0])
        self.encoder: Callable[[torch.Tensor], torch.Tensor] = None
        self.loss = SymmetricLoss()
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def compute_probability(self, z: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Computes probability that latent z aligns with waypoints

        Args:
            z (torch.Tensor): Latent trajectory [B, H_latent, C_latent]
            waypoints (torch.Tensor): Waypoints of course

        Returns:
            torch.Tensor: Scalar probability of alignment
        """
        # [B, H, C]
        out = F.sigmoid(self.model(z, waypoints))
        # [B]
        out = torch.mean(torch.mean(out, dim=-1), dim=-1)
        return out

    def compute_loss(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        Computes loss across each minibatch (where the scores should be identity matrix a la CLIP embeddings)

        Args:
            batch (dict[str, torch.Tensor]):
                - "trajectories" (torch.Tensor): Trajectory to embed in latent space [1, M, H, 3]
                - "courses" (torch.Tensor): Waypoints of course [1, M, N_gates, 4]

        Returns:
            dict[str, torch.Tensor]: Losses, "loss" key will always be there
        """
        assert self.encoder is not None, "Need to attach encoder for computing loss"

        B, M, H, C_traj = batch["trajectories"].shape
        _, _, N_gates, C_gate = batch["courses"].shape

        assert B == 1, "Batch size should be 1 because operating on one minibatch at a time"

        # [1, M, H, 3] -> [M, M, H, 3]
        trajectories = batch["trajectories"].expand(M, M, H, C_traj)

        # [M * M, H_latent, C_latent]
        # NOTE(shreepa): should learn with re parametrization trick here
        latents, _ = self.encoder(trajectories.reshape(M*M, H, C_traj))

        # [M * M, N_gates, C_gate]
        courses = batch["courses"].expand(M, M, N_gates, C_gate).reshape(M*M, N_gates, C_gate)

        # [M, M]
        p = self.compute_probability(latents, courses).reshape(M, M) * self.temperature

        return self.loss(p)
