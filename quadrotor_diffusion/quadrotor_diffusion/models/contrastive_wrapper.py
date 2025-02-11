import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from quadrotor_diffusion.models.vae import VAE_Wrapper
from quadrotor_diffusion.utils.nn.args import CourseEmbeddingArgs, VAE_WrapperArgs, VAE_EncoderArgs, VAE_DecoderArgs
from quadrotor_diffusion.models.nn_blocks import soft_argmax, soft_argmin
from quadrotor_diffusion.utils.logging import iprint as print


class ContrastiveWrapper(nn.Module):
    def __init__(self,
                 args: tuple[CourseEmbeddingArgs, VAE_WrapperArgs, VAE_EncoderArgs, VAE_DecoderArgs],
                 ):
        super().__init__()
        self.args = args

        # Non-linear projection of gates from R^4 to R^{embed_dim}
        course_encoder = [nn.Linear(args[0].gate_input_dim, args[0].hidden_dim), nn.Tanh()]
        for _ in range(args[0].n_layers):
            course_encoder.append(nn.Linear(args[0].hidden_dim, args[0].hidden_dim))
            course_encoder.append(nn.Tanh())
        course_encoder.append(nn.Linear(args[0].hidden_dim, args[0].embed_dim))
        self.course_encoder = nn.Sequential(*course_encoder)

        # Convert nx3 trajectory into (n/4) x 6 latent
        self.trajectory_encoder = VAE_Wrapper((args[1], args[2], args[3]))

        # Non-linear projection of trajectory latents from R^{latent_dim} into R^{embed_dim}
        trajectory_projection = [nn.Linear(args[2].latent_dim, args[0].hidden_dim), nn.Tanh()]
        for _ in range(args[0].n_layers):
            trajectory_projection.append(nn.Linear(args[0].hidden_dim, args[0].hidden_dim))
            trajectory_projection.append(nn.Tanh())
        trajectory_projection.append(nn.Linear(args[0].hidden_dim, args[0].embed_dim))
        self.trajectory_projection = nn.Sequential(*trajectory_projection)

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.traj_latent_downsample = torch.tensor(2**(len(args[2].channel_mults) - 1), dtype=torch.int16)

    @torch.no_grad()
    def embed_course(self, courses: torch.Tensor) -> torch.Tensor:
        """
        Computes embeddings for courses without gradient.
        Args:
            courses (torch.Tensor): Course embedding. Shape [batch_size, N_gates (6), N_features (4)]
        Returns:
           torch.Tensor: Course embeddings. Shape [batch_size, N_gates, embed_dim]
        """
        return self._embed_course(courses)

    def _embed_course(self, courses: torch.Tensor) -> torch.Tensor:
        """
        Computes embedding for courses.
        Args:
            courses (torch.Tensor): Course embedding. Shape [batch_size, N_gates, N_features]
        Returns:
           torch.Tensor: Course embeddings. Shape [batch_size, N_gates, embed_dim]
        """

        # [batch_size, N_gates, N_features] -> [batch_size, N_gates x embed_dim]
        course_embedding = self.course_encoder(courses)
        course_embedding = F.normalize(course_embedding, p=2, dim=-1)

        return course_embedding

    def get_cosine_similarity(self, course_embedding: torch.Tensor, traj_latent: torch.Tensor) -> torch.Tensor:
        """
        Computes similarity between each gate and every point on the trajectory.

        Args:
            courses (torch.Tensor): Course embedding. Shape [B, N_gates, embed_dim]
            traj_latent (torch.Tensor): Trajectory latent vectors. Shape [B, vae_horizon, vae_latent_dim]

        Returns:
            [B, N_gates, vae_horizon]: Alignment of each gate with every point on vae horizon
        """

        # 1. Compute trajectory embeddings

        # [B, vae_horizon, embed_dim]
        trajectory_embedding = self.trajectory_projection(traj_latent)
        trajectory_embedding = F.normalize(trajectory_embedding, p=2, dim=-1)

        # 2. Compute cosine similarities for each gate in the trajectory and each individual feature vector in the trajectory
        #    -> Each row corresponds to how gate_i aligns with each traj_feature_j
        #    -> Each column corresponds to one traj_feature_j aligns with each gate_i

        # [B, N_gates, embed_dim] @ [B, vae_horizon, embed_dim] -> [B, N_gates, vae_horizon]
        cosine_similarity = torch.matmul(course_embedding, trajectory_embedding.transpose(-2, -1))
        return cosine_similarity

    def get_alignment(self, course_embedding: torch.Tensor, traj_latent: torch.Tensor) -> torch.Tensor:
        """
        Computes one score: [-1, 1] that indicates how well a course aligns to a given trajectory.
        Args:
            courses (torch.Tensor): Course embedding. Shape [B, N_gates, embed_dim]
            traj_latent (torch.Tensor): Trajectory latent vectors. Shape [B, vae_horizon, vae_latent_dim]

        Returns:
            torch.Tensor: Alignment score
        """

        # 1: Compute cosine similarity
        cosine_similarity = self.get_cosine_similarity(course_embedding, traj_latent)

        # 2: Compute where in trajectory each gate aligns most with (soft_indices) and what this alignment is (max_gate_alignments)
        #      -> We want the soft_indices to be strictly monotonic (i.e. pass gates in order)
        #      -> We want each gate to have one point in the trajectory with high alignment

        # Both will be [B, N_gates]
        soft_indices, max_gate_alignments = soft_argmax(cosine_similarity)

        # 3: For every course, create pairs of gates and give score between -1 and +1 based on if the indices
        #      -> This shouldn't encourage slower trajectories (i.e. ones where the difference in soft indices is larger)
        #         because once the difference is 2 latent points (= 2*4/30 = 0.27 seconds) the score will be 0.96

        # [B, N_gates - 1]
        pairwise_index_differences = soft_indices[..., 1:] - soft_indices[..., :-1]
        # [B, N_gates - 1]
        pairwise_ordered_score = 2 * torch.sigmoid(2 * pairwise_index_differences) - 1
        # [B]
        _, min_order_scores = soft_argmin(pairwise_ordered_score)

        # 4: Identify the score for the gate farthest off trajectory
        #     -> This should ideally be a very positive number which means all gates are on the trajectory

        # [B]
        _, min_alignments = soft_argmin(max_gate_alignments)

        k = 5
        # Smooth-logic gate that checks for A>0 AND B>0
        alignment_scores = 2 * F.sigmoid(k*min_order_scores) * F.sigmoid(k*min_alignments) - 1

        return alignment_scores

    def compute_loss(self, batch: dict[str, torch.Tensor], epoch=0, **kwargs) -> torch.Tensor:
        """
        Computes contrastive loss between trajectory and course

        Args:
            batch (dict[str, torch.Tensor]): Has to have keys "trajectory" and "course"
                -> batch["trajectory"]: [B, Horizon, XYZ (3)]
                -> batch["course"]: [B, N_gates x N_features]
                -> batch["gate_positions"]: [B, N_gates] What idx each gate gets passed through in original trajectory

            epoch (int): Current epoch of training
            **kwargs: Captures other arguments which other compute_loss methods might use

        Returns:
            torch.Tensor: Loss dict from one of the `losses.py` losses
        """

        course_embedding = self._embed_course(batch["course"])

        # ======== Loss along horizon (row wise loss) ========
        # Teach the embeddings how to differentiate matches from neighboring embeddings

        # [B, VAE_horizon, Latent features]
        trajectory_mu, _ = self.trajectory_encoder.encode(batch["trajectory"], padding=self.args[0].vae_padding)

        # [B, N_gates, VAE_horizon]
        cosine_similarity = self.get_cosine_similarity(course_embedding, trajectory_mu) * torch.exp(self.temperature)

        # [B, N_gates]
        gate_positions = (batch["gate_positions"] + self.args[0].vae_padding) // self.traj_latent_downsample

        # Flatten target to [B * N_gates] and cosine_similarity to [B * N_gates, VAE horizon]
        target = gate_positions.reshape(-1)
        pred = cosine_similarity.reshape(-1, cosine_similarity.size(-1))

        # Compute the row-wise loss where there's a lot of negative samples
        row_loss = F.cross_entropy(pred, target, reduction='mean')

        # ======== Loss along column (col wise loss) ========
        # Teach the embeddings how to differentiate between different gates, but also similar gates

        B, num_gates = gate_positions.shape

        # Number of epochs to train without these hard examples
        vae_horizon = cosine_similarity.shape[-1]

        # Number of negative samples to train with + the real sample
        N_INVALID = 2

        # [B, N_INVALID, Horizon, 3]
        trajectories = batch["trajectory"].unsqueeze(1).repeat(1, N_INVALID, 1, 1)

        # Create negative samples where the x dimension is slightly shifted
        # trajectories[:, 1, :, 0] += torch.tensor([random.uniform(0.2, 0.5) for _ in range(B)],
        #                                          dtype=trajectories.dtype, device=trajectories.device).unsqueeze(1)

        # # Create negative samples where the y dimension is slightly shifted
        # trajectories[:, 2, :, 1] += torch.tensor([random.uniform(0.2, 0.5) for _ in range(B)],
        #                                          dtype=trajectories.dtype, device=trajectories.device).unsqueeze(1)

        # Create negative samples where the z dimension is flipped (i.e. gates at 0.3 are now at 0.525 and vice versa)
        trajectories[:, 1, :, 2] = (0.525 + 0.3) - trajectories[:, 1, :, 2]

        # [N_INVALID*B, Horizon, 3]
        trajectories = trajectories.view(N_INVALID*B, -1, 3)

        # [N_INVALID*B, VAE_Horizon, latent_dim]
        trajectory_mu, _ = self.trajectory_encoder.encode(trajectories, padding=self.args[0].vae_padding)

        # [B, N_INVALID, VAE_horizon, latent_dim]
        trajectory_mu = trajectory_mu.view(B, N_INVALID, -1, self.args[2].latent_dim)

        # [B, N_INVALID, N_gates, embed_dim]
        course_embedding_expanded = course_embedding.unsqueeze(1).expand(-1, N_INVALID, -1, -1)

        # [B, N_INVALID, N_gates, VAE_Horizon]
        cosine_similarity = self.get_cosine_similarity(course_embedding_expanded, trajectory_mu) * \
            torch.exp(-self.temperature)

        # [B, N_INVALID * N_gates, VAE_Horizon]
        cosine_similarity = cosine_similarity.reshape(B, -1, vae_horizon)

        # Fill target with no gate at every timestep
        target = torch.full((B, vae_horizon), -100, dtype=torch.long, device=cosine_similarity.device)
        for b in range(B):
            for g in range(num_gates):
                # Location along horizon where this gate is hit
                h = int(gate_positions[b, g].item())
                if h < vae_horizon and target[b, h] == -100:
                    target[b, h] = g

        # [B, VAE_horizon, N_gates]
        pred: torch.Tensor = cosine_similarity.transpose(1, 2)
        # Flatten target to [B * VAE horizon] and pred to [B * VAE horizon, N_gates]
        target = target.reshape(-1)
        pred = pred.reshape(-1, pred.size(-1))

        col_loss: torch.Tensor = F.cross_entropy(pred, target, ignore_index=-100, reduction='mean')

        # ======== Combine both forms of loss ========

        if epoch < 100:
            total_loss = 0.2 * col_loss + 0.8 * row_loss
        elif epoch < 25:
            total_loss = 0.9 * col_loss + 0.1 * row_loss
        else:
            total_loss = 0.5 * col_loss + 0.5 * row_loss

        return {
            "loss": total_loss,
            "col_ce_loss": col_loss,
            "row_ce_loss": row_loss,
            "col_ce_accuracy": torch.exp(-col_loss),
            "row_ce_accuracy": torch.exp(-row_loss),
            "total_accuracy": torch.exp(-total_loss),
        }
