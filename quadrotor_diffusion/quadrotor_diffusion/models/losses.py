from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ):
        """
        Computes L2 Loss between prediction and target

        Parameters
        - pred: network output [batch_size x horizon x states]
        - targ: expected output [batch_size x horizon x states]

        Returns:
        - Loss
        """

        loss = F.mse_loss(pred, targ, reduction='mean')
        return {
            "loss": loss,
            "L2": loss
        }


class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ):
        """
        Computes L1 Loss between prediction and target

        Parameters
        - pred: network output [batch_size x horizon x states]
        - targ: expected output [batch_size x horizon x states]

        Returns:
        - Loss
        """
        loss = F.l1_loss(pred, targ, reduction='mean')
        return {
            "loss": loss,
            "L1": loss
        }


class WeightedL1Loss(nn.Module):

    def __init__(self, params: tuple[float, float]):
        """

        Args:
            params (tuple[int, int]): (Percent horizon, weight (>1))
        """
        super().__init__()
        self.params = params

    def forward(self, pred, targ):
        """
        Computes weighted L1 Loss between prediction and target

        Parameters
        - pred: network output [batch_size x horizon x states]
        - targ: expected output [batch_size x horizon x states]

        Returns:
        - Loss
        """
        loss = F.l1_loss(pred, targ, reduction='none')

        _, H, _ = pred.shape
        weights = torch.ones_like(pred)
        weights[:, :int(H*self.params[0]), :] = self.params[1]

        loss = (weights * loss).mean()

        return {
            "loss": loss,
            "L1": loss
        }


class SmoothReconstructionLoss(nn.Module):

    def __init__(self, reconstruction_loss: nn.Module, betas: list[float]):
        """
        Computes reconstruction loss for a curve and its derivatives.

        Args:
            reconstruction_loss (nn.Module): Reconstruction loss
            betas (list[float]): For every beta i, the (i+1)th order derivative's loss is weighted by beta_i
        """
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.betas = betas

    def forward(self, pred, target) -> dict[str, torch.Tensor]:
        """
        Computes the smoothness loss for a curve.

        Parameters
        - pred: network output [batch_size x horizon x states]
        - target: expected output [batch_size x horizon x states]

        Returns:
        - Loss
        """
        total_loss = self.reconstruction_loss(pred, target)["loss"]

        recon_loss = dict()
        recon_loss["recon"] = total_loss

        diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
        diff_target = target[:, 1:, :] - target[:, :-1, :]
        for i, beta in enumerate(self.betas):
            nth_order_loss = self.reconstruction_loss(diff_pred, diff_target)["loss"]
            recon_loss[f"recon_{i+2}"] = nth_order_loss
            total_loss += beta * nth_order_loss

            # Compute next derivative (uniformly spaced points)
            diff_pred = diff_pred[:, 1:, :] - diff_pred[:, :-1, :]
            diff_target = diff_target[:, 1:, :] - diff_target[:, :-1, :]

        recon_loss["loss"] = total_loss
        return recon_loss


class VAE_Loss(nn.Module):
    def __init__(self, recon_loss: nn.Module, beta: float):
        super().__init__()

        self.recon_loss = recon_loss
        self.beta = beta

    def forward(self, pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args
        - pred: network output 
                pred[0]: [batch_size x compressed_horizon x states] (mu)
                pred[1]: [batch_size x compressed_horizon x states] (logvar)
                pred[2]: [batch_size x horizon x states] (reconstructed)
        - targ: expected output [batch_size x horizon x states]

        Returns:
        - Loss
        """

        kl_loss = -0.5 * torch.sum(1 + pred[1] - pred[0].pow(2) - pred[1].exp()) / torch.numel(pred[0])
        recon_loss_dict: dict = self.recon_loss(pred[2], target)

        recon_loss = recon_loss_dict["loss"]
        total_loss = recon_loss + self.beta * kl_loss

        loss_dict = dict()
        for key in recon_loss_dict:
            loss_dict[key] = recon_loss_dict[key]

        loss_dict["loss"] = total_loss
        loss_dict["VAE_Reconstruction"] = recon_loss
        loss_dict["KL_Divergence"] = kl_loss

        return loss_dict


class SymmetricLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            - pred: [M x M]

        Returns:
            - Loss
        """
        M, _ = pred.shape

        targets = torch.arange(M, device=pred.device)
        loss_row = F.cross_entropy(pred, targets)
        loss_col = F.cross_entropy(pred.t(), targets)
        loss = 0.5 * loss_row + 0.5 * loss_col

        return {
            "loss": loss,
            "row_loss": loss_row,
            "col_loss": loss_col,
            "symmetric": loss,
        }
