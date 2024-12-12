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


class SmoothWeightedL1(nn.Module):
    def __init__(self, alpha, w_l, w_g, epsilon=1e-6):
        super().__init__()
        """
        Computes a smooth L1 loss between prediction and target
        Parameters:
        - alpha: Sharpness around error=0
        - w_l: Weighting for prediction magnitude less than target magnitude
        - w_g: Weighting for prediction magnitude greater than target magnitude
        - epsilon: Small constant for numerical stability near zero
        """
        self.alpha = alpha
        self.w_l = w_l
        self.w_g = w_g
        self.epsilon = epsilon

    def forward(self, pred, targ):
        """
        Computes smooth weighted L1 loss based on absolute magnitude difference
        Parameters:
        - pred: network output [batch_size x horizon x states]
        - targ: expected output [batch_size x horizon x states]
        Returns:
        - Loss dictionary containing total loss and component
        """
        pred_mag = torch.sqrt(pred * pred + self.epsilon)
        targ_mag = torch.sqrt(targ * targ + self.epsilon)
        mag_diff = pred_mag - targ_mag

        weights = torch.sigmoid(self.alpha * mag_diff)
        l1 = F.l1_loss(pred, targ, reduction='none')

        # Compute weighted loss
        loss = torch.mean(
            self.w_g * weights * l1 +  # Higher weight when |pred| > |targ|
            self.w_l * (1 - weights) * l1  # Lower weight when |pred| < |targ|
        )

        return {
            "loss": loss,
            "L1": torch.mean(l1),
            "SmoothWeightedL1": loss,
        }


class SmoothnessLoss(nn.Module):

    def __init__(self, reconstruction_loss: nn.Module, order: int):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.smoothness_weighting = nn.Parameter(torch.tensor(1.0))
        self.recon_weighting = nn.Parameter(torch.tensor(1.0))
        self.order = order

    def forward(self, pred, target):
        """
        Computes the smoothness loss for a curve.

        Parameters
        - pred: network output [batch_size x horizon x states]
        - targ: expected output [batch_size x horizon x states]

        Returns:
        - Loss
        """

        diff = pred
        for _ in range(self.order):
            diff = diff[:, 1:, :] - diff[:, :-1, :]

        smoothness_loss = torch.mean(diff ** 2)
        recon_loss: dict = self.reconstruction_loss(pred, target)

        total_loss = self.recon_weighting * recon_loss["loss"] + \
            0.01 * self.smoothness_weighting * smoothness_loss

        recon_loss.update({
            "loss": total_loss,
            "s_recon_loss": recon_loss["loss"],
            "smoothness_loss": smoothness_loss
        })

        return recon_loss


class VAE_Loss(nn.Module):
    def __init__(self, recon_loss: nn.Module, beta: float):
        super().__init__()

        self.recon_loss = recon_loss
        self.beta = beta

    def forward(self, pred: Tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
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
