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
        return F.mse_loss(pred, targ, reduction='mean')


class TrackabilityLoss(nn.Module):
    def __init__(self, shape, intercept, exp):
        """
        Compute acceleration and velocity profile, such that:

        a(t) < shape * |v(t)|^exp + intercept

        """
        super().__init__()

        self.shape = shape
        self.intercept = intercept
        self.exp = exp

    def forward(self, trajectory):
        raise NotImplementedError
        velcoity = diffrentiate(trajectory)
        accel = diffrentiate(trajectory)
