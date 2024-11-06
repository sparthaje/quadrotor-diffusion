import torch
import torch.nn as nn

from quadrotor_diffusion.utils.schedulers import cosine_beta_schedule
from quadrotor_diffusion.models.losses import MSELoss
from quadrotor_diffusion.models.temporal import Unet1D
from quadrotor_diffusion.utils.args import Unet1DArgs, DiffusionWrapperArgs


class DiffusionWrapper(nn.Module):
    def __init__(
        self,
        diffusion_args: DiffusionWrapperArgs,
        unet_args: Unet1DArgs
    ):
        """
        Wrapper that noises sample and computes the loss for a denoising diffusion model
        """
        super().__init__()
        self.diffusion_args = diffusion_args
        self.unet_args = unet_args

        predict_epsilon = diffusion_args.predict_epsilon
        n_timesteps = diffusion_args.n_timesteps
        loss = diffusion_args.loss

        # Model to either predict epsilon or x_t directly
        self.model = Unet1D(unet_args)

        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon

        # Compute Scheduler Conflicts, betas is an array of dimension 1
        betas = cosine_beta_schedule(n_timesteps)
        alpha = 1. - betas
        alpha_bar = torch.cumprod(alpha, axis=0)
        self.register_buffer('alpha_bar', alpha_bar)

        if loss == "MSELoss":
            self.loss = MSELoss()
        else:
            raise NotImplementedError(f"{loss} loss module is not supported")

    def compute_loss(self, x_0):
        """
        Does a forward pass and computes the loss

        Parameters:
        - x_0: clean trajectories [batch_size x horizon x states]

        Return:
        - loss: Result from loss function
        """

        batch_size = len(x_0)

        # Sample noisy trajectories
        with torch.no_grad():
            # Select random timesteps for each sample in batch
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0.device).long()
            epsilon = torch.randn_like(x_0)

            # Unsqueeze alpha_bar so it goes from [1] to [1 x 1 x 1] allowing for broadcasting
            alpha_t = self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1)
            x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * epsilon

        model_output = self.model(x_t, t)
        # TODO(shreepa): add ability to train inpainting by fixing specific points in traj to inpaitned vals

        target = epsilon if self.predict_epsilon else x_0
        loss = self.loss(model_output, target)

        return loss

    def save(self, filepath):
        torch.save(
            {
                'torch_params': self.state_dict(),
                'diffusion_args': self.diffusion_args,
                'unet_args': self.unet_args,
            }, filepath
        )

    @staticmethod
    def load(filepath):
        state_dict = torch.load(filepath)
        diffusion_args = state_dict['diffusion_args']
        unet_args = state_dict['unet_args']

        model = DiffusionWrapper(diffusion_args, unet_args)
        model.load_state_dict(state_dict['model_state_dict'])

        return model
