from typing import Tuple

import torch
import torch.nn as nn
import einops

from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.models.losses import VAE_Loss, MSELoss, L1Loss, SmoothnessLoss
from quadrotor_diffusion.utils.nn.args import VAE_EncoderArgs, VAE_DecoderArgs, VAE_WrapperArgs
from quadrotor_diffusion.models.attention import LinearAttention, Residual, PreNorm
from quadrotor_diffusion.models.nn_blocks import (
    Conv1dNormalized,
    Downsample1d,
    Upsample1d
)


class ResNet1DBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=5):
        """
        Residual block for 1D data without time embeddings
        """
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dNormalized(c_in, c_out, kernel_size),
            Conv1dNormalized(c_out, c_out, kernel_size),
        ])

        # Transform input features if dimensions don't match
        self.residual_conv = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        """
        Parameters:
        x: [batch_size x c_in x horizon]
        """
        h = self.blocks[0](x)
        h = self.blocks[1](h)
        return h + self.residual_conv(x)


class Encoder1D(nn.Module):
    def __init__(self, args: VAE_EncoderArgs):
        """
        Encoder that compresses time series into a latent distribution
        """
        super().__init__()

        self.args = args
        assert args.channel_mults[0] == 1, "First channel multiplier must be 1"

        # Number of channels in each layer
        dims = [args.features * m for m in args.channel_mults]

        # Initial projection
        self.init_conv = Conv1dNormalized(args.traj_dim, args.features, kernel_size=5)

        # Down layers which reduce temporal dimension
        self.downs = nn.ModuleList([])
        for idx, c_in in enumerate(dims):
            if idx >= len(dims) - 1:
                break
            c_out = dims[idx + 1]

            self.downs.append(nn.ModuleList([
                ResNet1DBlock(c_in, c_out),
                Residual(PreNorm(c_out, LinearAttention(c_out, heads=4, dim_head=8))),
                ResNet1DBlock(c_out, c_out),
                Downsample1d(c_out)
            ]))

        # Final layers to produce mean and log variance
        final_features = dims[-1]
        self.final_block = ResNet1DBlock(final_features, final_features)
        self.to_mu = nn.Conv1d(final_features, args.latent_dim, 1)
        self.to_logvar = nn.Conv1d(final_features, args.latent_dim, 1)

    def forward(self, x):
        """
        x: [batch_size x horizon x traj_dim]
        returns: mu, logvar of shape [batch_size x latent_dim x compressed_horizon]
        """
        x = einops.rearrange(x, 'b h t -> b t h')
        x = self.init_conv(x)

        # Down sampling blocks
        for resnet1, attention, resnet2, downsample in self.downs:
            x = resnet1(x)
            x = attention(x)
            x = resnet2(x)
            x = downsample(x)

        # Final processing
        x = self.final_block(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)

        mu = einops.rearrange(mu, 'b l h -> b h l')
        logvar = einops.rearrange(logvar, 'b l h -> b h l')

        return mu, logvar


class Decoder1D(nn.Module):
    def __init__(self, args: VAE_DecoderArgs):
        """
        Decoder that reconstructs time series from latent representation
        """
        super().__init__()

        self.args = args

        # Number of channels in each layer
        dims = [args.latent_dim] + [args.features * m for m in args.channel_mults]

        # Initial projection
        self.init_conv = Conv1dNormalized(args.latent_dim, dims[1], kernel_size=5)

        # Up layers which increase temporal dimension
        self.ups = nn.ModuleList([])
        for idx, c_in in enumerate(dims[1:]):
            if idx >= len(dims) - 2:
                break
            c_out = dims[idx + 2]
            self.ups.append(nn.ModuleList([
                ResNet1DBlock(c_in, c_in),
                Residual(PreNorm(c_in, LinearAttention(c_in, heads=4, dim_head=8))),
                ResNet1DBlock(c_in, c_out),
                Upsample1d(c_out)
            ]))

        # Final layers to produce output
        self.final_block = ResNet1DBlock(dims[-1], dims[-1])
        self.to_out = nn.Conv1d(dims[-1], args.traj_dim, 1)

    def forward(self, z):
        """
        z: [batch_size x compressed_horizon x latent_dim]
        returns: [batch_size x horizon x traj_dim]
        """

        z = einops.rearrange(z, 'b h l -> b l h')
        x = self.init_conv(z)

        # Up sampling blocks
        for resnet1, attention, resnet2, upsample in self.ups:
            x = resnet1(x)
            x = attention(x)
            x = resnet2(x)
            x = upsample(x)

        # Final processing
        x = self.final_block(x)
        x = self.to_out(x)
        x = einops.rearrange(x, 'b t h -> b h t')

        return x


class VAE_Wrapper(nn.Module):
    def __init__(
        self,
        args: Tuple[VAE_WrapperArgs, VAE_EncoderArgs, VAE_DecoderArgs]
    ):
        """
        Variational Autoencoder for time series data
        """

        super().__init__()
        self.args = args

        assert isinstance(args[0], VAE_WrapperArgs)
        assert isinstance(args[1], VAE_EncoderArgs)
        assert isinstance(args[2], VAE_DecoderArgs)

        self.encoder = Encoder1D(args[1])
        self.decoder = Decoder1D(args[2])

        if args[0].loss == "MSELoss":
            self.loss = VAE_Loss(MSELoss(), args[0].beta)
        elif args[0].loss == "L1Loss":
            self.loss = VAE_Loss(L1Loss(), args[0].beta)
        elif args[0].loss == "L2Smooth":
            self.loss = VAE_Loss(SmoothnessLoss(MSELoss(), 1), args[0].beta)
        elif args[0].loss == "L1Smooth":
            self.loss = VAE_Loss(SmoothnessLoss(L1Loss(), 1), args[0].beta)
        else:
            raise NotImplementedError(f"{args[0].loss} loss module is not supported")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Perform reparameterization trick to sample from N(mu, var)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, x_0):
        """
        Does a forward pass and computes the loss

        Parameters:
        - x_0: clean trajectories [batch_size x horizon x states]

        Return:
        - loss: Result from loss function
        """

        mu, logvar = self.encoder(x_0)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        loss = self.loss((mu, logvar, reconstruction), x_0)
        return loss

    @torch.no_grad()
    def encode(self, x):
        """
        Encode input to latent distribution
        """
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z):
        """
        Decode latent vectors to time series
        """
        return self.decoder(z)

    @torch.no_grad()
    def sample(self, num_samples: int, horizon: int, device='cuda'):
        """
        Generate samples from random latent vectors
        """

        compressed_horizon = horizon // (2**len(self.args[1].channel_mults))
        z = torch.randn(num_samples, compressed_horizon, self.decoder.latent_dim, device=device)
        return self.decode(z)
