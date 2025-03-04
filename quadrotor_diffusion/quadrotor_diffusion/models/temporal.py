import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange, Reduce

from quadrotor_diffusion.models.nn_blocks import (
    SinusoidalPosEmb,
    Conv1dNormalized,
    Downsample1d,
    Upsample1d
)
from quadrotor_diffusion.models.attention import (
    LinearAttention,
    LinearCrossAttention,
    CrossAttention,
    PreNorm,
    Residual
)
from quadrotor_diffusion.utils.nn.args import Unet1DArgs
from quadrotor_diffusion.utils.logging import iprint as print


class ResNet1DBlock(nn.Module):
    def __init__(self, c_in, c_out, t_embed_dim, kernel_size=5):
        """
        Residual block for 1D data inspired by Stable Diffusion
        """
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dNormalized(c_in, c_out, kernel_size),
            Conv1dNormalized(c_out, c_out, kernel_size),
        ])

        # Transform input sinusoidal embeddings tino output dimensions
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(t_embed_dim, c_out),
            Rearrange("batch t -> batch t 1"),
        )

        # Transform input embeddings into output embeddings feature before adding residual connection
        self.residual_conv = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_embed):
        """
        Parameters:
        x: [batch_size x c_in x horizon]
        t_embed: [batch_size x embeed_dim]
        """
        out = self.blocks[0](x) + self.time_mlp(t_embed)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class Unet1D(nn.Module):
    def __init__(
        self,
        args: Unet1DArgs
    ):
        """
        1 dimensional Unet

        Projects the horizon x traj_dim data into horizon x features
        Down samples (factor of two) the horizon while scaling the feature dimension
        Up samples (factor of two) the horizon while dividing the feature dimension
        """
        super().__init__()

        traj_dim = args.traj_dim
        features = args.features
        channel_mults = args.channel_mults
        attentions = args.attentions

        assert channel_mults[0] == 1, "First scalar for channels in Unet must be 1"
        assert all(x < y for x, y in zip(channel_mults, channel_mults[1:])), "Channel scales must be increasing"

        assert len(attentions) == 3 and len(attentions[0]) == len(channel_mults) and len(attentions[1]) == 1 and \
            len(attentions[2]) == len(channel_mults) - 1, \
            "Attentions must have 3 subarrays for down scale, mid, upscale with appropriate number of bools"

        # Projects sinusoidal embeddings into number of features
        if args.context_mlp == "time":
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(features),
                nn.Linear(features, 4 * features),
                nn.Mish(),
                nn.Linear(4 * features, features)
            )

        # Network tries to learn how to embed waypoints for this UNET
        elif args.context_mlp == "waypoints":
            self.time_mlp = nn.Sequential(
                nn.Linear(4, 4*features),
                nn.Conv1d(4*features, 8*features, kernel_size=5, stride=1, padding=2),
                nn.MaxPool1d(kernel_size=2),
                nn.Mish(),
                nn.Conv1d(8*features, 4*features, kernel_size=5, stride=1, padding=2),
                nn.Linear(4*features, features),
                nn.Mish(),
                nn.Linear(features, features),
                Reduce("b n c -> b c", reduction="mean")
            )
        else:
            raise ValueError("Unknown context")

        # Number of channels in each layer from top down in the Unet
        dims = [traj_dim] + [features * m for m in channel_mults]

        # Down layers which cut horizon down by factors of 2^(len(channels_mult) - 1)
        self.downs = nn.ModuleList([])
        for idx, (c_in, attention) in enumerate(zip(dims, attentions[0])):
            if idx >= len(dims) - 1:
                break
            c_out = dims[idx + 1]
            self.downs.append(nn.ModuleList([
                ResNet1DBlock(c_in, c_out, t_embed_dim=features),
                ResNet1DBlock(c_out, c_out, t_embed_dim=features),
                Residual(PreNorm(c_out, LinearAttention(c_out))) if attention else nn.Identity(),
                Downsample1d(c_out) if c_out != dims[-1] else nn.Identity()
            ]))

        # Middle layers to transform mid features
        c_mid = dims[-1]
        self.middle = nn.ModuleList([
            ResNet1DBlock(c_mid, c_mid, t_embed_dim=features),
            Residual(PreNorm(c_mid, LinearAttention(c_mid))) if attentions[1][0] else nn.Identity(),
            ResNet1DBlock(c_mid, c_mid, t_embed_dim=features)
        ])

        self.conditioning = None
        if args.conditioning:
            self.conditioning_embedding = nn.Sequential(
                nn.Linear(4, c_mid // 2),
                nn.Mish(),
                nn.Linear(c_mid // 2, c_mid),
            )
            self.conditioning = Residual(PreNorm(c_mid, CrossAttention(c_mid, c_mid)))

        # Up layers which scales horizon up by factors of 2^(len(channels_mult))
        self.ups = nn.ModuleList([])
        dims_up = list(reversed(dims[1:]))
        for idx, (c_out, attention) in enumerate(zip(dims_up, attentions[2])):
            if idx >= len(dims_up) - 1:
                break
            c_in = dims_up[idx + 1]
            self.ups.append(nn.ModuleList([
                # Multiply c_out by two because we add the features of the same dimension from the left of UNET
                ResNet1DBlock(2 * c_out, c_in, t_embed_dim=features),
                ResNet1DBlock(c_in, c_in, t_embed_dim=features),
                Residual(PreNorm(c_in, LinearAttention(c_in))) if attention else nn.Identity(),
                Upsample1d(c_in)
            ]))

        self.final_conv = nn.Sequential(
            Conv1dNormalized(features, features, kernel_size=5),
            nn.Conv1d(features, traj_dim, kernel_size=1)
        )

    def forward(self, x, t, c=None):
        """
        x: [batch_size x horizon x traj_dim]
        t: if context_mlp == "time" [batch size]
           if context_mlp == "waypoints" [batch size x N_gates x 3]
        c: conditioning [batch_size, N_gates, 4]
        """

        x = einops.rearrange(x, "b h t -> b t h")
        t = self.time_mlp(t)

        skip_connections = []
        for resnet0, resnet1, attention, downsample in self.downs:
            x = resnet0(x, t)
            x = resnet1(x, t)
            x = attention(x)
            skip_connections.append(x)
            x = downsample(x)

        resnet0, attention, resnet1 = self.middle
        x = resnet0(x, t)
        x = attention(x)

        if c is not None and self.conditioning is not None:
            c = self.conditioning_embedding(c)
            c = einops.rearrange(c, 'b n f -> b f n')
            x = self.conditioning(x, c)
        elif c is not None and self.conditioning is None:
            raise ValueError("Need conditioning passed in")
        elif c is None and self.conditioning is not None:
            raise ValueError("Need conditioning in arguments")

        x = resnet1(x, t)

        for resnet0, resnet1, attention, upsample in self.ups:
            skip_connection = skip_connections.pop()
            x = torch.cat((x, skip_connection), dim=1)
            x = resnet0(x, t)
            x = resnet1(x, t)
            x = attention(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')

        return x
