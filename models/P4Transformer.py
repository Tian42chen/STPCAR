# %% Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .P4Convolution import P4DConv
from .Transformer import Transformer

# %% Define the model
class P4Transformer(nn.Module):
    def __init__(
        self,
        radius, nsamples, spatial_stride,      # P4DConv: spatial
        temporal_kernel_size, temporal_stride, # P4DConv: temporal
        emb_complex,                              # embedding: complex
        dim, depth, heads, dim_head,           # transformer
        mlp_dim, num_classes                   # output
    ):
        super().__init__()

        self.tube_embedding = P4DConv(
            mlp_plane=dim,
            spatial_kernel_size=[radius, nsamples],
            spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            temporal_padding=[1, 0],
        )

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.emb_complex = nn.Sequential(
            nn.GELU(),
            nn.Linear(2*dim, dim)
        ) if emb_complex else False

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )


    def forward(self, input):
        """
        Parameters
        ----------
        input : torch.Tensor
            (B, L, N, 3) tensor of sequence of the xyz positions of the points
        Returns
        -------
        output : torch.Tensor
            (B, num_classes) tensor of logits for classification
        """
        device = input.device
        points, features = self.tube_embedding(input) # [B, L, n, 3], [B, L, n, C]

        t = torch.arange(points.shape[1], dtype=torch.float32, device=device).view(1, points.shape[1], 1, 1)
        t=t.repeat(points.shape[0], 1, points.shape[2], 1) # [B, L, n, 1]
        points = torch.cat((points, t), dim=-1) # [B, L, n, 4]

        points = points.reshape(points.shape[0], -1, points.shape[-1]) # [B, L*n, 4]

        features = features.reshape(features.shape[0], -1, features.shape[-1]) # [B, L*n, C]

        points = self.pos_embedding(points.permute(0, 2, 1)).permute(0, 2, 1) # [B, L*n, C]


        if self.emb_complex:
            embedding = self.emb_complex(torch.cat((points, features), dim=-1)) # [B, L*n, C]
        else:
            embedding = points + features # [B, L*n, C]

        output = self.transformer(embedding) # [B, L*n, C']
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0] # [B, C']
        output = self.mlp_head(output) # [B, num_classes]

        return output