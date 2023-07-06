import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import pointnet2_utils

class P4DConv(nn.Module):
    def __init__(
        self,
        mlp_plane: int,
        spatial_kernel_size: [float, int],
        spatial_stride: int,
        temporal_kernel_size: int,
        temporal_stride: int=1,
        temporal_padding: [int, int] = None,
        bias: bool = False
    ):
        super().__init__()

        if temporal_padding is None:
            temporal_padding = [0,0]

        self.mlp_plane = mlp_plane
        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.bias = bias

        self.conv_d = nn.Conv2d(
            in_channels=4, out_channels=self.mlp_plane, kernel_size=1, stride=1, padding=0, bias=self.bias
        )



    def forward(self, points: torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        points : torch.Tensor
            (B, T, N, 3) tensor of sequence of the xyz positions of the points
        Returns
        -------
        new_points : torch.Tensor
            (B, T, N, 3) tensor of sequence of the xyz positions of the output points
        new_features : torch.Tensor
            (B, T, N, C) tensor of sequence of the features of the output points
        """
        device = points.device
        
        nframes=points.shape[1]
        npoints=points.shape[2]

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        points=torch.split(tensor=points, split_size_or_sections=1, dim=1)
        points=[torch.squeeze(input=point, dim=1).contiguous() for point in points] # (T, B, N, 3)

        points=[points[0]]*self.temporal_padding[0]+points+[points[0]]*self.temporal_padding[1] # (T+P, B, N, 3)

        new_points=[]
        new_features = []
        for t in range(self.temporal_kernel_size//2, len(points)-self.temporal_kernel_size//2, self.temporal_stride):
            anchor_idx=pointnet2_utils.furthest_point_sample(points[t], npoints//self.spatial_stride) # (B, N//S)
            anchor_points=pointnet2_utils.gather_operation(points[t].transpose(1, 2).contiguous(), anchor_idx) # (B, 3, N//S)
            anchor_points_expanded=torch.unsqueeze(anchor_points, 3) # (B, 3, N//S, 1)
            anchor_points=anchor_points.transpose(1, 2).contiguous() # (B, N//S, 3)

            new_feature=[]
            for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                neighbor_points=points[i] # (B, N, 3)

                idx=pointnet2_utils.ball_query(self.r, self.k, neighbor_points, anchor_points) # (B, N//S, K)

                neighbor_points=pointnet2_utils.grouping_operation(neighbor_points.transpose(1, 2).contiguous(), idx) # (B, 3, N//S, K)

                points_displacement=neighbor_points-anchor_points_expanded # (B, 3, N//S, K)
                t_displacement=(i - t) * torch.ones_like(points_displacement[:, :1], device=device) # (B, 1, N//S, K)
                displacement = torch.cat((points_displacement, t_displacement), dim=-3) # (B, 4, N//S, K)

                feature = self.conv_d(displacement) # (B, C, N//S, K) f(dx, dy, dz, dt)=Wd(dx, dy, dz, dt)^T
                # feature = self.mlp(feature)
                feature = torch.max(feature, dim=-1, keepdim=False)[0] # (B, C, N//S)
                new_feature.append(feature)
            
            new_feature = torch.stack(tensors=new_feature, dim=1) # (B, T, C, N//S)
            new_feature = torch.max(new_feature, dim=1, keepdim=False)[0] # (B, C, N//S)

            new_points.append(anchor_points)
            new_features.append(new_feature.transpose(1, 2))

        new_points = torch.stack(tensors=new_points, dim=1) # (B, T', N//S, 3)
        new_features = torch.stack(tensors=new_features, dim=1) # (B, T', N//S, C)

        return new_points, new_features