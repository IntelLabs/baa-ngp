# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np
import sys
import typing
import torch

from .ngp import NGPRadianceField

sys.path.append("..")
from camera_utils import cam2world
from lie_utils import se3_to_SE3
from pose_utils import construct_pose, compose_poses
from utils import Rays


class BAradianceField(torch.nn.Module):
    """Bundle-Ajusting radiance field."""
    def __init__(
        self,
        num_frame: int,  # The number of frames.
        aabb: typing.Union[torch.Tensor, typing.List[float]],
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        n_features_per_level: int = 2, 
        log2_hashmap_size: int = 19,
        base_resolution: int = 14,
        max_resolution: int = 4096,
        use_viewdirs: bool = True,
        c2f: typing.Optional[float] = None,
        dof: int = 6, # degree of freedom of input transformation.
        num_input_dim: int = 3,
        device: str = 'cpu',
        testing: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_frame = num_frame
        self.use_viewdirs = use_viewdirs
        self.nerf = NGPRadianceField(aabb=aabb, 
                                     num_dim=num_input_dim,
                                     num_layers=num_layers,
                                     hidden_dim=hidden_dim,
                                     geo_feat_dim=geo_feat_dim,
                                     n_levels=n_levels,
                                     log2_hashmap_size=log2_hashmap_size,
                                     base_resolution=base_resolution,
                                     n_features_per_level=n_features_per_level,
                                     max_resolution=max_resolution,                              
                                     use_viewdirs=use_viewdirs)
        self.c2f = c2f

        # noise addition for blender.
        
        se3_noise = torch.randn(num_frame, dof, device=device) * 0.15
        self.pose_noise = se3_to_SE3(se3_noise)
        # Learnable embedding. 
        self.se3_refine = torch.nn.Embedding(num_frame, dof, device=device)
        torch.nn.init.zeros_(self.se3_refine.weight)
        # TODO:see if we want to register buffer for this variable.
        self.progress = torch.nn.Parameter(torch.tensor(0.))
        self.k = torch.arange(n_levels-1, dtype=torch.float32, device=device)
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.device = device
        self.testing = testing

    def query_density(self, x, occ_sample=False):
        if self.c2f is None or (self.testing and not occ_sample):
            return self.nerf.query_density(x)
        return self.nerf.query_density(x, weights=self.get_weights())


    def get_weights(self):
        c2f_start, c2f_end = self.c2f
        # to prevent all weights to be 0, we leave the features from the first grid alone.
        alpha = (self.progress.data - c2f_start) / (c2f_end - c2f_start) * (self.n_levels-1)
        weight = (1 - (alpha - self.k).clamp_(min=0., max=1.).mul_(np.pi).cos_()) / 2
        weights = torch.cat([torch.ones(self.n_features_per_level, device=self.device), weight.repeat_interleave(self.n_features_per_level)])
        return weights


    def forward(self, positions, directions):
        if self.c2f is None or self.testing:
            return self.nerf(positions, directions)
        return self.nerf(positions, directions, weights=self.get_weights())

    def get_poses(self, idx=None, sim3=None, gt_poses=None, pose_refine_test=None, test_photo=True, mode='train'):
        if mode=='train':
            assert self.training, 'set mode to train during non-training time.'
            pose_noises = self.pose_noise[idx]
            init_poses = compose_poses([pose_noises, gt_poses])
            # add learnable pose correction
            assert idx is not None, "idx cannot be None during training."
            se3_refine = self.se3_refine.weight[idx] # [B, 6] idx corresponds to frame number.
            poses_refine = se3_to_SE3(se3_refine) # [1, 3, 4]
            # add learnable pose correction
            poses = compose_poses([poses_refine, init_poses])
        elif mode in ['val', 'eval', 'test-optim']:
            assert sim3 is not None, 'val, eval, test-optim needs sim3.'
            # align test pose to refined coordinate system (up to sim3)
            center = torch.zeros(1, 1, 3, device=gt_poses.device)
            center = cam2world(center, gt_poses)[:, 0] # [B, HW, 3], [B, 3, 4]
            center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1 # [B, 3]
            R_aligned = gt_poses[...,:3]@sim3.R # [B, 3, 3]
            t_aligned = (-R_aligned@center_aligned[...,None])[...,0] #[B, 3]
            poses = construct_pose(R=R_aligned, t=t_aligned)
            # additionally factorize the remaining pose imperfection
            if test_photo and mode != "val":
                poses = compose_poses([pose_refine_test, poses])
        return poses

    def query_rays(self, grid_3D, idx=None, sim3=None,
                   gt_poses=None, pose_refine_test=None, 
                   test_photo=True, mode='train'):
        """
        Assumption: perspective camera.

        Args:
            grid_3D: 3D grid in camera coordinate (training: pre-sampled).
            idx: frame ids.
            sim3: procruste analysis alignement offset.
            gt_poses: ground truth world-to-camera poses.
            pose_refine_test: test pose codes to be optimized during test evaluation.
            test_photo: whether this is test time optimization.
            mode: "train", "val", "test", "eval", "test-optim"
        """
        poses = self.get_poses(
            idx=idx, sim3=sim3, gt_poses=gt_poses, pose_refine_test=pose_refine_test, test_photo=test_photo,
            mode=mode)
        # given the intrinsic/extrinsic matrices, get the camera center and ray directions
        center_3D = torch.zeros_like(grid_3D) # [B, N, 3]
        # transform from camera to world coordinates
        grid_3D = cam2world(grid_3D, poses) # [B, N, 3], [B, 3, 4] -> [B, 3]
        center_3D = cam2world(center_3D, poses) # [B, N, 3]
        directions = grid_3D - center_3D # [B, N, 3]
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )
        if mode in ['train', 'test-optim']:
            # This makes loss calculate easier. No more reshaping is needed.
            center_3D = torch.reshape(center_3D, (-1, 3))
            viewdirs = torch.reshape(viewdirs, (-1, 3))
        return Rays(origins=center_3D, viewdirs=viewdirs)

    def update_progress(self, progress):
        self.progress.data.fill_(progress)

