"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
import torch.nn.functional as F
import tqdm
from utils import (
    render_image_with_occgrid,
)
from pose_utils import construct_pose
from camera_utils import cam2world, rotation_distance, procrustes_analysis
from lie_utils import se3_to_SE3
from easydict import EasyDict as edict
from utils import Rays


def prealign_cameras(pred_poses, gt_poses):
    """
    Pre-align predicted camera poses with gt camera poses so that they are adjusted to similar
    coordinate frames. This is helpful for evaluation.
    Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py#L107

    Args:
        pred_poses: predicted camera transformations. Tensor of shape [N, 3, 4]
        gt_poses: ground truth camera transformations. Tensor of shape [N, 3, 4]
    Returns:
        pose_aligned: aligned predicted camera transformations.
        sim3: 3D similarity transform information. Dictionary with keys 
            ['t0', 't1', 's0', 's1', 'R']. 
    """
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3,device=pred_poses.device)
    pred_centers = cam2world(center, pred_poses)[:,0] # [N,3]
    gt_centers = cam2world(center, gt_poses)[:,0] # [N,3]
    try:
        # sim3 has keys of ['t0', 't1', 's0', 's1', 'R']
        sim3 = procrustes_analysis(gt_centers, pred_centers)
    except:
        print("warning: SVD did not converge for procrustes_analysis...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3,device=pred_poses.device))
    # align the camera poses
    center_aligned = (pred_centers-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pred_poses[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose_aligned = construct_pose(R=R_aligned, t=t_aligned)
    return pose_aligned, sim3


def evaluate_camera_alignment(pose_aligned,pose_GT):
    """
    Evaluate camera pose estimation with average Rotation and Translation errors.
    Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py#L125
    """
    # measure errors in rotation and translation
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned-t_GT)[..., 0].norm(dim=-1)
    error = edict(R=R_error, t=t_error)
    return error


def evaluate_test_time_photometric_optim(
    radiance_field, estimator, render_step_size, alpha_thre, 
    cone_angle, data, sim3, lr_pose, device, near_plane, far_plane, test_iter=100,
    writer=None):
    """
    Evaluate test time photometric optim.
    Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py#L154
    """
    # lr_pose is 1.e-3  for synthetic, 3.e-3 for llff.
    # use another se3 Parameter to absorb the remaining pose errors
    se3_refine_test = torch.nn.Parameter(torch.zeros(6, device=device))
    pose_optimizer = torch.optim.Adam([se3_refine_test], lr=lr_pose) 
    iterator = tqdm.trange(test_iter, desc="test-time optim.", leave=False,position=1)
    grid_3D = data["grid_3D"]
    pixels = data["pixels"]
    gt_poses = data['gt_w2c']
    h, w, c = pixels.shape
    pixels = pixels.reshape((h*w, c))
    render_bkgd = data["color_bkgd"]
    radiance_field.eval()
    estimator.eval()
    for it in iterator:
        pose_refine_test = se3_to_SE3(se3_refine_test)
        # During test optimization, no training on 
        rays = radiance_field.query_rays(grid_3D=grid_3D,
                                         idx=None, # won't be used for optimization on test views.
                                         sim3=sim3,
                                         gt_poses=gt_poses,
                                         pose_refine_test=pose_refine_test,
                                         test_photo=True,
                                         mode='test-optim')
        origins = rays.origins
        viewdirs = rays.viewdirs
        ray_idx = torch.randperm(len(origins), device=device)[:1024]
        sampled_origins = origins[ray_idx]
        sampled_viewdirs = viewdirs[ray_idx]
        sampled_rays = Rays(origins=sampled_origins, viewdirs=sampled_viewdirs)
        rgb, _, _, n_rendering_samples = render_image_with_occgrid(
            # scene
            radiance_field=radiance_field,
            estimator=estimator,
            rays=sampled_rays,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if n_rendering_samples == 0:
            continue
        pose_optimizer.zero_grad()
        loss = F.smooth_l1_loss(rgb, pixels[ray_idx])
        if writer is not None:
            writer.add_scalar("eval_optim_loss", loss, it)
        loss.backward()
        pose_optimizer.step()
        iterator.set_postfix(loss="{:.3f}".format(loss))
    return gt_poses, pose_refine_test

