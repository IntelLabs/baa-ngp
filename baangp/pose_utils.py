"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""
# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch


def to_hom(X):
    """
    get homogeneous coordinates of the input
    
    Args:
        X: tensor of shape [H*W, 2]
    Returns:
        torch tensor of shape [H*W, 3]
    """
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

    
def construct_pose(R=None,t=None):
    # construct a camera pose from the given R and/or t
    assert(R is not None or t is not None)
    if R is None:
        if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
    elif t is None:
        if not isinstance(R,torch.Tensor): R = torch.tensor(R)
        t = torch.zeros(R.shape[:-1],device=R.device)
    else:
        if not isinstance(R,torch.Tensor): R = torch.tensor(R)
        if not isinstance(t,torch.Tensor): t = torch.tensor(t)
    assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
    R = R.float()
    t = t.float()
    pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
    assert(pose.shape[-2:]==(3,4))
    return pose


def invert_pose(pose,use_inverse=False):
    # invert a camera pose
    R,t = pose[...,:3],pose[...,3:]
    R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
    t_inv = (-R_inv@t)[...,0]
    pose_inv = construct_pose(R=R_inv,t=t_inv)
    return pose_inv


def compose_poses(pose_list):
    # compose a sequence of poses together
    # pose_new(x) = poseN o ... o pose2 o pose1(x)
    pose_new = pose_list[0]
    for pose in pose_list[1:]:
        pose_new = compose_pair(pose_new,pose)
    return pose_new


def compose_pair(pose_a, pose_b):
    # pose_new(x) = pose_b o pose_a(x)
    R_a,t_a = pose_a[...,:3], pose_a[...,3:]
    R_b,t_b = pose_b[...,:3], pose_b[...,3:]
    R_new = R_b@R_a
    t_new = (R_b@t_a+t_b)[...,0]
    pose_new = construct_pose(R=R_new,t=t_new)
    return pose_new