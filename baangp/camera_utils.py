"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from easydict import EasyDict as edict
import numpy as np
from pose_utils import invert_pose, to_hom, construct_pose, compose_poses
import torch


# basic operations of transforming 3D points between world/camera/image coordinates
def cam2world(X, pose):
    """
    Args:
        X: 3D points in camera space with [H*W, 3]
        pose: camera pose. camera_from_world, or world_to_camera [3, 3]
    Returns:
        transformation in world coordinate system.
    """
    # X of shape 64x3
    # X_hom is of shape 64x4
    X_hom = to_hom(X)
    # pose_inv is world_from_camera pose is of shape 3x4
    pose_inv = invert_pose(pose)
    # world = camera * world_from_camera
    return X_hom@pose_inv.transpose(-1,-2)


def img2cam(X, cam_intr):
    """
    Args:
        X: 3D points in image space.
        cam_intr: camera intrinsics. camera_from_image
    Returns:
        trasnformation in camera coordinate system
    """
    # camera = image * image_from_camera
    return X@cam_intr.inverse().transpose(-1,-2)


def procrustes_analysis(X0,X1): # [N,3]
    """ procruste analysis: statistical shape analysis used to analyse the distribution of a set of shapes.

    Often, these points are selected on the continuous surface of complex objects, such as a human bone,
    and in this case they are called landmark points. The shape of an object can be considered as a member 
    of an equivalence class formed by removing the translational, rotational and uniform scaling components.

    Reference: https://www.youtube.com/watch?v=nnJvBa_ERL0
    Assume the point correspondences are known. subject to RR^T = I, det(R) = 1 for R*, T* where argmin(|Ai-RBi - T|^2) for point i.
    differentiate by translation and set to zero, we have:
    T* = bar(A) - R bar(B), where bar is the average points.
    substitue translation into objective function, we have
    R* = argmin(|Ai - RBi - (bar(A) - R bar(B))|^2) = argmin(|(Ai - bar(A)) - R (Bi - bar(B))|^2)
    here Ai - bar(A) and Bi - bar(B) are centering the point sets.
    After centering, estimate remaining rotation between point sets.
    R* = argmin|Ai - RBi|^ subject to RR^T = I, det(R) = 1 becomes an orthogonal procrustes problem.
    due to invariant to cyclic permutation of matrix products tr(ABC) = tr(CAB) = tr(BCA)
    expand the function square
    R* = argmin[tr(Ai^TAi) + tr((RBi)^T(RBi)) - 2 tr(Ai^TRBi)] where tr(Bi^TR^TRBi) = tr(Bi^TBi)
    = argmax tr(Ai^TRBi)
    = argmax tr(RBiAi^T)
    compute svd of BiAi^T,  
    A = UDV^T, where mxn (orthogonal columns), nxn (non-negative entries, singular values, descending order), nxn (orthogonal matrix)
    R* = argmax tr(RUDV^T) = argmax tr(V^T R UD)
    tr(V^TRUD) = tr(ZD) = sum_i^N zii Dii, matrix Z is orthogonal since it is a product of three orthogonal matrices.
    Diagonal elements of Z are less than or equal to 1. The rotation objective is maximized when Z = I.
    Z = V^T RU = I
    T* = bar(A) - R* bar(B)
    R* = VU^T
    BiAi^T = UDV^T 
    """
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0 # centering points
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0 # scaling points
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()

    # If R.det < 0 --> improper rotations with reflections.
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3


def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle



def get_novel_view_poses(pose_anchor,N=60,scale=1):
    # create circular viewpoints (small oscillations)
    theta = torch.arange(N)/N*2*np.pi
    R_x = angle_to_rotation_matrix((theta.sin()*0.05).asin(),"X")
    R_y = angle_to_rotation_matrix((theta.cos()*0.05).asin(),"Y")
    pose_rot = construct_pose(R=R_y@R_x)
    pose_shift = construct_pose(t=[0,0,-4*scale])
    pose_shift2 = construct_pose(t=[0,0,3.8*scale])
    pose_oscil = compose_poses([pose_shift,pose_rot,pose_shift2])
    pose_novel = compose_poses([pose_oscil,pose_anchor.cpu()[None]])
    return pose_novel


def angle_to_rotation_matrix(a,axis):
    # get the rotation matrix from Euler angle around specific axis
    roll = dict(X=1,Y=2,Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack([torch.stack([a.cos(),-a.sin(),O],dim=-1),
                     torch.stack([a.sin(),a.cos(),O],dim=-1),
                     torch.stack([O,O,I],dim=-1)],dim=-2)
    M = M.roll((roll,roll),dims=(-2,-1))
    return M