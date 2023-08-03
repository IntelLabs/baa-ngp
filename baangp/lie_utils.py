"""
Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License


import torch


def so3_to_SO3(w): # [...,3]
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[...,None,None]
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    R = I+A*wx+B*wx@wx
    return R

def SO3_to_so3(R,eps=1e-7): # [...,3,3]
    trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
    theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
    lnR = 1/(2*taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1))
    w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
    w = torch.stack([w0,w1,w2],dim=-1)
    return w

def se3_to_SE3(wu): # [...,3]
    w,u = wu.split([3,3],dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[...,None,None]
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    Rt = torch.cat([R,(V@u[...,None])],dim=-1)
    return Rt

def SE3_to_se3(Rt,eps=1e-8): # [...,3,4]
    R,t = Rt.split([3,1],dim=-1)
    w = SO3_to_so3(R)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[...,None,None]
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
    u = (invV@t)[...,0]
    wu = torch.cat([w,u],dim=-1)
    return wu    

def so3_t3_to_SE3(w, u):
    R = so3_to_SO3(w)
    Rt = torch.cat([R,u[...,None]],dim=-1)
    return Rt

def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                        torch.stack([w2,O,-w0],dim=-1),
                        torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx

def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
    
def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans


def so2_to_SO2(theta): # [...,1]
    thetax = torch.stack([torch.cat([theta.cos(), -theta.sin()], dim=-1),
                          torch.cat([theta.sin(), theta.cos()], dim=-1)], dim=-2)
    R = thetax
    return R


def se2_to_SE2(delta): # [...,3]
    u,theta = delta.split([2,1], dim=-1)
    A = taylor_A(theta)
    B = taylor_B(theta)
    V = torch.stack([torch.cat([A,-B], dim=-1),
                     torch.cat([B,A], dim=-1)], dim=-2)
    R = so2_to_SO2(theta)
    Rt = torch.cat([R,V@u[...,None]],dim=-1)
    return Rt


def sl3_to_SL3(h):
    # homography: directly expand matrix exponential
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
    h1,h2,h3,h4,h5,h6,h7,h8 = h.chunk(8,dim=-1)
    A = torch.stack([torch.cat([h5,h3,h1],dim=-1),
                     torch.cat([h4,-h5-h6,h2],dim=-1),
                     torch.cat([h7,h8,h6],dim=-1)],dim=-2)
    # computes the matrix exponential of a square matrix.
    # The exponential of X is well-defined.
    H = A.matrix_exp()
    return H