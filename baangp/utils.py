"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
import random
from typing import Optional, Dict


import numpy as np
import torch
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.volrend import (
    rendering,
)
import collections

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple.
    """
    return type(tup)(*(None if x is None else fn(x) for x in tup))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)

        rays_o = chunk_rays.origins
        rays_d = chunk_rays.viewdirs

        def sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                if timestamps is not None:
                    # dnerf
                    t = (
                        timestamps[ray_indices]
                        if radiance_field.training
                        else timestamps.expand_as(positions[:, :1])
                    )
                    sigmas = radiance_field.query_density(positions, t)
                else:
                    sigmas = radiance_field.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=t_starts.device)
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                if timestamps is not None:
                    # dnerf
                    t = (
                        timestamps[ray_indices]
                        if radiance_field.training
                        else timestamps.expand_as(positions[:, :1])
                    )
                    rgbs, sigmas = radiance_field(positions, t, t_dirs)
                else:
                    rgbs, sigmas = radiance_field(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_o.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )


def load_ckpt(save_dir:str, models: Dict = {}, 
              optimizers: Dict = {},
              schedulers: Dict = {},
              iteration: int = None):
    # check if any ckpt file exists.
    if iteration is not None:
        checkpoint_path = os.path.join(save_dir, "model", f"{iteration}.ckpt")
    else:
        checkpoint_path = os.path.join(save_dir, "model.ckpt")
    if os.path.exists(checkpoint_path):
        print(f"{checkpoint_path} exist, Loading checkpoint.")
        checkpoint = torch.load(checkpoint_path)
        # return the loaded model and the step size.
    else:
        checkpoint = None
    if checkpoint is not None:
        for key in models:
            if isinstance(checkpoint[key], list):
                for i, state_dict in enumerate(checkpoint[key]):
                    models[key][i].load_state_dict(state_dict)
            else:
                models[key].load_state_dict(checkpoint[key])
        for key in optimizers:
            optimizers[key].load_state_dict(checkpoint[key])
        for key in schedulers:
            schedulers[key].load_state_dict(checkpoint[key])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            epoch = None
        if 'iter' in checkpoint:
            iteration = checkpoint['iter']
        else:
            iteration = None
    else:
        print("No checkpoint is found.")
        epoch = None
        iteration = None
        return models, optimizers, schedulers, epoch, iteration, False
    print("Loaded from ")
    if iteration is not None:
        print(f"iteration {iteration} ")
    if epoch is not None:
        print(f"epoch {epoch} ")
    if iteration is None and epoch is None:
        print("final model")
    return models, optimizers, schedulers, epoch, iteration, True


def save_ckpt(models: Dict,
              save_dir: str,
              epoch: int = None,
              iteration: int = None,
              optimizers: Dict = None,
              schedulers: Dict = None, 
              final: bool = False) -> None:
    """
    Save model to the save directory
    """
    if final:
        model_name = "model.ckpt"
    else:
        if iteration is not None:
            model_name = os.path.join("model", f"{iteration}.ckpt")
            os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
        elif epoch is not None:
            model_name = os.path.join("model", f"{epoch}.ckpt")
            os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
        else:
            model_name = "model.ckpt"

    save_model_path = os.path.join(save_dir, model_name)
    checkpoint = {}
    if epoch is not None:
        checkpoint['epoch'] = epoch + 1
    if iteration is not None:
        checkpoint['iter'] = iteration

    for key in models.keys():
        if isinstance(models[key], list):
            modules = []
            for module in models[key]:
                modules.append(module.state_dict())
            checkpoint[key] = modules
        else:
            checkpoint[key] = models[key].state_dict()
    if optimizers is not None:
        for key in optimizers:
            checkpoint[key] = optimizers[key].state_dict()
    if schedulers is not None:
        for key in schedulers:
            checkpoint[key] = schedulers[key].state_dict()
    assert not os.path.exists(save_model_path), f"{save_model_path} already exists!"
    torch.save(checkpoint, save_model_path)
    if final:
        print("Model is saved to %s"% save_model_path)
    else:
        print("Model ")
        if epoch is not None:            
            print(f"epoch {epoch} ")
        if iteration is not None:
            print(f"iteration {iteration} ")
        print("is saved to %s"% save_model_path)
