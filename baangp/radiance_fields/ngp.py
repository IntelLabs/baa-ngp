"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Callable, List, Union


try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


class NGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        num_layers: int = 2,
        hidden_dim: int = 64,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        n_features_per_level: int = 2, 
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        """
        Args:
            aabb: region of interest.
            num_dim: input dimension.
            num_layers: number of hidden layers.
            hidden_dim: dimension of hidden layers.
            num_layers_color: number of hidden layers for color network.
            hidden_dim_color: dimension of hidden layers for color network.
            n_features_per_level: number of features per level.
            use_viewdirs: whether or not to use viewdirections
            density_activation: density activation function.
            base_resolution: may be set based on the aabb for 3D case.
            max_resolution: maximum resolution of the hashmap for the base mlp.
            geo_feat_dim: mlp input dimension.
            n_levels: number of levels used for hash grid.
            log2_hashmap_size: hash map size will be defined as 2^log2_hashmap_size. When N^num_dim <= 2^19, the hash grid maps 1:1,
                whereas when N^num_dim > 2^19, there is a chance of collission. N max is 2048, and N min is 16 in the original instant-NGP.
        Returns:
            NGPradianceField instance for bounded scene.
        """
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size
        self.n_features_per_level = n_features_per_level


        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        mlp_encoding_config = {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            }
        network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            }
        self.encoding = tcnn.Encoding(n_input_dims=num_dim, encoding_config=mlp_encoding_config)
        self.mlp_base = tcnn.Network(n_input_dims=self.encoding.n_output_dims, 
                                     n_output_dims=1 + self.geo_feat_dim,
                                     network_config=network_config) 
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    (
                        self.direction_encoding.n_output_dims
                        if self.use_viewdirs
                        else 0
                    )
                    + self.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            )

    def query_density(self, x, weights:torch.Tensor = None, return_feat: bool = False):
        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        encoded_x = self.encoding(x.view(-1, self.num_dim))
        if weights is not None:
            _, n_features = encoded_x.shape
            assert n_features == len(weights)
            # repeats last set of features for 0 mask levels.
        
            available_features = encoded_x[:, weights > 0]
            if len(available_features) <= 0:
                assert False, "no features are selected!"
            assert len(available_features) > 0
            coarse_features = available_features[:, -self.n_features_per_level:]
            coarse_repeats = coarse_features.repeat(1, self.n_levels)
            encoded_x = encoded_x * weights + coarse_repeats * (1 - weights)
        x = (
            self.mlp_base(encoded_x)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
        weights: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(positions, weights=weights, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore

