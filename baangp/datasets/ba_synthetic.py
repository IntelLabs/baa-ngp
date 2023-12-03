"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import imageio.v2 as imageio
import json
import numpy as np
import os
from PIL import Image
import sys
import torch

sys.path.append("..")

from camera_utils import img2cam
from lie_utils import se3_to_SE3
from pose_utils import to_hom, construct_pose, compose_poses, invert_pose


def parse_raw_camera(pose_raw):
    """Convert pose from camera_to_world to world_to_camera and follow the right, down, forward coordinate convention."""
    pose_flip = construct_pose(R=torch.diag(torch.tensor([1,-1,-1]))) # right, up, backward --> right down, forward
    pose = compose_poses([pose_flip, pose_raw[:3]])
    pose = invert_pose(pose) # world_from_camera --> camera_from_world
    return pose
    

def _load_renderings(root_fp: str, subject_id: str, split: str, factor: float):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camfromworld = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        if factor != 1:
            h, w = rgba.shape[:2]
            image = Image.fromarray(rgba)
            resized_image = image.resize((int(w/factor), int(h/factor)))
            rgba = np.array(resized_image)
        w2c = parse_raw_camera(torch.tensor(frame["transform_matrix"], dtype=torch.float32))
        camfromworld.append(w2c)
        images.append(rgba)

    images = torch.from_numpy(np.stack(images, axis=0)).to(torch.uint8)
    camfromworld = torch.stack(camfromworld)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    return images, camfromworld, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    RAW_WIDTH, RAW_HEIGHT = 800, 800

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        factor: float = 1,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        dof: int = 6,
        noise: float = 0.15
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.factor = factor
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.noise = noise
        if split == "trainval":
            _images_train, _camfromworld_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train", factor=factor
            )
            _images_val, _camfromworld_val, _ = _load_renderings(
                root_fp, subject_id, "val", factor=factor
            )
            images = torch.cat([_images_train, _images_val])
            camfromworld = torch.cat(
                [_camfromworld_train, _camfromworld_val]
            )
            self.focal = _focal_train            
        else:
            images, camfromworld, self.focal = _load_renderings(
                root_fp, subject_id, split, factor=factor
            )
        assert images.shape[1:3] == (self.RAW_HEIGHT//factor, self.RAW_WIDTH//factor)
        self.height, self.width = images.shape[1:3]
        K = torch.tensor(
            [
                [self.focal, 0, self.width / 2.0],
                [0, self.focal, self.height / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        self.images = images.to(device)
        self.camfromworld = camfromworld.to(device)
        self.K = K.to(device)
        self.num_rays = num_rays
        self.training = (self.num_rays is not None) and (
            split in ["train", "trainval"]
        )

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba = data["rgba"]
        def rgba_to_pixels(rgba):
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

            if self.training:
                if self.color_bkgd_aug == "random":
                    color_bkgd = torch.rand(3, device=self.images.device)
                elif self.color_bkgd_aug == "white":
                    color_bkgd = torch.ones(3, device=self.images.device)
                elif self.color_bkgd_aug == "black":
                    color_bkgd = torch.zeros(3, device=self.images.device)
            else:
                # just use white during inference
                color_bkgd = torch.ones(3, device=self.images.device)

            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
            return pixels, color_bkgd
        pixels, color_bkgd = rgba_to_pixels(rgba)
        if self.training:
            return {
                "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
                "color_bkgd": color_bkgd,  # [3,]
                **{k: v for k, v in data.items() if k not in ["rgba"]},
            }

        # pre-calculate camera centers in camera coordinate to 
        # save time during training.
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # Compute image coordinate grid.
        if self.training:
            # Randomly select num_ray images and sample one ray per image.
            # allow duplicates, so one image may have more rays.
            if self.batch_over_images:
                assert self.num_rays is not None, "self.training is True, must pass in a num_rays."
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(self.num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.width, size=(self.num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(self.num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
        # adds 0.5 here.
        xy_grid = torch.stack([x,y],dim=-1).view(-1, 2) + 0.5 # [HW,2] or [B, N_rays, 2]
        # self.K is of shape [3,3]
        grid_3D = img2cam(to_hom(xy_grid), self.K) # [B, 2], [3, 3] -> [B, 3]
        images = self.images
        rgba = images[image_id, y, x] / 255.0

        w2c = torch.reshape(self.camfromworld[image_id], (-1, 3, 4)) # [3, 4] or (num_rays, 3, 4)
        if self.training:
            rgba = torch.reshape(rgba, (-1, 4))
            grid_3D = torch.reshape(grid_3D, (-1, 1, 3)) # extra dimension is needed for query_rays.
            return {
                "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
                "grid_3D": grid_3D,  # [h, w, 3] or [num_rays, 3]
                "gt_w2c": w2c, # [num_images, 3, 4]
                "image_id": image_id, # [num_images]]
            }
        else:
            rgba = torch.reshape(rgba, (self.height, self.width, 4))
            grid_3D = torch.reshape(grid_3D, (self.height, self.width, 3))
            return {
                "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
                "grid_3D": grid_3D,  # [h, w, 3] or [num_rays, 3]
                "gt_w2c": w2c, # [num_images, 3, 4]
                "image_id": image_id # [num_images]]
            }


