# Copyright 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import argparse
from easydict import EasyDict as edict
from lpips import LPIPS
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
import torchvision.transforms.functional as torchvision_F
import tqdm
from datasets.ba_synthetic import SubjectLoader
from evaluation_utils import (
    evaluate_camera_alignment,
    evaluate_test_time_photometric_optim,
    prealign_cameras
)
from lie_utils import se3_to_SE3
from nerfacc.estimators.occ_grid import OccGridEstimator
from pose_utils import compose_poses
from radiance_fields.baangp import BAradianceField
from utils import (
    render_image_with_occgrid,
    set_random_seed,
    load_ckpt,
    save_ckpt
)
import visualization_utils as viz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        help="the root dir of the dataset",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=SubjectLoader.SUBJECT_IDS,
        help="which scene to use",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="random seed"
    )

    parser.add_argument(
        "--c2f",
        type=float,
        nargs="+",
        action="extend"
    )

    parser.add_argument("--save-dir", type=str,
        required=True,
        help="The output root directory for saving models.")
    args = parser.parse_args()

    device = "cuda:0"
    
    set_random_seed(args.seed)
    if os.path.exists(args.save_dir):
        print('%s exists!'%args.save_dir)
    else:
        print("Creating %s"%args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    # training parameters
    lr = 1.e-2
    lr_end = 1.e-4
    lr_pose = 1.e-3 #1.e-2
    lr_pose_end = 1.e-5 #1.e-3
    optim_lr_pose = 1.e-3
    max_steps = 40000 # 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 1e-6
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"factor": 2}
    test_dataset_kwargs = {"factor": 2}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        batch_over_images=True,
        device=device,
        **train_dataset_kwargs,
    )
    print("Found %d train images"%len(train_dataset.images))
    print("Train image shape", train_dataset.images.shape)
    print("Setup the test dataset.")
    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        device=device,
        batch_over_images=False,
        **test_dataset_kwargs,
    )
    print("Found %d test images."%len(test_dataset.images))
    print("Test image shape", test_dataset.images.shape)
    print(f"Setup Occupancy Grid. Grid resolution is {grid_resolution}")

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)

    # setup the radiance field we want to train.
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    
    radiance_field = BAradianceField(
        num_frame=len(train_dataset),
        aabb=estimator.aabbs[-1],
        device=device,
        c2f=args.c2f,
        ).to(device)
    
    print("Setting up optimizers...")
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=lr, eps=1e-15, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=(lr_end/lr)**(1./max_steps)
            )
    models = {"radiance_field": radiance_field, "estimator": estimator}
    schedulers={"scheduler": scheduler}
    optimizers={"optimizer": optimizer}

    pose_optimizer = torch.optim.Adam(models['radiance_field'].se3_refine.parameters(), lr=lr_pose)
    pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        pose_optimizer,
        gamma=(lr_pose_end/lr_pose)**(1./max_steps)
    )
    schedulers["pose_scheduler"] = pose_scheduler
    optimizers["pose_optimizer"] = pose_optimizer

    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()


    models, optimizers, schedulers, epoch, iteration, has_checkpoint = load_ckpt(save_dir=args.save_dir, models=models, optimizers=optimizers, schedulers=schedulers)
    # training
    if not has_checkpoint:
        tic = time.time()
        loader = tqdm.trange(max_steps + 1, desc="training", leave=False)
        for step in loader:
            models['radiance_field'].train()
            models["estimator"].train()

            i = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            pixels = data["pixels"]
            grid_3D = data["grid_3D"]
            gt_poses = data["gt_w2c"] # [num_ray, 3, 4]
            image_ids = data["image_id"]
            
            if args.c2f is not None:
                models["radiance_field"].update_progress(step/max_steps)

            def occ_eval_fn(x):
                density = models["radiance_field"].query_density(x)
                return density * render_step_size

            # update occupancy grid
            models["estimator"].update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2,
            )

            for key in optimizers:
                # setting gradient to None to avoid extra computations.
                optimizers[key].zero_grad(set_to_none=True)

            # query rays

            rays = models["radiance_field"].query_rays(idx=image_ids, grid_3D=grid_3D, gt_poses=gt_poses, mode='train')
            # render
            rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                radiance_field=models["radiance_field"],
                estimator=models["estimator"],
                rays=rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )
            if n_rendering_samples == 0:
                loader.set_postfix(it=step, loss="skipped")
                continue

            if target_sample_batch_size > 0:
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays * (target_sample_batch_size / float(n_rendering_samples))
                )
                train_dataset.update_num_rays(num_rays)

            # compute loss
            loss = F.smooth_l1_loss(rgb, pixels)
            # do not unscale it because we are using Adam.
            scaled_train_loss = grad_scaler.scale(loss)
            scaled_train_loss.backward()
            for key in optimizers:
                optimizers[key].step()
            for key in schedulers:
                schedulers[key].step()
            loader.set_postfix(it=step, loss="{:.4f}".format(scaled_train_loss[0]))

            if step % 10000 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(loss) / np.log(10.0)
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | psnr={psnr:.2f} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                    f"max_depth={depth.max():.3f} | "
                )
        save_ckpt(save_dir=args.save_dir, iteration=step, models=models, optimizers=optimizers, schedulers=schedulers, final=True)
    else:
        step = iteration
    # evaluation
    print("Done training, start evaluation:")
    models["radiance_field"].eval()
    models["estimator"].eval()
    models['radiance_field'].testing = True

    with torch.no_grad():
        print("Plotting final pose alignment.")
        pose_refine = se3_to_SE3(models["radiance_field"].se3_refine.weight)
        gt_poses = train_dataset.camfromworld
        pred_poses = compose_poses([pose_refine, models["radiance_field"].pose_noise, gt_poses])
        pose_aligned, sim3 = prealign_cameras(pred_poses, gt_poses)
        error = evaluate_camera_alignment(pose_aligned, gt_poses)
        rot_error = np.rad2deg(error.R.mean().item())
        trans_error = error.t.mean().item()
        print("--------------------------")
        print("{} train rot error:   {:8.3f}".format(step, rot_error)) # to use numpy, the value needs to be on cpu first.
        print("{} train trans error: {:10.5f}".format(step, trans_error))
        print("--------------------------")
        # dump numbers
        quant_fname = os.path.join(args.save_dir, "quant_pose.txt")
        with open(quant_fname,"w") as file:
            for i, (err_R, err_t) in enumerate(zip(error.R, error.t)):
                file.write("{} {} {}\n".format(i, err_R.item(), err_t.item()))
    
        pose_aligned_detached, gt_poses_detached = pose_aligned.detach().cpu(), gt_poses.detach().cpu()
        fig = plt.figure(figsize=(10, 10))
        cam_dir = os.path.join(args.save_dir, "poses")
        os.makedirs(cam_dir, exist_ok=True)
        png_fname = viz.plot_save_poses_blender(fig=fig,
                                                pose=pose_aligned_detached, 
                                                pose_ref=gt_poses_detached, 
                                                path=cam_dir, 
                                                ep=step)
    
    # evaluate novel view synthesis
    test_dir = os.path.join(args.save_dir, "test_pred_view")
    os.makedirs(test_dir,exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    res = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        gt_poses, pose_refine_test = evaluate_test_time_photometric_optim(
            radiance_field=models["radiance_field"], estimator=models["estimator"],
            render_step_size=render_step_size,
            cone_angle=cone_angle,
            data=data, 
            sim3=sim3, lr_pose=optim_lr_pose, test_iter=100,
            alpha_thre=alpha_thre,
            device=device,
            near_plane=near_plane,
            far_plane=far_plane,
            )
        with torch.no_grad():
            rays = models["radiance_field"].query_rays(idx=None,
                                                       sim3=sim3, 
                                                       gt_poses=gt_poses, 
                                                       pose_refine_test=pose_refine_test, 
                                                       mode='eval',
                                                       test_photo=True,
                                                       grid_3D=data['grid_3D'])
            # rendering
            rgb, opacity, depth, _ = render_image_with_occgrid(
                # scene
                radiance_field=models["radiance_field"],
                estimator=models["estimator"],
                rays=rays,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd = data["color_bkgd"],
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )
            # evaluate view synthesis
            invdepth = 1/ depth
            loaded_pixels = data["pixels"]
            h, w, c = loaded_pixels.shape
            pixels = loaded_pixels.permute(2, 0, 1)
            rgb_map = rgb.view(h, w, 3).permute(2, 0, 1)
            invdepth_map = invdepth.view(h, w)
            mse = F.mse_loss(rgb_map, pixels)
            psnr = (-10.0 * torch.log(mse) / np.log(10.0)).item()
            ssim_val = ssim(rgb_map[None, ...], pixels[None, ...]).item()
            ms_ssim_val = ms_ssim(rgb_map[None, ...], pixels[None, ...]).item()
            lpips_loss_val = lpips_fn(rgb, loaded_pixels).item()
            res.append(edict(psnr=psnr, ssim=ssim_val, ms_ssim=ms_ssim_val, lpips=lpips_loss_val))
            # dump novel views
            rgb_map_cpu = rgb_map.cpu()
            gt_map_cpu = pixels.cpu()
            depth_map_cpu = invdepth_map.cpu()
            torchvision_F.to_pil_image(rgb_map_cpu).save("{}/rgb_{}.png".format(test_dir,i))
            torchvision_F.to_pil_image(gt_map_cpu).save("{}/rgb_GT_{}.png".format(test_dir,i))
            torchvision_F.to_pil_image(depth_map_cpu).save("{}/depth_{}.png".format(test_dir,i))
            
    plt.close()
    # show results in terminal
    avg_psnr = np.mean([r.psnr for r in res])
    avg_ssim = np.mean([r.ssim for r in res])
    avg_ms_ssim = np.mean([r.ms_ssim for r in res])
    avg_lpips = np.mean([r.lpips for r in res])
    print("--------------------------")
    print("PSNR:  {:8.2f}".format(avg_psnr))
    print("SSIM:  {:8.3f}".format(avg_ssim))
    print("MS-SSIM:  {:8.3f}".format(avg_ms_ssim))
    print("LPIPS: {:8.3f}".format(avg_lpips))
    print("--------------------------")
    # dump numbers to file
    quant_fname = os.path.join(args.save_dir, "quant.txt")
    with open(quant_fname,"w") as file:
        for i,r in enumerate(res):
            file.write("{} {} {} {} {}\n".format(i, r.psnr, r.ssim, r.ms_ssim, r.lpips))
            
    # assume the test view synthesis are already generated
    print("writing videos...")
    rgb_vid_fname = os.path.join(args.save_dir, "test_view_rgb.mp4")
    depth_vid_fname = os.path.join(args.save_dir, "test_view_depth.mp4")
    os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_dir, rgb_vid_fname))
    os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_dir, depth_vid_fname))

    print("Training and evaluation stops.")
        