# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, loss_cls_3d, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import json

import numpy as np
from hyper import myloss
from utils.loss_utils import ssim_multispectral


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    print('train cameras', sorted([cam.image_name for cam in scene.getTrainCameras()]))
    print('test cameras', sorted([cam.image_name for cam in scene.getTestCameras()]))
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    model = gaussians.model

    debug = False
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # print(viewpoint_cam.image_name)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, rendered_embeddings = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], \
            render_pkg["radii"], render_pkg["render_object"]

        # Object Loss
        # gt_obj = viewpoint_cam.objects.cuda().long()
        # logits = classifier(objects)
        # loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        # loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)
        # loss_obj = 0.

        # breakpoint()  # comment out to debug

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = gt_image[:opt.channels]
        # print('gt_image max', gt_image.max())

        # Ll1 = l1_loss(image, gt_image)

        if iteration == first_iter:
            print('image shape {}'.format(image.shape))
            print('gt_image shape {}'.format(gt_image.shape))
            print('rendered_embeddings shape {}'.format(rendered_embeddings.shape))
            #print('features_dc {}'.format(gaussians._features_dc.shape))
            #print('features_rest {}'.format(gaussians._features_rest.shape))

        pred = model(rendered_embeddings)  # dim_out, H, W
        pred = pred[:opt.channels]

        if iteration == first_iter:
            print('pred shape {}'.format(pred.shape))

        Ll12 = myloss(pred, gt_image)
        # print('Ll12 {}'.format(Ll12))

        loss_obj_3d = None
        if opt.reg3d and (iteration % opt.reg3d_interval == 0):
            # regularize at certain intervals
            # logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            logits3d = model(gaussians._objects_dc.permute(2, 0, 1))
            prob_obj3d = torch.softmax(logits3d, dim=0).squeeze().permute(1, 0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val,
                                      opt.reg3d_max_points, opt.reg3d_sample_size)
            # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj \
            #       + loss_obj_3d
            #print(Ll1, loss_obj_3d)
            loss = Ll12 + loss_obj_3d
        else:
            # loss = (1.0 - opt.lambda_dssim) * Ll12 + opt.lambda_dssim * (1.0 - ssim(pred, gt_image)) # + loss_obj   ssim(image,...
            loss = (1.0 - opt.lambda_dssim) * Ll12 + opt.lambda_dssim * (1.0 - ssim(pred, gt_image))
            # loss = Ll12

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "gaussians": f"{gaussians.get_xyz.shape[0]}"})
                # provar progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, testing_iterations, scene, render, (pipe, background), loss_obj_3d, model, opt)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}"
                #                                                  .format(iteration), 'classifier.pth'))
                torch.save(model.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}"
                                                                 .format(iteration), 'model.pth'))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 \
                        or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if debug:
            break  # una sola iteracio


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, testing_iterations, scene: Scene, renderFunc, renderArgs,
                    loss_obj_3d, model, opt):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test',
                               'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            path_out = os.path.join(args.model_path, 'eval', config['name'])
            os.makedirs(path_out, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    rendered_embeddings = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render_object"]
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    gt_image = gt_image[:opt.channels]
                    pred = model(rendered_embeddings)
                    pred = pred[:opt.channels]
                    l1_test += myloss(pred, gt_image).double()
                    #if idx in [0,2,4,6]: #idx == 0:
                    pred = pred.detach().cpu().numpy()
                    with open(os.path.join(path_out, '{}_{}.npy').format(viewpoint.image_name, iteration), 'wb') as f:
                        np.save(f, pred)

                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {}".format(iteration, config['name'], l1_test))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)


    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    # args.densify_until_iter = config.get("densify_until_iter", 15000)  # hides argument in command line
    # args.num_classes = config.get("num_classes", 200)

    # already set at config/gaussian_dataset/train.json (what do they do?)
    # args.reg3d_interval = config.get("reg3d_interval", 2)
    # args.reg3d_k = config.get("reg3d_k", 5)
    # args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    # args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    # args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    args_lp = lp.extract(args)
    args_op = op.extract(args)
    args_pp = pp.extract(args)

    print('args', args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    print("Optimizing " + args.model_path)

    training(args_lp, args_op, args_pp, args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    with open(os.path.join(args.model_path, "all_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    # All done
    print("\nTraining complete.")

