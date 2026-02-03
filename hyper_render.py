# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def feature_to_rgb(features, method, downsample):
    # Input features shape: (16, H, W)
    print('features {}'.format(features.shape))
    if downsample:
        features = features[:, ::downsample, ::downsample]
        print('downsampled features {}'.format(features.shape))

    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    # features_reshaped = features.view(features.shape[0], -1).T
    features_reshaped = torch.reshape(features, (features.shape[0], H * W)).T
    print('features reshaped {}'.format(features_reshaped.shape))

    if method == 'pca':
        # Apply PCA and get the first 3 components
        pca = PCA(n_components=3)
        result = pca.fit_transform(features_reshaped.cpu().numpy())
    elif method == 't-sne':
        t_sne = TSNE(
            n_components=3,
            perplexity=30,
            init="random",
            n_iter=250,
            random_state=0,
            verbose=3,
        )
        result = t_sne.fit_transform(features_reshaped)
    else:
        assert False, 'invalid method {}'.format(method)

    # Reshape back to (H, W, 3)
    result = result.reshape(H, W, 3)
    # Normalize to [0, 255]
    result_normalized = 255 * (result - result.min()) / (result.max() - result.min())
    rgb_array = result_normalized.astype('uint8')
    return rgb_array


def render_set(render_path, split, iteration, views, gaussians, pipeline, background, model, dim_reduction_method,
               downsample, num_frames):
    gts_path = os.path.join(render_path, split, "gt")
    #embeddings_path = os.path.join(render_path, split, "{}_{}".format(dim_reduction_method, iteration), "embeddings")
    #pred_path = os.path.join(render_path, split, "{}_{}".format(dim_reduction_method, iteration), "pred")
    pred_path = os.path.join(render_path, split, "pred")
    makedirs(gts_path, exist_ok=True)
    #makedirs(embeddings_path, exist_ok=True)
    makedirs(pred_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendered_embeddings = results["render_object"]  # dim_out, H, W
        rendered_embeddings = rendered_embeddings.cpu()
        #dim_reduced_embeddings = feature_to_rgb(rendered_embeddings, dim_reduction_method, downsample)  # rgb
        #Image.fromarray(dim_reduced_embeddings).save(os.path.join(embeddings_path, view.image_name + ".png"))  # '{0:05d}'.format(idx)

        with torch.no_grad():
            pred = model(rendered_embeddings.cuda())  # C, H, W
            pred = pred.cpu()

        #dim_reduced_pred = feature_to_rgb(pred, dim_reduction_method, downsample)
        pred_ch0 = pred[0]  # shape: (H, W)

        pred_ch0 = pred_ch0.numpy()
        pred_ch0 = (pred_ch0 - pred_ch0.min()) / (pred_ch0.max() - pred_ch0.min() + 1e-8)
        pred_ch0 = (pred_ch0 * 255).astype(np.uint8)

        #Image.fromarray(dim_reduced_pred).save(os.path.join(pred_path, view.image_name + ".png")) # '{0:05d}'.format(idx)
        Image.fromarray(pred_ch0, mode="L").save(os.path.join(pred_path, view.image_name + ".png"))
        #np.save(os.path.join(pred_path, view.image_name), pred)

        gt = view.original_image.cpu()
        gt_ch0 = gt[0]
        gt_ch0 = gt_ch0.numpy()
        gt_ch0 = (gt_ch0 - gt_ch0.min()) / (gt_ch0.max() - gt_ch0.min() + 1e-8)
        gt_ch0 = (gt_ch0 * 255).astype(np.uint8)
        Image.fromarray(gt_ch0, mode="L").save(os.path.join(gts_path, view.image_name + ".png"))
        #np.save(os.path.join(gts_path, view.image_name), gt)

        if idx == num_frames:
            break


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, op: OptimizationParams,
                skip_train: bool, skip_test: bool, method_dim_reduction: str, downsample: int, num_frames: int) :
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.training_setup(op)

        model = gaussians.model
        model.load_state_dict(torch.load(
            os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "model.pth")))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(os.path.join(dataset.model_path, "render"), "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipeline, background, model, method_dim_reduction, downsample, num_frames)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(os.path.join(dataset.model_path, "render"), "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, pipeline, background, model, method_dim_reduction, downsample, num_frames)

        #scene.save_ply(iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--method_dim_reduction", choices=["t-sne", "pca"], default="t-sne")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    args_lp = lp.extract(args)
    args_op = op.extract(args)
    args_pp = pp.extract(args)

    render_sets(args_lp, args.iteration, args_pp, args_op,
                args.skip_train, args.skip_test, args.method_dim_reduction, args.downsample, args.num_frames)
