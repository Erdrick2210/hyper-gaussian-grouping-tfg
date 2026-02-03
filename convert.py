#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--num_channels", default=3, type=int)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) \
        + " --SiftExtraction.domain_size_pooling=true"
        #+ " --SiftExtraction.estimate_affine_shape=true" + " --SiftExtraction.domain_size_pooling=true"
    # added last 3 to increase number of 3d points, https://colmap.github.io/faq.html

    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu) \
        + " --SiftMatching.guided_matching=true"
    # added last to increase number of 3d points, https://colmap.github.io/faq.html

    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.

for channel in range(args.num_channels):
    image_path_channel = args.source_path + "/channels_distorted/{}".format(channel)
    output_path_channel = args.source_path + "/channels_undistorted/{}".format(channel)
    os.makedirs(output_path_channel, exist_ok=True)
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + image_path_channel + "\
        --input_path " + args.source_path + "/distorted/sparse/0 \
        --output_path " + output_path_channel + "\
        --output_type COLMAP")

    print(img_undist_cmd)

    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # TODO: all files for different channels should be equal, check it with diff
    if channel == 0:
        for file in ['sparse', 'stereo', 'run-colmap-geometric.sh', 'run-colmap-photometric.sh']:
            shutil.move(os.path.join(output_path_channel, file), args.source_path)
    else:
        shutil.rmtree(os.path.join(output_path_channel, 'sparse'))
        shutil.rmtree(os.path.join(output_path_channel, 'stereo'))
        os.remove(os.path.join(output_path_channel, 'run-colmap-geometric.sh'))
        os.remove(os.path.join(output_path_channel, 'run-colmap-photometric.sh'))

# Rebuild multispectral images as npy files, like in
# mms-data_birdhouse/mms-data_demosaicked_undistorted/scenes/birdhouse/modalities/multispectral
# but now using the undistorted images
from PIL import Image  # there's no skimage in this environment
import numpy as np

multispectral_path = args.source_path + "/images"
os.makedirs(multispectral_path, exist_ok=True)
fnames = os.listdir(args.source_path + "/channels_undistorted/0/images")
fnames.sort()
for fn in fnames:
    channels = []
    for nchannel in range(args.num_channels):
        full_name = args.source_path + "/channels_undistorted/{}/images/{}".format(nchannel, fn)
        channel_ima = np.asarray(Image.open(full_name))
        channel_ima = channel_ima[:, :, 0]  # all three RGB channels are equal and in the range [0...255]
        channels.append(channel_ima)

    multispectral_ima = np.stack(channels) / 255.0  # C, H, W as Pytorch wants and in the range [0.0...1.0]
    np.save(os.path.join(multispectral_path, fn.split('.')[0]), multispectral_ima)



"""
files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)
"""
if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")

    # # instead of this, get the file names of the images from the file_names argument
    # with open(os.path.join(args.path_source, args.file_names), 'r') as f:
    #    files = f.read().splitlines()

    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
