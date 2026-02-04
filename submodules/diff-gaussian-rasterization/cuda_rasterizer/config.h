/*
 * Copyright (C) 2023, Gaussian-Grouping
 * Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
 * All rights reserved.
 * ------------------------------------------------------------------------
 * Modified from codes in Gaussian-Splatting 
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_OBJECTS 24 // 8 16 24 32 // Default 16, identity encoding // 38 maximum
// To be changed in :
// submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h
// arguments/__init__.py
// scene/gaussian_model.py
// esborrar directori submodules/diff-gaussian-rasterization/build
// then hyper-gaussian-grouping-tfg/submodules/diff-gaussian-rasterization$ pip install .
// si no funciona fer:
// export CC=/usr/bin/gcc-9
// export CXX=/usr/bin/g++-9
// export CUDAHOSTCXX=/usr/bin/g++-9
#define BLOCK_X 16
#define BLOCK_Y 16

#endif