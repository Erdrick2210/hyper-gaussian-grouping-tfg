#!/bin/bash

# conda activate gaussian_grouping
# bash hyper.sh

SOURCE_PATH='data/multi-modal-studio/birdhouse'
DATASET_NAME='birdhouse'
NUM_CHANNELS=9 # canviar en arguments/__init__.py tambe pel train

PATH_MAGICK="/usr/local/bin/magick"

CONVERT=true
TRAIN=false
RENDER=false
MAX_ITERS="70000"  # "120000" "70000" "30000" "500"

# si es canvia la mida de l'embedding s'ha de modificar en:
# submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h
# arguments/__init__.py
# scene/gaussian_model.py

# aquesta variable es per separar el tipus d'experiment per dataset en la carpeta output. Per exemple: basement_embedding_24 o basement_ex5
EXPERIMENT="embedding_24" # "embedding_8" "embedding_16" "embedding_24" "embedding_32"
#EXPERIMENT="ex5" # "ex1" "ex2" "ex3" "ex4" "ex5" "ex6"

GPU=0

if $TRAIN; then
    ITERS="100 1000 2000 5000 10000 20000 30000 50000 70000 80000 90000 100000 110000"$MAX_ITERS
    DIM_EMBEDDING=24  # 8, 16, 24, 32
    # si es canvia la mida de l'embedding s'ha de modificar en:
    # submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h
    # arguments/__init__.py
    # scene/gaussian_model.py   
fi

if $RENDER; then
    METHOD_DIM_REDUCTION="t-sne"  # pca t-sne
    DOWNSAMPLE=8 #2
    NUM_FRAMES=4
fi

# previously:
# hyper-gaussian-grouping-tfg/data/multi-modal-studio$ python -i mms.py
# data/birdhouse should have input/ with the rgb images made of 3 channels each, and channels_distorted with each channel
# of each view in a different image. This is done by data/multi-modal-studio/mms.py

if $CONVERT; then
    python convert.py \
      --no_gpu \
      --source_path $SOURCE_PATH \
      --magick_executable $PATH_MAGICK \
      --num_channels $NUM_CHANNELS #\
      #--skip_matching
fi

if $TRAIN; then
    # python -m pdb hyper_train.py  \ comment out to debug, and insert sentence breakpoint() in python to add a breakpoint
    CUDA_VISIBLE_DEVICES=$GPU python hyper_train.py  \
      --source_path $SOURCE_PATH \
      --model_path output/${DATASET_NAME}_${EXPERIMENT} \
      --config_file config/gaussian_dataset/train.json \
      --iterations $MAX_ITERS \
      --test_iterations $ITERS \
      --save_iterations $ITERS \
      --eval \
      --checkpoint_iterations $ITERS #\
      #--start_checkpoint output/${DATASET_NAME}_${EXPERIMENT}/chkpnt30000.pth

    # mes parametres a
    #  arguments/__init__.py in classes ModelParams, OptimizationParams, PipelineParams, HyperParams
    #  config/gaussian_dataset/train.json
    #  potser encara a hyper_train.py

    # a scene/dataset_readers.py, a la funcio readColmapSceneInfo(), llffhold=8 vol dir que agafa 1 de cada 8 imatges/camares
    # (no random sino de 8 en 8 començant per la primera) per test i la resta per train si: es posa --eval i no es
    # posa --train_split
fi

# Embedding and prediction rendering using trained model
if $RENDER; then
    CUDA_VISIBLE_DEVICES=$GPU python hyper_render.py \
      --model_path output/${DATASET_NAME}_${EXPERIMENT} \
      --skip_train \
      --iteration $MAX_ITERS \
      --method_dim_reduction $METHOD_DIM_REDUCTION \
      --downsample $DOWNSAMPLE \
      --num_frames $NUM_FRAMES
fi

# avaluar amb python -i avaluacio.py
# visualitzar output/.../point_cloud/iteration_XXX/point_cloud.ply amb https://superspl.at/editor