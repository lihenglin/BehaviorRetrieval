#!/bin/bash

#VAE Training example
python train_embedder.py \
  --config configs/embedder_VAE_image.json \
  --name Embedder_VAE_square_image \
  --dataset datasets/square_machine_policy/square_400_paired.hdf5 & \

python train_embedder.py \
  --config configs/embedder_VAE_lowdim.json \
  --name Embedder_VAE_square_lowdim \
  --dataset datasets/square_machine_policy/square_400_paired.hdf5 & \
  
wait
