#!/bin/bash

export MODEL_NAME="/data/hzh_data/huggingface/PixArt-XL-2-512x512"
export DATA_DIR="/data/hzh_data/benchmark/stage3/data_info"
export OUTPUT_DIR="/data/hzh_data/benchmark/ewc_fisher/nature"

# Run with DeepSpeed
deepspeed --num_gpus=8 \
  scripts/generate_fisher_matrix_domain.py \
  --deepspeed /home/wx1349603/PixArt-alpha-hzh/ds_config/ft_ds_config.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --data_info="natural_train_v3.json" \
  --resolution=512 \
  --fisher_samples=1000 \
  --fisher_batch_size=4 \
  --pre_compute_text_embeddings \
  --output_path=$OUTPUT_DIR/fisher_matrix.pt \
  --mixed_precision="fp16"

export MODEL_NAME="/data/hzh_data/huggingface/PixArt-XL-2-512x512"
export DATA_DIR="/data/hzh_data/benchmark/stage3/data_info"
export OUTPUT_DIR="/data/hzh_data/benchmark/ewc_fisher/body"

# Run with DeepSpeed
deepspeed --num_gpus=8 \
  scripts/generate_fisher_matrix_domain.py \
  --deepspeed /home/wx1349603/PixArt-alpha-hzh/ds_config/ft_ds_config.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --data_info="body_train_v4.json" \
  --resolution=512 \
  --fisher_samples=1000 \
  --fisher_batch_size=4 \
  --pre_compute_text_embeddings \
  --output_path=$OUTPUT_DIR/fisher_matrix.pt \
  --mixed_precision="fp16"

export MODEL_NAME="/data/hzh_data/huggingface/PixArt-XL-2-512x512"
export DATA_DIR="/data/hzh_data/benchmark/stage3/data_info"
export OUTPUT_DIR="/data/hzh_data/benchmark/ewc_fisher/cross_old"

# Run with DeepSpeed
deepspeed --num_gpus=8 \
  scripts/generate_fisher_matrix_domain.py \
  --deepspeed /home/wx1349603/PixArt-alpha-hzh/ds_config/ft_ds_config.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --data_info="cross_train_v3.json" \
  --resolution=512 \
  --fisher_samples=10 \
  --fisher_batch_size=4 \
  --pre_compute_text_embeddings \
  --output_path=$OUTPUT_DIR/fisher_matrix.pt \
  --mixed_precision="fp16"