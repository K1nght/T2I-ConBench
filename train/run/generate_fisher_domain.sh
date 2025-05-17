#!/bin/bash

export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export DATA_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/domain/data_info"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/ewc_fisher/nature"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  scripts/generate_fisher_matrix_domain.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --data_info "nature.json" \
  --resolution=512 \
  --fisher_samples=20 \
  --fisher_batch_size=4 \
  --pre_compute_text_embeddings \
  --output_path=$OUTPUT_DIR/fisher_matrix.pt \
  --mixed_precision="fp16"

export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/ewc_fisher/body"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  scripts/generate_fisher_matrix_domain.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --data_info "body.json" \
  --resolution=512 \
  --fisher_samples=20 \
  --fisher_batch_size=4 \
  --pre_compute_text_embeddings \
  --output_path=$OUTPUT_DIR/fisher_matrix.pt \
  --mixed_precision="fp16"

export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/ewc_fisher/cross"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  scripts/generate_fisher_matrix_domain.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --data_info "cross.json" \
  --resolution=512 \
  --fisher_samples=20 \
  --fisher_batch_size=4 \
  --pre_compute_text_embeddings \
  --output_path=$OUTPUT_DIR/fisher_matrix.pt \
  --mixed_precision="fp16"