#!/bin/bash

# Paths to models and output directory
export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export INSTANCE_DIR1="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog"
export INSTANCE_DIR2="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog3"
export INSTANCE_DIR3="/opt/data/private/hzhcode/T2I-ConBench-data/item/cat2"
export INSTANCE_DIR4="/opt/data/private/hzhcode/T2I-ConBench-data/item/shiny_sneaker"
export INSTANCE_PROMPT1="A photo of p0h1 dog."
export INSTANCE_PROMPT2="A photo of k5f2 dog."
export INSTANCE_PROMPT3="A photo of s5g3 cat."
export INSTANCE_PROMPT4="A photo of b9l1 sneaker."

export CLASS_DIR1="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog_prior_images"
export CLASS_DIR2="/opt/data/private/hzhcode/T2I-ConBench-data/item/cat_prior_images"
export CLASS_DIR3="/opt/data/private/hzhcode/T2I-ConBench-data/item/sneaker_prior_images"


export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/ewc_fisher/p0h1"
deepspeed --num_gpus=2 scripts/generate_fisher_matrix_item.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/item.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR1 \
  --instance_prompt="$INSTANCE_PROMPT1" \
  --output_dir=$OUTPUT_DIR \
  --fisher_samples=100 \
  --fisher_batch_size=2 \
  --resolution=512 \
  --pre_compute_text_embeddings \
  --mixed_precision="fp16"

export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/ewc_fisher/k5f2"
deepspeed --num_gpus=2 scripts/generate_fisher_matrix_item.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/item.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR2 \
  --instance_prompt="$INSTANCE_PROMPT2" \
  --output_dir=$OUTPUT_DIR \
  --fisher_samples=100 \
  --fisher_batch_size=2 \
  --resolution=512 \
  --pre_compute_text_embeddings \
  --mixed_precision="fp16"

export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/ewc_fisher/s5g3"
deepspeed --num_gpus=2 scripts/generate_fisher_matrix_item.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/item.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR3 \
  --instance_prompt="$INSTANCE_PROMPT3" \
  --output_dir=$OUTPUT_DIR \
  --fisher_samples=100 \
  --fisher_batch_size=2 \
  --resolution=512 \
  --pre_compute_text_embeddings \
  --mixed_precision="fp16"

export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/ewc_fisher/b9l1"
deepspeed --num_gpus=2 scripts/generate_fisher_matrix_item.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/item.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR4 \
  --instance_prompt="$INSTANCE_PROMPT4" \
  --output_dir=$OUTPUT_DIR \
  --fisher_samples=100 \
  --fisher_batch_size=2 \
  --resolution=512 \
  --pre_compute_text_embeddings \
  --mixed_precision="fp16"