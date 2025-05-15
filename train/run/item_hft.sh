#!/bin/bash

export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export INSTANCE_DIR1="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog"
export INSTANCE_DIR2="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog3"
export INSTANCE_DIR3="/opt/data/private/hzhcode/T2I-ConBench-data/item/cat2"
export INSTANCE_DIR4="/opt/data/private/hzhcode/T2I-ConBench-data/item/shiny_sneaker"
export CLASS_DIR1="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog_prior_images"
export CLASS_DIR2="/opt/data/private/hzhcode/T2I-ConBench-data/item/cat_prior_images"
export CLASS_DIR3="/opt/data/private/hzhcode/T2I-ConBench-data/item/sneaker_prior_images"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/hft/items"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/item/train_pixart_hft.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/item.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dirs="${INSTANCE_DIR1},${INSTANCE_DIR2},${INSTANCE_DIR3},${INSTANCE_DIR4}" \
  --output_dir=$OUTPUT_DIR \
  --instance_prompts="A photo of p0h1 dog.,A photo of k5f2 dog.,A photo of s5g3 cat.,A photo of b9l1 sneaker." \
  --with_prior_preservation \
  --class_data_dir="$CLASS_DIR1,$CLASS_DIR1,$CLASS_DIR2,$CLASS_DIR3" \
  --class_prompt="A photo of a dog.,A photo of a dog.,A photo of a cat.,A photo of a sneaker." \
  --num_class_images=500 \
  --prior_loss_weight=0.02 \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" \
  --hft_separate_layer \
  --hft_freeze_ratio=0.5