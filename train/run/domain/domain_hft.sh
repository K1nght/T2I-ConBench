#!/bin/bash

export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export DATA_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/domain/data_info"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/hft/nature"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_hft.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --data_info="nature.json" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=32 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=320 \
  --checkpointing_steps=320 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" \
  --hft_separate_layer \
  --hft_freeze_ratio=0.5

sleep 5

export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/hft/nature/run/transformer-320"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/hft/nature_body"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_hft.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --load_transformer_path=$TRANSFORMER_PATH \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --data_info "body.json" "cross.json" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=32 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=320 \
  --checkpointing_steps=320 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" \
  --hft_separate_layer \
  --hft_freeze_ratio=0.5
 