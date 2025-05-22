#!/bin/bash
export PATH_TO_DATA="your/path/to/data"
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
export DATA_DIR="${PATH_TO_DATA}/domain/data_info"
export OUTPUT_DIR="${PATH_TO_DATA}/train_results/replay/nature"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_ft.py \
  --deepspeed ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --data_info "nature.json" "coco_500_replay.json" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=32 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=96000 \
  --checkpointing_steps=32000 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16"

sleep 5

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/replay/nature/run/transformer-320"
export OUTPUT_DIR="${PATH_TO_DATA}/train_results/replay/nature-body"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_ft.py \
  --deepspeed ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --load_transformer_path=$TRANSFORMER_PATH \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --data_info "body.json" "cross.json" "coco_500_replay.json" "nature_replay.json" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=32 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=96000 \
  --checkpointing_steps=32000 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" 
 