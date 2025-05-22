#!/bin/bash
export PATH_TO_DATA="your/path/to/data"
# items-nature-body
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
export DATA_DIR="${PATH_TO_DATA}/domain/data_info"
export OUTPUT_DIR="${PATH_TO_DATA}/train_results/joint/nature-body-items"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_ft.py \
  --deepspeed ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --data_info "coco_500_replay.json" "nature.json" "body.json" "cross.json" "dreambooth.json" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=32 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=192000 \
  --checkpointing_steps=96000 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16"

