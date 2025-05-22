#!/bin/bash

# nature-body-items
export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqlora/nature-body/run/transformer-320"
export INSTANCE_DIR1="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog"
export INSTANCE_DIR2="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog3"
export INSTANCE_DIR3="/opt/data/private/hzhcode/T2I-ConBench-data/item/cat2"
export INSTANCE_DIR4="/opt/data/private/hzhcode/T2I-ConBench-data/item/shiny_sneaker"
export CLASS_DIR1="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog_prior_images"
export CLASS_DIR2="/opt/data/private/hzhcode/T2I-ConBench-data/item/cat_prior_images"
export CLASS_DIR3="/opt/data/private/hzhcode/T2I-ConBench-data/item/sneaker_prior_images"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqlora/nature-body-items"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/item/train_pixart_lora.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/item.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --load_transformer_path=$TRANSFORMER_PATH \
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
  --lora_type "seqlora" \
  --lora_rank 16 \
  --lora_target_modules "all"

# items-nature-body
export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqlora/items/run/3/transformer"
export DATA_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/domain/data_info"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqlora/items-nature"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_lora.py \
  --deepspeed /opt/data/private/hzhcode/T2I-ConBench/train/ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --load_transformer_path=$TRANSFORMER_PATH \
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
  --lora_type "seqlora" \
  --lora_rank 16 \
  --lora_target_modules "all"

sleep 5

export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqlora/items-nature/run/transformer-320"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqlora/items-nature-body"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_lora.py \
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
  --lora_type "seqlora" \
  --lora_rank 16 \
  --lora_target_modules "all"
