#!/bin/bash
export PATH_TO_DATA="your/path/to/data"
# nature-body-items
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/clora/nature-body/run/transformer-96000"
export INSTANCE_DIR1="${PATH_TO_DATA}/item/dog"
export INSTANCE_DIR2="${PATH_TO_DATA}/item/dog3"
export INSTANCE_DIR3="${PATH_TO_DATA}/item/cat2"
export INSTANCE_DIR4="${PATH_TO_DATA}/item/shiny_sneaker"
export CLASS_DIR1="${PATH_TO_DATA}/item/dog_prior_images"
export CLASS_DIR2="${PATH_TO_DATA}/item/cat_prior_images"
export CLASS_DIR3="${PATH_TO_DATA}/item/sneaker_prior_images"
export OUTPUT_DIR="${PATH_TO_DATA}/train_results/clora/nature-body-items"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/item/train_pixart_lora.py \
  --deepspeed ds_config/item.json \
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
  --max_train_steps=500 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" \
  --lora_type "clora" \
  --lora_rank 16 \
  --lora_target_modules "all" \
  --lamda 1e6

# items-nature-body
export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/clora/items/run/3/transformer"
export DATA_DIR="${PATH_TO_DATA}/domain/data_info"
export OUTPUT_DIR="${PATH_TO_DATA}/train_results/clora/items-nature"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_lora.py \
  --deepspeed ds_config/domain.json \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --load_transformer_path=$TRANSFORMER_PATH \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --data_info="nature.json" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=96000 \
  --checkpointing_steps=32000 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" \
  --lora_type "clora" \
  --lora_rank 16 \
  --lora_target_modules "all" \
  --lamda 1e6

sleep 5

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/clora/items-nature/run/transformer-96000"
export OUTPUT_DIR="${PATH_TO_DATA}/train_results/clora/items-nature-body"

# Run with DeepSpeed
deepspeed --num_gpus=2 \
  train_scripts/domain/train_pixart_lora.py \
  --deepspeed ds_config/domain.json \
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
  --max_train_steps=96000 \
  --checkpointing_steps=32000 \
  --pre_compute_text_embeddings \
  --seed="0" \
  --mixed_precision="fp16" \
  --lora_type "clora" \
  --lora_rank 16 \
  --lora_target_modules "all" \
  --lamda 1e6
