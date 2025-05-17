#!/bin/bash

# Paths to models and output directory
export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512/transformer"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/item/dog_prior_images"

# Prompt to generate images for
export PROMPT="a photo of a dog"

# Run the generation script
python scripts/generate_item_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --validation_prompt="$PROMPT" \
  --output_dir=$OUTPUT_DIR \
  --num_prior=50 \
  --seed=42 \
  --fp16

# Paths to models and output directory
export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512/transformer"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/item/cat_prior_images"

# Prompt to generate images for
export PROMPT="a photo of a cat"

# Run the generation script
python scripts/generate_item_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --validation_prompt="$PROMPT" \
  --output_dir=$OUTPUT_DIR \
  --num_prior=50 \
  --seed=42 \
  --fp16

# Paths to models and output directory
export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512/transformer"
export OUTPUT_DIR="/opt/data/private/hzhcode/T2I-ConBench-data/item/sneaker_prior_images"

# Prompt to generate images for
export PROMPT="a photo of a sneaker"

# Run the generation script
python scripts/generate_item_prior.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --validation_prompt="$PROMPT" \
  --output_dir=$OUTPUT_DIR \
  --num_prior=50 \
  --seed=42 \
  --fp16