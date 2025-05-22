#!/bin/bash

# Base paths
export PATH_TO_DATA="your/path/to/data"
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
export TRANSFORMER_PATH="${PATH_TO_DATA}/models/PixArt-alpha/PixArt-XL-2-512x512/transformer"
export BASE_OUTPUT_DIR="${PATH_TO_DATA}/item"

# Define arrays for categories and prompts
categories=("dog" "cat" "sneaker")
prompts=("a photo of a dog" "a photo of a cat" "a photo of a sneaker")

# Loop through categories and generate images
for i in "${!categories[@]}"; do
    export OUTPUT_DIR="${BASE_OUTPUT_DIR}/${categories[$i]}_prior_images"
    export PROMPT="${prompts[$i]}"
    
    echo "Generating images for ${categories[$i]}..."
    python scripts/generate_item_prior.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --transformer_path=$TRANSFORMER_PATH \
        --validation_prompt="$PROMPT" \
        --output_dir=$OUTPUT_DIR \
        --num_prior=50 \
        --seed=42 \
        --fp16
done