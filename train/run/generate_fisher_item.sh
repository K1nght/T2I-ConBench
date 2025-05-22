#!/bin/bash

# Paths to models and output directory
export PATH_TO_DATA="your/path/to/data"
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"

# Define arrays for instance directories and prompts
INSTANCE_DIRS=(
    "/opt/data/private/hzhcode/T2I-ConBench-data/item/dog"
    "/opt/data/private/hzhcode/T2I-ConBench-data/item/dog3"
    "/opt/data/private/hzhcode/T2I-ConBench-data/item/cat2"
    "/opt/data/private/hzhcode/T2I-ConBench-data/item/shiny_sneaker"
)

INSTANCE_PROMPTS=(
    "A photo of p0h1 dog."
    "A photo of k5f2 dog."
    "A photo of s5g3 cat."
    "A photo of b9l1 sneaker."
)

OUTPUT_NAMES=(
    "p0h1"
    "k5f2"
    "s5g3"
    "b9l1"
)

# Loop through all instances
for i in "${!INSTANCE_DIRS[@]}"; do
    export OUTPUT_DIR="${PATH_TO_DATA}/train_results/ewc_fisher/${OUTPUT_NAMES[$i]}"
    
    deepspeed --num_gpus=2 scripts/generate_fisher_matrix_item.py \
        --deepspeed ds_config/item.json \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --instance_data_dir="${INSTANCE_DIRS[$i]}" \
        --instance_prompt="${INSTANCE_PROMPTS[$i]}" \
        --output_dir=$OUTPUT_DIR \
        --fisher_samples=100 \
        --fisher_batch_size=2 \
        --resolution=512 \
        --pre_compute_text_embeddings \
        --mixed_precision="fp16"
done