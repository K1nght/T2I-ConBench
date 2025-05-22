#!/bin/bash
export PATH_TO_DATA="your/path/to/data"
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
export DATA_DIR="${PATH_TO_DATA}/domain/data_info"

# Define domains and their corresponding output directories
domains=("nature" "body" "cross")
base_output_dir="${PATH_TO_DATA}/train_results/ewc_fisher"

# Loop through each domain
for domain in "${domains[@]}"; do
    export OUTPUT_DIR="${base_output_dir}/${domain}"
    
    # Run with DeepSpeed
    deepspeed --num_gpus=2 \
        scripts/generate_fisher_matrix_domain.py \
        --deepspeed ds_config/domain.json \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --data_dir=$DATA_DIR \
        --data_info "${domain}.json" \
        --resolution=512 \
        --fisher_samples=20 \
        --fisher_batch_size=4 \
        --pre_compute_text_embeddings \
        --output_path=$OUTPUT_DIR/fisher_matrix.pt \
        --mixed_precision="fp16"
done