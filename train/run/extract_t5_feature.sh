export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"

# Define all datasets to be processed
datasets=(
    "body"
    "cross"
    "nature"
    "body_replay"
    "cross_replay"
    "nature_replay"
    "coco_500_replay"
    "dreambooth"
)

export PATH_TO_DATA="your/path/to/data"

# Process each dataset in a loop
for dataset in "${datasets[@]}"; do
    dir="${PATH_TO_DATA}/domain/${dataset}"
    CUDA_VISIBLE_DEVICES=0,1 python scripts/extract_t5_features.py \
        --json_path "${PATH_TO_DATA}/domain/data_info/${dataset}.json" \
        --pretrained_model_name_or_path $MODEL_NAME \
        --t5_save_root "$dir/caption_feature"
done