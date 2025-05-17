export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"

# 定义所有需要处理的数据集
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

# 循环处理每个数据集
for dataset in "${datasets[@]}"; do
    dir="/opt/data/private/hzhcode/T2I-ConBench-data/domain/${dataset}"
    CUDA_VISIBLE_DEVICES=0,1 python scripts/extract_t5_features.py \
        --json_path "/opt/data/private/hzhcode/T2I-ConBench-data/domain/data_info/${dataset}.json" \
        --pretrained_model_name_or_path $MODEL_NAME \
        --t5_save_root "$dir/caption_feature"
done