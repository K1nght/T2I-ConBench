CUDA="0"
export PATH_TO_DATA="your/path/to/data"
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
export PROMPTS_FILE="${PATH_TO_DATA}/test_prompts/fid/COCO_3w.txt"
export DIR="${PATH_TO_DATA}/inference_results"

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/seqft/items/run/3/transformer"
export OUTPUT_DIR="${DIR}/fid/seqft/items"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/seqft/nature_body/run/transformer-96000"
export OUTPUT_DIR="${DIR}/fid/seqft/nature_body"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/seqft/items-nature-body/run/transformer-96000"
export OUTPUT_DIR="${DIR}/fid/seqft/items-nature-body"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/seqft/nature-body-items/run/3/transformer"
export OUTPUT_DIR="${DIR}/fid/seqft/nature-body-items"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=1 \
  --seed=42 \
  --fp16