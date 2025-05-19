CUDA="0"

export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
export PROMPTS_FILE="/opt/data/private/hzhcode/T2I-ConBench-data/test_prompts/compbench/complex_val.txt"
DIR="/opt/data/private/hzhcode/T2I-ConBench-data/inference_results"


export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqft/items/run/3/transformer"
export OUTPUT_DIR="${DIR}/comp/seqft/items"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --validation_length=10 \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 

export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqft/nature_body/run/transformer-320"
export OUTPUT_DIR="${DIR}/comp/seqft/nature_body"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --validation_length=10 \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 

export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqft/items-nature-body/run/transformer-320"
export OUTPUT_DIR="${DIR}/comp/seqft/items-nature-body"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --validation_length=10 \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 

export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqft/nature-body-items/run/3/transformer"
export OUTPUT_DIR="${DIR}/comp/seqft/nature-body-items"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --validation_length=10 \
  --num_validation_images=1 \
  --seed=42 \
  --fp16