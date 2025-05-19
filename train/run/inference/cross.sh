#!/bin/bash
CUDA=0

export MODEL_NAME="/opt/data/private/hzhcode/huggingface/models/PixArt-alpha/PixArt-XL-2-512x512"
DIR="/opt/data/private/hzhcode/T2I-ConBench-data/inference_results"

for file in "item+item"; do
export PROMPTS_FILE="/opt/data/private/hzhcode/T2I-ConBench-data/test_prompts/cross/${file}.txt"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqft/items/run/3/transformer"
export OUTPUT_DIR="${DIR}/cross/seqft/items/${file}"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --validation_length=10 \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 
done 

for file in "domain+domain"; do
export PROMPTS_FILE="/opt/data/private/hzhcode/T2I-ConBench-data/test_prompts/cross/${file}.txt"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqft/nature_body/run/transformer-320"
export OUTPUT_DIR="${DIR}/cross/seqft/nature_body/${file}"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --validation_length=10 \
  --num_validation_images=1 \
  --seed=42 \
  --fp16 
done 

for file in "item+item" "item+body" "item+nature" "domain+domain"; do 
export PROMPTS_FILE="/opt/data/private/hzhcode/T2I-ConBench-data/test_prompts/cross/${file}.txt"
export TRANSFORMER_PATH="/opt/data/private/hzhcode/T2I-ConBench-data/train_results/seqft/items-nature-body/run/transformer-320"
export OUTPUT_DIR="${DIR}/cross/seqft/items-nature-body/${file}"

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
export OUTPUT_DIR="${DIR}/cross/seqft/nature-body-items/${file}"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --validation_length=10 \
  --num_validation_images=1 \
  --seed=42 \
  --fp16

done 