#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH_TO_DATA="your/path/to/data"

export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export OUTPUT_ROOT="${PATH_TO_DATA}/inference_results/cross/seqft"
export REF_IMAGE_DIR="${PATH_TO_DATA}/item"
export REF_NATURE_DIR="domain_ref_img"
export OUTPUT_DIR_ROOT="${PATH_TO_DATA}/inference_results/cross_all_results/seqft"

declare -A SCRIPT_MAP
SCRIPT_MAP[domain+domain]="eval_scipts/cross/domain_domain.py"
SCRIPT_MAP[item+item]="eval_scipts/cross/item_item.py"
SCRIPT_MAP[item+nature]="eval_scipts/cross/item_nature.py"
SCRIPT_MAP[item+body]="eval_scipts/cross/item_body.py"

declare -A QA_MAP
QA_MAP[domain+domain]="qa_text/domain_domain_qa.txt"
QA_MAP[item+item]="qa_text/item_item_qa.txt"
QA_MAP[item+nature]="qa_text/item_nature_qa.txt"
QA_MAP[item+body]="qa_text/item_body_qa.txt"
# For item+nature, also set ref_nature_dir

for DIR in $OUTPUT_ROOT/*/; do
  SUBDIR_NAME=$(basename "$DIR")
  for SUB in domain+domain item+item item+nature item+body; do
    IMAGE_FILE="$DIR/$SUB"
    if [[ -d "$IMAGE_FILE" ]]; then
      SCRIPT=${SCRIPT_MAP[$SUB]}
      QA_FILE=${QA_MAP[$SUB]}
      OUTDIR="$OUTPUT_DIR_ROOT/$SUBDIR_NAME/$SUB"
      mkdir -p "$OUTDIR"
      OUTPUT_JSON="$OUTDIR/results.json"
      OUTPUT_TXT="$OUTDIR/results.txt"
      echo "Testing $IMAGE_FILE with $SCRIPT ..."
      if [[ "$SUB" == "item+nature" ]]; then
        python "$SCRIPT" \
          --image_dir="$IMAGE_FILE" \
          --qa_file="$QA_FILE" \
          --model_name="$MODEL_PATH" \
          --output_file="$OUTPUT_JSON" \
          --txt_output="$OUTPUT_TXT" \
          --ref_image_dir="$REF_IMAGE_DIR" \
          --ref_nature_dir="$REF_NATURE_DIR"
      elif [[ "$SUB" == "domain+domain" ]]; then
        python "$SCRIPT" \
          --image_dir="$IMAGE_FILE" \
          --qa_file="$QA_FILE" \
          --model_name="$MODEL_PATH" \
          --output_file="$OUTPUT_JSON" \
          --txt_output="$OUTPUT_TXT" \
          --ref_nature_dir="$REF_NATURE_DIR"
      else
        python "$SCRIPT" \
          --image_dir="$IMAGE_FILE" \
          --qa_file="$QA_FILE" \
          --model_name="$MODEL_PATH" \
          --output_file="$OUTPUT_JSON" \
          --txt_output="$OUTPUT_TXT" \
          --ref_image_dir="$REF_IMAGE_DIR"
      fi
    fi
  done
done