#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

PATH_TO_DATA="your/path/to/data"

METHOD_ROOT="${PATH_TO_DATA}/inference_results/item/seqft"
OUTPUT_DIR="${PATH_TO_DATA}/inference_results/item/seqft_results"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
REF_IMAGE_DIR="${PATH_TO_DATA}/item"

for data in items nature-body-items items-nature-body; do
  IMAGE_DIR="$METHOD_ROOT/$data"
  if [[ -d "$IMAGE_DIR" ]]; then
    OUTDIR="$OUTPUT_DIR/$data"
    mkdir -p "$OUTDIR"
    OUTPUT_JSON="$OUTDIR/results.json"
    OUTPUT_TXT="$OUTDIR/results.txt"
    echo "Running item.py for $IMAGE_DIR ..."
    python eval_scipts/item.py \
      --image_dir="$IMAGE_DIR" \
      --output_file="$OUTPUT_JSON" \
      --model_name="$MODEL_PATH" \
      --txt_output="$OUTPUT_TXT" \
      --ref_image_dir="$REF_IMAGE_DIR"
  fi
done


