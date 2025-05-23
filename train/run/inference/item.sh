CUDA="0"
export PATH_TO_DATA="your/path/to/data"
export MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
export PROMPTS_FILE1="${PATH_TO_DATA}/test_prompts/item/p0h1_dog_dog.txt"
export PROMPTS_FILE2="${PATH_TO_DATA}/test_prompts/item/k5f2_dog_dog3.txt"
export PROMPTS_FILE3="${PATH_TO_DATA}/test_prompts/item/s5g3_cat_cat2.txt"
export PROMPTS_FILE4="${PATH_TO_DATA}/test_prompts/item/b9l1_sneaker_shiny_sneaker.txt"
DIR="${PATH_TO_DATA}/inference_results"

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/seqft/items/run"
export OUTPUT_DIR="${DIR}/item/seqft/items"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf_seq.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_files $PROMPTS_FILE1 $PROMPTS_FILE2 $PROMPTS_FILE3 $PROMPTS_FILE4 \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=2 \
  --seed=42 \
  --fp16  

for file in "p0h1_dog_dog" "k5f2_dog_dog3" "s5g3_cat_cat2" "b9l1_sneaker_shiny_sneaker"; do

export PROMPTS_FILE="${PATH_TO_DATA}/test_prompts/item/${file}.txt"
DIR="${PATH_TO_DATA}/inference_results"

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/seqft/items-nature-body/run/transformer-96000"
export OUTPUT_DIR="${DIR}/item/seqft/items-nature-body/3_${file}"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=2 \
  --seed=42 \
  --fp16 

export TRANSFORMER_PATH="${PATH_TO_DATA}/train_results/seqft/nature-body-items/run/3/transformer"
export OUTPUT_DIR="${DIR}/item/seqft/nature-body-items/3_${file}"

CUDA_VISIBLE_DEVICES=${CUDA} python scripts/inference_hf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_PATH \
  --prompts_file=$PROMPTS_FILE \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=2 \
  --seed=42 \
  --fp16

done 

