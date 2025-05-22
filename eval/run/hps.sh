#!/bin/bash
CUDA="0"
PATH_TO_DATA="your/path/to/data"

save_to="${PATH_TO_DATA}/inference_results/domain/hps.txt"

for prompt in "nature" "body";do
prompt_path="data/test_prompts/domain/${prompt}.txt"

for method in seqft; do 
for data in nature nature_body nature_body_items items_nature_body; do 

dir="${PATH_TO_DATA}/inference_results/domain/$method/$data"
images_folder="${dir}/${prompt}"
detailed_save_to="${dir}/detailed_${prompt}.txt"

CUDA_VISIBLE_DEVICES=${CUDA} python eval_scripts/hps.py --images_folder ${images_folder} --prompt_path ${prompt_path} --save_to ${save_to} --detailed_save_to ${detailed_save_to}
done 
done 

done