#!/bin/bash
CUDA="0"
PATH_TO_DATA="your/path/to/data"

save_to="${PATH_TO_DATA}/inference_results/fid/results.txt"

for method in seqft; do 
for data in nature nature_body nature_body_items items_nature_body; do 

images_folder="${PATH_TO_DATA}/inference_results/fid/$method/$data"

CUDA_VISIBLE_DEVICES=${CUDA} python eval_scripts/fid.py --generations_path ${images_folder} --output_file ${save_to}
done 
done 
