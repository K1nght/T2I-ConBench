#!/bin/bash
CUDA="0"

save_to="/opt/data/private/hzhcode/T2I-ConBench-data/inference_results/fid/results.txt"

for method in seqft; do 
for data in nature nature_body nature_body_items items_nature_body; do 

images_folder="/opt/data/private/hzhcode/T2I-ConBench-data/inference_results/fid/$method/$data"

CUDA_VISIBLE_DEVICES=${CUDA} python eval_scripts/fid.py --generations_path ${images_folder} --output_file ${save_to}
done 
done 
