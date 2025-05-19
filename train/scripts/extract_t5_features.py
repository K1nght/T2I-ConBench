import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import torch
import numpy as np
import json
from tqdm import tqdm
import argparse
import threading
from queue import Queue
from pathlib import Path

from transformers import T5EncoderModel, T5Tokenizer


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=120):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask):
    text_input_ids = input_ids.to(text_encoder.device)

    attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def compute_text_embeddings(tokenizer, text_encoder, prompt, max_length=120):
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=max_length)
        prompt_embeds = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
        )

    return prompt_embeds, text_inputs

def extract_caption_t5_do(q):
    while not q.empty():
        item = q.get()
        extract_caption_t5_job(item)
        q.task_done()


def extract_caption_t5_job(item):
    global mutex
    global tokenizer
    global text_encoder
    global t5_save_dir
    global count
    global total_item

    with torch.no_grad():
        save_path = os.path.join(t5_save_dir, f"{item['idx']}")
        if os.path.exists(save_path + ".npz"):
            count += 1
            return
        caption = item['prompt'].strip()
        if isinstance(caption, str):
            caption = [caption]
        try:
            mutex.acquire()
            prompt_embeds, text_inputs = compute_text_embeddings(tokenizer, text_encoder, caption, max_length=120)
            mutex.release()
            emb_dict = {
                'caption_feature': prompt_embeds.float().cpu().data.numpy(),
                'attention_mask': text_inputs.attention_mask.cpu().data.numpy(),
            }
            np.savez_compressed(save_path, **emb_dict)
            count += 1
        except Exception as e:
            print(e)
    print(f"CUDA: {os.environ['CUDA_VISIBLE_DEVICES']}, processed: {count}/{total_item}, token length: 120, saved at: {save_path}")


def extract_caption_t5():
    global tokenizer
    global text_encoder
    global t5_save_dir
    global count
    global total_item

    # global images_extension
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    t5_save_dir = args.t5_save_root
    count = 0
    os.makedirs(t5_save_dir, exist_ok=True)

    train_data_json = json.load(open(args.json_path, 'r'))
    train_data = train_data_json[args.start_index: args.end_index]
    total_item = len(train_data)

    global mutex
    mutex = threading.Lock()
    jobs = Queue()

    for item in tqdm(train_data):
        jobs.put(item)

    for _ in range(20):
        worker = threading.Thread(target=extract_caption_t5_do, args=(jobs,))
        worker.start()

    jobs.join()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=1000000, type=int)
    
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--t5_save_root', default='data/data_toy/caption_feature', type=str)
    parser.add_argument('--pretrained_model_name_or_path', default='output/pretrained_models', type=str)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare extracted caption t5 features for training
    extract_caption_t5()

