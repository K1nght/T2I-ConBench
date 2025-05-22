import argparse
from datasets import load_dataset
import os
import json
import shutil
import random
from PIL import Image
import io
from tqdm import tqdm

def main(save_dir):
    # Step 1: Load and save images from the dataset to specified folders
    print("Starting to load images from dataset...")
    dataset = load_dataset("google/dreambooth")

    # Specify folders to extract
    target_folders = ['dog', 'dog3', 'cat2', 'shiny_sneaker']

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through dataset and save images from specified folders
    for item in target_folders:
        dataset = load_dataset("google/dreambooth", name=item)
        item_save_dir = os.path.join(save_dir, "item", item)
        os.makedirs(item_save_dir, exist_ok=True)
        for split in dataset.keys():
            for idx, item in tqdm(enumerate(dataset[split]), desc=f'Processing {split}'):
                # Get image filename
                filename = f"{idx:08d}.png"
                # Get image data
                image = item['image']
                # Build save path
                save_path = os.path.join(item_save_dir, filename)
                # Ensure target directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save image
                image.save(save_path)

    print("Dataset images loading completed!")

    # Step 2: Process saved images and create dreambooth and replay data
    print("Starting to process images and create dreambooth data...")

    # Source and target folders
    prompts = {
        'dog': 'A photo of p0h1 dog.',
        'dog3': 'A photo of k5f2 dog.',
        'cat2': 'A photo of s5g3 cat.',
        'shiny_sneaker': 'A photo of b9l1 sneaker.'
    }

    # Create target directory
    dreambooth_dir = os.path.join(save_dir, "domain", "dreambooth", "Img")
    os.makedirs(dreambooth_dir, exist_ok=True)

    # Process all images and create dreambooth.json
    dreambooth_data = []
    dreambooth_replay_data = []
    global_idx = 0  # Use global idx counter

    for item in tqdm(target_folders, desc='Processing folders'):
        source_path = os.path.join(save_dir, "item", item)
        if not os.path.exists(source_path):
            continue
        
        # Get all images in the folder
        images = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly select one image for replay
        replay_image = random.choice(images)
        
        # Copy all images to dreambooth folder
        for img in images:
            new_name = f"{global_idx:08d}.png"
            shutil.copy2(
                os.path.join(source_path, img),
                os.path.join(dreambooth_dir, new_name)
            )
            # Add to dreambooth data list
            dreambooth_data.append({
                "path": os.path.join(dreambooth_dir, new_name),
                "prompt": prompts[item],
                "idx": global_idx
            })
            
            # If it's the selected replay image, add to replay data list
            if img == replay_image:
                dreambooth_replay_data.append({
                    "path": os.path.join(dreambooth_dir, new_name),
                    "prompt": prompts[item],
                    "idx": global_idx
                })
            
            global_idx += 1

    # Save dreambooth.json
    data_info_save_dir = os.path.join(save_dir, "domain", "data_info")
    os.makedirs(data_info_save_dir, exist_ok=True)
    with open(os.path.join(data_info_save_dir, 'dreambooth.json'), 'w') as f:
        json.dump(dreambooth_data, f, indent=2)

    # Save dreambooth_replay.json
    with open(os.path.join(data_info_save_dir, 'dreambooth_replay.json'), 'w') as f:
        json.dump(dreambooth_replay_data, f, indent=2)

    print("Processing completed!")
    print(f"All images have been saved to {dreambooth_dir} folder")
    print("Created dreambooth.json and dreambooth_replay.json files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DreamBooth dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save the processed data')
    args = parser.parse_args()
    
    main(args.save_dir)

