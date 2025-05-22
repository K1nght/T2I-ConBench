from datasets import load_dataset
import os
import json
from PIL import Image
import io
from tqdm import tqdm
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Process T2I-ConBench dataset')
parser.add_argument('--save_dir', type=str, required=True,
                    help='Base directory to save processed data')
args = parser.parse_args()

dataset = load_dataset(
    "T2I-ConBench/T2I-ConBench", 
    revision="refs/convert/parquet",
)

save_dir = [
    ("body", 2358),
    ("body_replay", 235),
    ("coco_500_replay", 500),
    ("cross", 1821),
    ("cross_replay", 182),
    ("nature", 2512),
    ("nature_replay", 251),
]

# Create base save directory
base_save_dir = args.save_dir
os.makedirs(base_save_dir, exist_ok=True)
# Create data_info directory
data_info_dir = os.path.join(base_save_dir, "data_info")
os.makedirs(data_info_dir, exist_ok=True)

# Calculate total number of samples
total_samples = sum(num for _, num in save_dir)
print(f"Total samples to process: {total_samples}")

# Create total progress bar
pbar = tqdm(total=total_samples, desc="Total Progress")

# Process each domain's data
current_idx = 0

for domain, num_samples in save_dir:
    print(f"\nProcessing domain: {domain}")
    
    # Create domain directory and image subdirectory
    domain_dir = os.path.join(base_save_dir, domain)
    img_dir = os.path.join(domain_dir, "Img")
    os.makedirs(img_dir, exist_ok=True)
    
    # Prepare list to store prompts
    prompts_list = []
    
    # Create domain-level progress bar
    domain_pbar = tqdm(total=num_samples, desc=f"{domain}", leave=False)
    
    # Save specified number of samples
    for i in range(num_samples):
        sample = dataset["train"][current_idx + i]
        
        # Save image
        image_data = sample["image"]
        if isinstance(image_data, dict):  # If image is in dictionary format (contains bytes)
            image = Image.open(io.BytesIO(image_data["bytes"]))
        elif isinstance(image_data, (bytes, bytearray)):  # If it's raw byte data
            image = Image.open(io.BytesIO(image_data))
        else:  # If it's already a PIL Image object
            image = image_data
            
        image_path = os.path.join(img_dir, f"{i:08d}.png")
        image.save(image_path)
        
        # Store prompt information
        prompt_info = {
            "path": os.path.join(domain_dir, "Img", f"{i:08d}.png"),
            "prompt": sample["prompt"],
            "idx": i
        }
        prompts_list.append(prompt_info)
        
        # Update progress bars
        pbar.update(1)
        domain_pbar.update(1)
    
    # Close domain progress bar
    domain_pbar.close()
    
    # Save prompts to json file
    json_path = os.path.join(data_info_dir, f"{domain}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_list, f, ensure_ascii=False, indent=2)
    
    # Update current index
    current_idx += num_samples
    print(f"Completed {domain}: {num_samples} samples")

# Close total progress bar
pbar.close()

print("\nAll processing completed!")
print(f"Total processed samples: {total_samples}")
