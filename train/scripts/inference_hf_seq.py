import os
import time
import argparse
import torch
from datetime import datetime
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer

# Add command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Run PixArtAlpha inference")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or Hugging Face model ID"
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        required=True,
        help="Parent directory containing multiple transformer model checkpoints, subdirectories should be named 0,1,2 etc."
    )
    parser.add_argument(
        "--prompts_files",
        nargs='+',
        type=str,
        default=None,
        help="List of text file paths containing multiple prompts, one prompt per line"
    )
    parser.add_argument(
        "--validation_length",
        type=int,
        default=1e6,
        help="Number of prompts to use from prompt files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help="Input image for validation (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use half precision inference"
    )
    return parser.parse_args()

# Main program
def main():
    args = parse_args()
    
    # Set device and weight type
    device = args.device
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    # Check and create output directory
    if not hasattr(args, 'output_dir'):
        args.output_dir = 'outputs'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving images to directory: {args.output_dir}")

    # Check if transformer subdirectories exist
    transformer_dirs = []
    i = 0
    while os.path.exists(os.path.join(args.transformer_path, str(i), "transformer")):
        transformer_dirs.append(os.path.join(args.transformer_path, str(i), "transformer"))
        i += 1
    
    if not transformer_dirs:
        raise ValueError(f"No numbered subdirectories found in {args.transformer_path}")
    
    print(f"Found {len(transformer_dirs)} transformer checkpoint directories")

    # Check prompt files
    prompts_list = []
    prompts_filenames = []
    for file_path in args.prompts_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
            prompts = prompts[:args.validation_length]
            prompts_list.append(prompts)
            # Extract filename from path (without path and extension)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            prompts_filenames.append(filename)
            print(f"Read {len(prompts)} prompts from file {file_path}")
        else:
            print(f"Warning: Prompt file {file_path} does not exist")
            prompts_list.append([])
            prompts_filenames.append(f"invalid_file_{len(prompts_filenames)}")
    
    # Initialize pipeline
    pipeline = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,  # Don't load transformer yet, will load based on directory later
        torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(device)
    
    # Set random seed generator
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    
    # Process each transformer directory
    for idx, transformer_dir in enumerate(transformer_dirs):
        print(f"\nLoading transformer model from {transformer_dir}...")
        
        # Load current transformer model
        transformer = Transformer2DModel.from_pretrained(transformer_dir, torch_dtype=weight_dtype)
        transformer.to(device)
        
        # Update pipeline's transformer
        pipeline.transformer = transformer
        
        # Process each prompts file separately
        for i in range(idx + 1):
            if i < len(prompts_list) and prompts_list[i]:
                # Get current prompts
                current_prompts = prompts_list[i]
                current_filename = prompts_filenames[i]
                
                # Create output directory for specific transformer and prompts file
                output_subdir = os.path.join(args.output_dir, f"{idx}_{current_filename}")
                os.makedirs(output_subdir, exist_ok=True)
                
                print(f"Processing {len(current_prompts)} prompts from file '{current_filename}' using transformer {idx}")
                print(f"Generating {args.num_validation_images} images for each prompt")
                
                # Run inference
                for j, prompt in tqdm(enumerate(current_prompts), total=len(current_prompts), 
                                      desc=f"Transformer {idx} - {current_filename} generating images"):
                    # Create subdirectory for each prompt
                    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:100])
                    
                    # Generate num_validation_images for each prompt
                    for img_idx in range(args.num_validation_images):
                        # Record start time
                        start_time = time.time()
                        
                        # Generate image
                        pipeline_args = {"prompt": prompt}
                        image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
                        
                        # Calculate generation time
                        generation_time = time.time() - start_time
                        print(f"Prompt {j+1}/{len(current_prompts)}, Image {img_idx+1}/{args.num_validation_images} completed,"
                              f" prompt: '{prompt}', time taken: {generation_time:.2f} seconds")
                        
                        # Save image
                        # Include image index in filename
                        filename = f"{j}_{img_idx}_{safe_prompt}_seed{args.seed}.png"
                        image_path = os.path.join(output_subdir, filename)
                        image.save(image_path)
                        print(f"Image saved to: {image_path}")

if __name__ == "__main__":
    main()