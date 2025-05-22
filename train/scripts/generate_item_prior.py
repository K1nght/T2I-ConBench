import os
import time
import argparse
import torch
from datetime import datetime
from tqdm import tqdm

from diffusers import PixArtAlphaPipeline, Transformer2DModel


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
        help="Path to transformer model"
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="Prompt for image generation"
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
        "--num_prior",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
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
        help="Whether to use half-precision inference"
    )
    return parser.parse_args()

# Main program
def main():
    args = parse_args()
    
    # Set device and weight type
    device = args.device
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    transformer = Transformer2DModel.from_pretrained(args.transformer_path, torch_dtype=weight_dtype)
    transformer.to(device)

    # Load pipeline
    pipeline = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        transformer=transformer, 
        torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(device)

    # Check and create output directory
    if not hasattr(args, 'output_dir'):
        args.output_dir = 'outputs'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving images to directory: {args.output_dir}")

    # Use single prompt
    prompt = args.validation_prompt
    print(f"Using prompt: '{prompt}'")

    # Run inference
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    images = []
    
    # Generate num_prior images for single prompt
    for j in tqdm(range(args.num_prior), desc="Generating images", total=args.num_prior):
        # Record start time
        start_time = time.time()
        
        # Generate image
        pipeline_args = {"prompt": prompt}
        image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
        images.append(image)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        print(f"Image {j+1}/{args.num_prior} generated, prompt: '{prompt}', time taken: {generation_time:.2f} seconds")
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use first 50 characters of prompt as filename (to avoid long filenames)
        filename = f"{j}_{prompt[:50]}_{timestamp}.png"
        image_path = os.path.join(args.output_dir, filename)
        image.save(image_path)
        print(f"Image saved to: {image_path}")

if __name__ == "__main__":
    main()