import os
import time
import argparse
import torch
from datetime import datetime
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
from peft import PeftModel, PeftConfig
from peft.tuners.lora import LoraLayer

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
        "--prompts_file",
        type=str,
        default=None,
        help="Path to text file containing multiple prompts, one per line"
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
        "--validation_length",
        type=int,
        default=1e6,
        help="Number of prompts to use from the file"
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
        help="Input images for validation (optional)"
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

@torch.inference_mode()
def load_base_and_all_loras(args, weight_dtype):
    # use Transformer2DModel for PixArt-Alpha
    # you can change the model to other models,
    # for SD v1.4, we use UNet2DConditionModel
    # In this case, you need to change the subfolder to "unet"
    model = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)

    # Freeze the transformer parameters before adding adapters
    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)

    # if no adapter exists, return the base model
    # if args.transformer_path is None or args.transformer_path == "":
    if args.transformer_path is None:
        print("No LoRA adapters found. Loading base model.")
        return [], model

    lora_paths = {}
    for subdir in os.listdir(args.transformer_path):
        sub_path = os.path.join(args.transformer_path, subdir)
        if os.path.isdir(sub_path) and "adapter_config.json" in os.listdir(sub_path):
            lora_paths[subdir] = sub_path
    lora_adapters = sorted(lora_paths.keys())

    print(f"Loading {len(lora_adapters)} LoRA adapters from {args.transformer_path}")

    # load all adapters to the base model
    adapters = sorted(lora_paths.items())
    first_adapter, first_adapter_path = adapters[0]
    model = PeftModel.from_pretrained(model, first_adapter_path, adapter_name=first_adapter)

    for name, path in adapters[1:]:
        model.load_adapter(path, adapter_name=name)

    model = model.merge_and_unload(safe_merge=True, progressbar=True, adapter_names=lora_adapters)

    return model

# Main program
@torch.inference_mode()
def main():
    args = parse_args()

    # Set device and weight type
    device = args.device
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    transformer = load_base_and_all_loras(args, weight_dtype)
    transformer.to(device).eval()

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

    # Read prompts
    prompts = []
    if args.prompts_file is not None and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        prompts = prompts[:args.validation_length]
        print(f"Read {len(prompts)} prompts from file {args.prompts_file}")
    else:
        raise ValueError(f"File {args.prompts_file} does not exist")

    # Run inference
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    images = []

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating images"):
        # Record start time
        start_time = time.time()

        # Generate num_validation_images for each prompt
        for img_idx in range(args.num_validation_images):
            # Generate image
            pipeline_args = {"prompt": prompt}
            image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
            images.append(image)

            # Calculate generation time
            generation_time = time.time() - start_time
            print(f"Image {i+1}/{len(prompts)} generated, prompt: '{prompt}', time taken: {generation_time:.2f} seconds")

            # Save image
            # Use first 100 characters of prompt as filename (to avoid long filenames)
            filename = f"{i}_{img_idx}_{prompt[:100]}_seed{args.seed}.png"
            image_path = os.path.join(args.output_dir, filename)
            image.save(image_path)
            print(f"Image saved to: {image_path}")

if __name__ == "__main__":
    main()