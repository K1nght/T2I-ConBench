#!/usr/bin/env python
# Script to generate and save Fisher matrices for EWC regularization

import argparse
import logging
import os
import gc
import random
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import deepspeed as ds
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer

# Fix the logger initialization
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch


def compute_fisher_matrix(
    transformer,
    train_dataset,
    tokenizer,
    text_encoder,
    vae,
    noise_scheduler,
    device,
    num_samples,
    batch_size,
    weight_dtype,
    text_encoder_use_attention_mask=False,
    with_prior_preservation=False,
    dataloaders_kwargs=None,
):
    """
    Computes the Fisher Information Matrix for Elastic Weight Consolidation (EWC).
    The Fisher matrix approximates the importance of each parameter.
    This version is compatible with DeepSpeed ZeRO Stage 2/3.
    """
    logger.info("Computing Fisher Information Matrix for EWC...")
    
    if dataloaders_kwargs is None:
        dataloaders_kwargs = {}
    
    # Create dataloader with specified batch size for Fisher computation
    fisher_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),
        **dataloaders_kwargs
    )
    
    # Initialize parameter dictionaries - we'll compute on rank 0 and distribute
    is_main_process = ds.comm.get_rank() == 0
    
    # Create a DeepSpeed engine for Fisher computation
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer_cls = torch.optim.AdamW
    fisher_optimizer = optimizer_cls(
        params_to_optimize,
        lr=1e-4,  # Learning rate doesn't matter for Fisher computation
        betas=(0.9, 0.999),
        weight_decay=0.0,  # No weight decay for Fisher computation
        eps=1e-8,
    )
    
    # Calculate appropriate train batch size for DeepSpeed
    # To avoid "micro_batch > 0" error, ensure train_batch_size >= world_size
    world_size = ds.comm.get_world_size()
    ds_train_batch_size = max(batch_size * world_size, world_size)
    
    # Initialize DeepSpeed for Fisher computation
    fisher_model_engine, _, _, _ = ds.initialize(
        model=transformer,
        optimizer=fisher_optimizer,
        config={
            "zero_optimization": {"stage": 2}, 
            "train_batch_size": ds_train_batch_size,
            "gradient_accumulation_steps": 1,
        },
    )
    
    # For collecting squared gradients
    fisher_dict = None
    if is_main_process:
        fisher_dict = {n: torch.zeros_like(p.data) for n, p in transformer.named_parameters() if p.requires_grad}
    
    # Set model to training mode
    fisher_model_engine.train()
    
    # Process batches for Fisher computation
    num_processed_batches = 0
    progress_bar = tqdm(total=num_samples, desc="Computing Fisher Matrix", 
                        disable=not is_main_process)
    
    while num_processed_batches < num_samples:
        for batch in fisher_dataloader:
            # Zero gradients at the beginning of each batch
            fisher_model_engine.zero_grad()
            
            # Move data to device
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            
            # Convert images to latent space using VAE
            if vae is not None:
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
            else:
                model_input = pixel_values
                
            # Sample noise
            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            
            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()
            
            # Add noise to the model input
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            
            # Get text embeddings - modified to handle pre-computed embeddings
            if text_encoder is None:
                # Use pre-computed text embeddings
                encoder_hidden_states = batch["input_ids"].to(device=device, dtype=weight_dtype)
            else:
                # Compute text embeddings on-the-fly
                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    batch["input_ids"].to(device=device, dtype=weight_dtype),
                    batch["attention_mask"].to(device=device, dtype=weight_dtype) if "attention_mask" in batch else None,
                    text_encoder_use_attention_mask,
                )
                
            # Prepare micro-conditions
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if getattr(fisher_model_engine.module, 'config', fisher_model_engine.module).sample_size == 128:
                resolution = torch.tensor([train_dataset.size, train_dataset.size]).repeat(bsz, 1)
                aspect_ratio = torch.tensor([float(train_dataset.size / train_dataset.size)]).repeat(bsz, 1)
                resolution = resolution.to(dtype=weight_dtype, device=model_input.device)
                aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=model_input.device)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
                
            # Convert inputs to correct dtype
            noisy_model_input = noisy_model_input.to(dtype=fisher_model_engine.module.dtype)
            encoder_hidden_states = encoder_hidden_states.to(dtype=fisher_model_engine.module.dtype)
            timesteps = timesteps.to(dtype=fisher_model_engine.module.dtype)
            
            # Forward pass with DeepSpeed engine
            model_pred = fisher_model_engine(
                noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                added_cond_kwargs=added_cond_kwargs
            ).sample.chunk(2, 1)[0]
            
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
            # Compute loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Backward pass using DeepSpeed engine
            fisher_model_engine.backward(loss)
            
            # Gather and accumulate squared gradients on the main process
            if is_main_process:
                # We need to extract the full gradient from the DeepSpeed engine
                for name, param in fisher_model_engine.module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # In DeepSpeed ZeRO, gradients are partitioned across GPUs
                        # Use the optimizer state to get the gradient
                        if hasattr(fisher_optimizer, "get_partitioned_gradients"):
                            # For ZeRO-3 this would be needed, but not for ZeRO-2
                            grad = fisher_optimizer.get_partitioned_gradients(name)
                        else:
                            # For ZeRO-2, the gradients are already properly gathered
                            grad = param.grad.data.clone()
                        
                        # Accumulate squared gradients
                        fisher_dict[name] += grad.pow(2)
            
            # Synchronize after each batch to ensure clean gradient state
            ds.comm.barrier()
            
            # Update progress
            num_processed_batches += 1
            if is_main_process:
                progress_bar.update(1)
            
            if num_processed_batches >= num_samples:
                break
    
    # Average the Fisher Information Matrix
    if is_main_process:
        for name in fisher_dict:
            fisher_dict[name] /= num_processed_batches

        progress_bar.close()
        logger.info(f"Finished computing Fisher matrix from {num_processed_batches} batches")
    
    # Broadcast a flag to inform all processes that Fisher computation is complete
    completion_tensor = torch.tensor([1.0], device=device)
    ds.comm.broadcast(completion_tensor, 0)
    
    # Clean up the DeepSpeed engine used for Fisher computation
    del fisher_model_engine
    del fisher_optimizer
    
    # Force a garbage collection to free up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Only rank 0 returns the Fisher dict, others return None
    if is_main_process:
        return fisher_dict
    else:
        return None


def set_seed(seed):
    """Sets the random seed for all libraries used in the script."""
    if seed is None:
        return
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Fisher matrices for EWC regularization")
    
    # Model configuration
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type to use for training. Choose between 'epsilon' or 'v_prediction' or leave `None`.",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="Prompt with identifier specifying the instance.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="Folder containing the training data of class images for prior preservation.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="Prompt to specify class images for prior preservation.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help="Number of class images for prior preservation loss.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        help="Flag to add prior preservation.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=120,
        help="The maximum length of the tokenizer.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        help="Whether to use attention mask for the text encoder",
    )
    
    # Image processing options
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop the input images to the resolution.",
    )
    
    # Fisher computation settings
    parser.add_argument(
        "--fisher_samples",
        type=int,
        default=100,
        help="Number of samples to use for Fisher matrix computation.",
    )
    parser.add_argument(
        "--fisher_batch_size",
        type=int,
        default=1,
        help="Batch size to use for Fisher matrix computation.",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the computed Fisher matrix.",
    )
    
    # Hardware and performance settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible Fisher matrix computation.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For deepspeed: local rank for distributed training on GPUs",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed configuration file path",
    )
    
    # Add pre_compute_text_embeddings argument
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during Fisher computation and will leave more GPU memory available.",
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    if args.with_prior_preservation:
        if args.class_data_dir is None or args.class_prompt is None:
            raise ValueError("When using prior preservation, you must specify both class_data_dir and class_prompt")
    
    return args


def main():
    args = parse_args()
    
    # Initialize DeepSpeed
    ds.init_distributed()
    
    # Configure logging
    if ds.comm.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(args.output_dir, "fisher_computation.log"),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        # Log all hyperparameters
        logger.info("==========================================================")
        logger.info("          FISHER MATRIX COMPUTATION PARAMETERS           ")
        logger.info("==========================================================")
        hyperparams = vars(args)
        for key in sorted(hyperparams.keys()):
            logger.info(f"{key}: {hyperparams[key]}")
        logger.info("==========================================================")
    
    # Set the seed
    set_seed(args.seed)
    
    # Set up device and dtype
    device = torch.device("cuda", ds.comm.get_local_rank())
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Load models
    if ds.comm.get_rank() == 0:
        logger.info(f"Loading models from {args.pretrained_model_name_or_path}")
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler", 
        torch_dtype=weight_dtype
    )
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer", 
        torch_dtype=weight_dtype
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        torch_dtype=weight_dtype
    )
    text_encoder.to(device)
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        torch_dtype=weight_dtype
    )
    vae.to(device)
    
    transformer = Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype=weight_dtype
    )
    transformer.to(device)
    
    # Set models to eval mode
    text_encoder.eval()
    vae.eval()
    transformer.train()  # Train mode for transformer to compute proper gradients
    
    # Pre-compute text embeddings if enabled
    pre_computed_encoder_hidden_states = None
    pre_computed_class_prompt_encoder_hidden_states = None
    
    if args.pre_compute_text_embeddings:
        if ds.comm.get_rank() == 0:
            logger.info("Pre-computing text embeddings")
        
        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )
            return prompt_embeds
        
        # Compute embeddings for instance prompt
        pre_computed_encoder_hidden_states = compute_text_embeddings(args.instance_prompt)
        
        # Compute embeddings for class prompt if needed
        if args.with_prior_preservation and args.class_prompt is not None:
            pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(args.class_prompt)
        
        # Free up memory by deleting text encoder and tokenizer
        del text_encoder
        text_encoder = None
        del tokenizer
        tokenizer = None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        if ds.comm.get_rank() == 0:
            logger.info("Finished pre-computing text embeddings")
    
    # Create dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
    )
    
    if ds.comm.get_rank() == 0:
        logger.info(f"Dataset contains {len(train_dataset)} images")
    
    # Compute Fisher matrix
    fisher_dict = compute_fisher_matrix(
        transformer=transformer,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        device=device,
        num_samples=args.fisher_samples,
        batch_size=args.fisher_batch_size,
        weight_dtype=weight_dtype,
        text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
        with_prior_preservation=args.with_prior_preservation,
        dataloaders_kwargs={"num_workers": args.dataloader_num_workers}
    )
    
    # Save Fisher matrix
    if ds.comm.get_rank() == 0 and fisher_dict is not None:
        # Generate a descriptive filename for the Fisher matrix
        dataset_name = os.path.basename(os.path.normpath(args.instance_data_dir))
        fisher_filename = f"fisher_matrix.pt"
        fisher_path = os.path.join(args.output_dir, fisher_filename)
        
        logger.info(f"Saving Fisher matrix to {fisher_path}")
        torch.save(fisher_dict, fisher_path)
        logger.info(f"Fisher matrix saved successfully")
    
    # Clean up resources
    del text_encoder
    del vae
    del transformer
    gc.collect()
    torch.cuda.empty_cache()
    
    if ds.comm.get_rank() == 0:
        logger.info("Fisher matrix computation completed successfully!")


if __name__ == "__main__":
    main() 