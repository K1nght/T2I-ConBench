import argparse
import logging
import os
import gc
import json
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm.auto import tqdm
from pathlib import Path
import deepspeed as ds
from PIL import Image
from PIL.ImageOps import exif_transpose
import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset


import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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

def get_data_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return os.path.join(DATA_ROOT, data_dir)

class ExternalData(Dataset):
    def __init__(self,
                 root,
                 image_list_json='data_info.json',
                 resolution=256,
                 center_crop=False,
                 max_length=120,
                 logger=None,
                 weight_dtype=None,
                 ):
        self.root = get_data_path(root)
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.max_lenth = max_length
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_samples = []
        self.txt_feat_samples = []
        self.weight_dtype = weight_dtype
        self.logger = logger
        self.logger.info(f"T5 max token length: {self.max_lenth}")

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, json_file))
            self.logger.info(f"{json_file} data volume: {len(meta_data)}")
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([
                item['path'] for item in meta_data_clean
            ])
            self.txt_samples.extend([item['prompt'] for item in meta_data_clean])
            self.txt_feat_samples.extend([
                os.path.join(
                    "/".join(item['path'].split('/')[:-2]),
                    'caption_feature',
                    f'{item["idx"]}.npz'
                ) for item in meta_data_clean
            ])

        self.logger.info(f"Total data volume: {len(self.meta_data_clean)}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )   

    def getdata(self, index):
        example = {}
        img_path = self.img_samples[index]
        instance_image = Image.open(img_path)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        npz_path = self.txt_feat_samples[index]
        txt = self.txt_samples[index]

        attention_mask = torch.ones(1, 1, self.max_lenth)     # 1x1xT
        txt_info = np.load(npz_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        example["instance_prompt_ids"] = txt_fea
        example["instance_attention_mask"] = attention_mask

        return example

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details {self.img_samples[idx]}: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)

        return meta_data

    def sample_subset(self, ratio):
        sampled_idx = random.sample(list(range(len(self))), int(len(self) * ratio))
        self.img_samples = [self.img_samples[i] for i in sampled_idx]

    def __len__(self):
        return len(self.img_samples)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def collate_fn(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

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
    args,
    dataloaders_kwargs=None,
):
    """
    Computes the Fisher Information Matrix for Elastic Weight Consolidation (EWC).
    The Fisher matrix approximates the importance of each parameter.
    This version is compatible with DeepSpeed ZeRO Stage 2/3.
    """
    logger.info("Computing Fisher Information Matrix for EWC using DeepSpeed...")
    
    if dataloaders_kwargs is None:
        dataloaders_kwargs = {}
    
    # Create dataloader with specified batch size for Fisher computation
    fisher_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
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
    
    # Process batches for Fisher computation
    num_processed_batches = 0
    progress_bar = tqdm(total=num_samples, desc="Computing Fisher Matrix",
                        disable=not is_main_process)
    
    for batch in fisher_dataloader:
        if num_processed_batches >= num_samples:
            break
            
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
        
        # Get text embeddings
        if args.pre_compute_text_embeddings:
            encoder_hidden_states = batch["input_ids"].to(device=device, dtype=weight_dtype)
        else:
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"].to(device=device, dtype=weight_dtype),
                batch["attention_mask"].to(device=device, dtype=weight_dtype) if "attention_mask" in batch else None,
            )
            
        # Prepare micro-conditions
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if getattr(fisher_model_engine.module, 'config', fisher_model_engine.module).sample_size == 128:
            resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1)
            aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Fisher Information Matrix for PixArt-Alpha EWC.")
    
    # Model configuration
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    # Training data configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--data_info",
        type=str,
        default="data_info.json",
        required=True,
        nargs="+",
        help="The data info json file(s). Can specify multiple JSON files by space-separating them.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=120, # Pixart-Alpha's max length 120
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    
    # Image processing options
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the dataset will be resized to this resolution",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop the input images to the resolution.",
    )
    
    # Fisher computation settings
    parser.add_argument(
        "--fisher_samples", 
        type=int, 
        default=1000,
        help="Number of batches to use for Fisher Information Matrix computation."
    )
    parser.add_argument(
        "--fisher_batch_size",
        type=int,
        default=4,
        help="Batch size for Fisher Information Matrix computation."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the computed Fisher matrix."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible computation."
    )
    
    # DeepSpeed arguments
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training on GPUs",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed configuration file path. If not specified, a default configuration will be used.",
    )
    
    args = parser.parse_args()
    
    # Check for environment variable LOCAL_RANK and update args if needed
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize distributed training with DeepSpeed
    ds.init_distributed()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set up device - use the device from DeepSpeed's local rank
    device = torch.device(f"cuda:{ds.comm.get_local_rank()}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    is_main_process = ds.comm.get_rank() == 0
    
    if is_main_process:
        logger.info(f"Using device: {device}, dtype: {weight_dtype}")
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load models
    if is_main_process:
        logger.info(f"Loading models from {args.pretrained_model_name_or_path}")
    
    # Load scheduler, tokenizer, text encoder, VAE, and transformer
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
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        torch_dtype=weight_dtype
    )
    vae.requires_grad_(False)
    vae.to(device)
    
    transformer = Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype=weight_dtype
    )
    transformer.to(device)
    
    # Set models to eval/train mode
    text_encoder.eval()
    vae.eval()
    transformer.train()  # Train mode for transformer to compute proper gradients
    
    # Pre-compute text embeddings if enabled
    if args.pre_compute_text_embeddings:
        logging.info("Pre-computing text embeddings")
        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

    # Load dataset
    if is_main_process:
        logger.info(f"Loading dataset from {args.data_dir}")
    
    train_dataset = ExternalData(
        root=args.data_dir,
        image_list_json=args.data_info,
        resolution=args.resolution,
        center_crop=args.center_crop,
        max_length=args.tokenizer_max_length,
        logger=logger,
        weight_dtype=weight_dtype,
    )
    
    if is_main_process:
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
        args=args,
        dataloaders_kwargs={"num_workers": args.dataloader_num_workers}
    )
    
    # Save Fisher matrix (only on rank 0)
    if is_main_process and fisher_dict is not None:
        logger.info(f"Saving Fisher matrix to {args.output_path}")
        torch.save(fisher_dict, args.output_path)
        logger.info("Fisher matrix computation completed successfully.")
    
    # Clean up resources
    del text_encoder
    del vae
    del transformer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 