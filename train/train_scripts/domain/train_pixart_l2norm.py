import argparse
import logging
import math
import os
import gc
import copy 
import json
import random
import shutil
import warnings
import deepspeed as ds
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from datetime import datetime

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import transformers
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


# Fix the logger initialization
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def log_validation(
    text_encoder, 
    tokenizer, 
    vae, 
    transformer, 
    args, 
    weight_dtype,
    global_step,
    prompt_embeds,
    text_inputs,
    negative_prompt_embeds,
    negative_text_inputs,
    timestamped_output_dir,
    tb_writer=None,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images for each prompt in validation_prompts list."
    )
    # Make sure to use the local rank from DeepSpeed's comm object
    device = torch.device("cuda", ds.comm.get_local_rank())
    
    # Load pipeline
    pipeline = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        transformer=transformer, 
        text_encoder=text_encoder, 
        vae=vae, 
        torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(device)

    # Get list of validation prompts
    validation_prompts = args.validation_prompts
    
    all_images = []
    
    # Process each validation prompt
    for prompt_idx, validation_prompt in enumerate(validation_prompts):
        logger.info(f"Processing validation prompt: {validation_prompt}")
        
        if args.pre_compute_text_embeddings:
            # Use pre-computed embeddings for this specific prompt
            prompt_embeds_current = prompt_embeds[prompt_idx] if isinstance(prompt_embeds, list) else prompt_embeds
            text_inputs_current = text_inputs[prompt_idx] if isinstance(text_inputs, list) else text_inputs
            
            prompt_attention_mask = text_inputs_current.attention_mask
            negative_prompt_attention_mask = negative_text_inputs.attention_mask
            pipeline_args = {
                "prompt_embeds": prompt_embeds_current,
                "negative_prompt_embeds": negative_prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_attention_mask": negative_prompt_attention_mask,
                "negative_prompt": None,
                "prompt": None
            }
        else:
            pipeline_args = {"prompt": validation_prompt}

        # run inference
        generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
        images = []
        if args.validation_images is None:
            for img_idx in range(args.num_validation_images):
                image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
                images.append(image)
        else:
            for image in args.validation_images:
                image = Image.open(image)
                image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
                images.append(image)
        
        all_images.extend(images)

        # Save images for this prompt
        save_dir = os.path.join(timestamped_output_dir, "validation", f"validation_{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        for i, image in enumerate(images):
            prompt_safe = validation_prompt.replace("/", "_").replace(" ", "_")[:50]  # Create safe filename
            image.save(os.path.join(save_dir, f"prompt{prompt_idx}_{i}_{prompt_safe}.png"))

    # Handle trackers - modified to use the passed tensorboard writer
    if tb_writer is not None and (args.report_to == "tensorboard" or args.report_to == "all"):
        np_images = np.stack([np.asarray(img) for img in all_images])
        tb_writer.add_images("validation", np_images, global_step, dataformats="NHWC")
    
    # Log to wandb if available
    if (args.report_to == "wandb" or args.report_to == "all") and is_wandb_available():
        import wandb
        if wandb.run is not None:
            # Group images by prompt for better organization
            for prompt_idx, validation_prompt in enumerate(validation_prompts):
                prompt_images = all_images[prompt_idx * args.num_validation_images:(prompt_idx + 1) * args.num_validation_images]
                wandb.log({
                    f"validation_prompt_{prompt_idx}": [
                        wandb.Image(image, caption=f"{i}: {validation_prompt}") 
                        for i, image in enumerate(prompt_images)
                    ]
                })

    del pipeline
    torch.cuda.empty_cache()

    return all_images

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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script for PixArt-Alpha.")
    
    # Add DeepSpeed arguments
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed configuration file path",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For deepspeed: local rank for distributed training on GPUs",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        help="ZeRO optimization stage for DeepSpeed (0, 1, 2, 3)",
    )
    
    # Model configuration
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--load_transformer_path",
        type=str,
        default=None,
        help="Path to transformer model.",
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
        "--validation_prompt", 
        type=str, 
        default=None, 
        help="A prompt that is sampled during training for inference. Deprecated, use validation_prompts instead."
    )
    parser.add_argument(
        "--validation_prompts", 
        type=str, 
        default=None, 
        help="Comma-separated list of prompts that are sampled during training for inference."
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
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
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    
    # Training process settings
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=16, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    
    # Optimizer and learning rate settings
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=500, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    # Validation and logging settings
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    
    # Output and checkpoint settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pixart-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    
    # Hardware and performance settings
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    # Add L2 regularization parameter
    parser.add_argument(
        "--l2_reg_weight",
        type=float,
        default=10.0,
        help="Weight for L2 regularization against the initial model parameters",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Handle validation prompts (convert string to list if needed)
    if args.validation_prompts is not None:
        args.validation_prompts = [p.strip() for p in args.validation_prompts.split(",")]
    elif args.validation_prompt is not None:
        # For backward compatibility
        args.validation_prompts = [args.validation_prompt]
    else:
        args.validation_prompts = []
        
    return args

def set_seed(seed):
    """
    Sets the random seed for all libraries used in the script.
    """
    if seed is None:
        return
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = parse_args()
    
    # Initialize DeepSpeed before anything else
    ds.init_distributed()
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # timestamped_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    timestamped_output_dir = os.path.join(args.output_dir, f"run")
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    # Configure logging only on the main process
    if ds.comm.get_rank() == 0:
        logging.basicConfig(
            filename=os.path.join(timestamped_output_dir, "train.log"),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        # Log all hyperparameters in a structured format
        logger.info("==========================================================")
        logger.info("                TRAINING HYPERPARAMETERS                  ")
        logger.info("==========================================================")
        hyperparams = vars(args)
        # Sort for consistent display
        for key in sorted(hyperparams.keys()):
            logger.info(f"{key}: {hyperparams[key]}")
        logger.info("==========================================================")
                
        # Save hyperparameters as JSON for programmatic access
        with open(os.path.join(timestamped_output_dir, "hyperparameters.json"), "w") as f:
            # Convert non-serializable objects to strings
            cleaned_params = {}
            for k, v in hyperparams.items():
                if isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
                    cleaned_params[k] = v
                else:
                    cleaned_params[k] = str(v)
            json.dump(cleaned_params, f, indent=2)
    
    # Set up wandb if needed
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if ds.comm.get_rank() == 0:  # Only initialize wandb on the main process
            wandb.init(project="pixart-alpha", name=f"run_{timestamp}")
            # Log hyperparameters to wandb
            wandb.config.update(vars(args))

    # Set the training seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if ds.comm.get_rank() == 0:  # Only create directories on the main process
        if timestamped_output_dir is not None:
            os.makedirs(timestamped_output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights to half-precision
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load models and move them to the correct device based on local_rank
    device = torch.device("cuda", ds.comm.get_local_rank())
    
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

    # Pre-compute text embeddings if enabled
    if args.pre_compute_text_embeddings:
        logging.info("Pre-computing text embeddings")

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                )

            return prompt_embeds, text_inputs
        validation_prompt_negative_prompt_embeds, validation_prompt_negative_text_inputs = compute_text_embeddings("")

        # Handle multiple validation prompts
        if args.validation_prompts:
            validation_prompt_encoder_hidden_states = []
            validation_prompt_encoder_text_inputs = []
            
            for prompt in args.validation_prompts:
                hidden_states, text_inputs = compute_text_embeddings(prompt)
                validation_prompt_encoder_hidden_states.append(hidden_states)
                validation_prompt_encoder_text_inputs.append(text_inputs)
        else:
            validation_prompt_encoder_hidden_states = None
            validation_prompt_encoder_text_inputs = None

        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
    else:
        validation_prompt_encoder_hidden_states = None
        validation_prompt_encoder_text_inputs = None
        validation_prompt_negative_prompt_embeds = None
        validation_prompt_negative_text_inputs = None

    # Dataset and DataLoaders creation:
    train_dataset = ExternalData(
        root=args.data_dir,
        image_list_json=args.data_info,
        resolution=args.resolution,
        center_crop=args.center_crop,
        max_length=args.tokenizer_max_length,
        logger=logger,
        weight_dtype=weight_dtype,
    )

    # Create sampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=ds.comm.get_world_size(),
        rank=ds.comm.get_rank(),
        shuffle=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    if args.load_transformer_path is not None:
        logger.info(f"Loading transformer from {args.load_transformer_path}")
        transformer = Transformer2DModel.from_pretrained(args.load_transformer_path, torch_dtype=weight_dtype)
    else:
        logger.info(f"Loading transformer from {args.pretrained_model_name_or_path}")
        transformer = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)
    transformer.to(device)

    # FIX: Explicitly set transformer parameters to require gradients
    for param in transformer.parameters():
        param.requires_grad = True

    # # Create a copy of the transformer model for L2 regularization
    reference_transformer = copy.deepcopy(transformer)
    reference_transformer.requires_grad_(False)  # Freeze the reference model
    reference_transformer.to(device)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * ds.comm.get_world_size()

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # FIX: Check if we have parameters to optimize
    if len(params_to_optimize) == 0:
        raise ValueError("No parameters to optimize. Check model configuration.")

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * ds.comm.get_world_size(),
        num_training_steps=args.max_train_steps * ds.comm.get_world_size(),
    )

    # DeepSpeed engine setup - simplify parameter filtering
    model_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    
    # Create DeepSpeed model engine with explicit config path
    model_engine, optimizer, _, lr_scheduler = ds.initialize(
        model=transformer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_parameters=model_parameters,
        config=args.deepspeed,
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * ds.comm.get_world_size() * args.gradient_accumulation_steps

    if ds.comm.get_rank() == 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Output directory = {timestamped_output_dir}")

    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not (ds.comm.get_rank() == 0),
    )

    # Initialize tensorboard writer once if needed
    tb_writer = None
    if ds.comm.get_rank() == 0 and (args.report_to == "tensorboard" or args.report_to == "all"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(timestamped_output_dir, "logs"))
            logger.info(f"Initialized TensorBoard writer at {os.path.join(timestamped_output_dir, 'logs')}")
        except ImportError:
            logger.warning("TensorBoard specified in report_to but not installed. Skipping TensorBoard logging.")

    for epoch in range(first_epoch, args.num_train_epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)

            if vae is not None:
                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
            else:
                model_input = pixel_values

            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # Get the text embedding for conditioning
            if args.pre_compute_text_embeddings:
                encoder_hidden_states = batch["input_ids"].to(device=device, dtype=weight_dtype)
            else:
                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    batch["input_ids"].to(device=device, dtype=weight_dtype),
                    batch["attention_mask"].to(device=device, dtype=weight_dtype),
                )

            # Prepare micro-conditions.
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if getattr(model_engine.module, 'config', model_engine.module).sample_size == 128:
                resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1)
                aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1)
                resolution = resolution.to(dtype=weight_dtype, device=model_input.device)
                aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=model_input.device)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            noisy_model_input = noisy_model_input.to(dtype=model_engine.module.dtype)
            encoder_hidden_states = encoder_hidden_states.to(dtype=model_engine.module.dtype)
            timesteps = timesteps.to(dtype=model_engine.module.dtype)

            # Predict the noise residual
            model_pred = model_engine(
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

            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # Add L2 regularization loss between current model and reference model
            l2_loss = 0.0
            for param, ref_param in zip(model_engine.module.parameters(), reference_transformer.parameters()):
                if param.requires_grad:
                    l2_loss += F.mse_loss(param, ref_param.detach(), reduction='sum')
            # Scale the L2 loss and add it to the original loss
            l2_loss = args.l2_reg_weight * l2_loss

            # Add L2 regularization loss to the total loss
            total_loss = loss + l2_loss

            # Backpropagate with DeepSpeed engine
            model_engine.backward(total_loss)
            
            # Calculate gradient norm for logging
            if (global_step % 10 == 0) and (ds.comm.get_rank() == 0) and args.report_to != "none":
                grad_norm = 0.0
                for param in model_engine.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
            
            model_engine.step()
            
            # Gather loss from all processes
            if ds.comm.get_world_size() > 1:
                loss_list = [torch.zeros_like(loss) for _ in range(ds.comm.get_world_size())]
                ds.comm.all_gather(loss_list, loss)
                train_loss += sum(loss_list).item() / len(loss_list)
            else:
                train_loss += loss.item()

            # Log metrics to tensorboard/wandb
            if (global_step % 10 == 0) and (ds.comm.get_rank() == 0) and args.report_to != "none":
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm if 'grad_norm' in locals() else 0.0,
                    "train/step": global_step,
                    "train/epoch": epoch,
                }
                
                # Log to tensorboard
                if args.report_to == "tensorboard" or args.report_to == "all":
                    try:
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                    except AttributeError:
                        logger.warning("Tensorboard not available. Install it with pip install tensorboard")
                
                # Log to wandb
                if (args.report_to == "wandb" or args.report_to == "all") and is_wandb_available():
                    import wandb
                    if wandb.run is not None:
                        wandb.log(logs, step=global_step)
            
            progress_bar.update(1)
            global_step += 1

            # Log and save
            if global_step % args.checkpointing_steps == 0:
                # Save only the transformer model
                transformer_save_path = os.path.join(timestamped_output_dir, f"transformer-{global_step}")
                os.makedirs(transformer_save_path, exist_ok=True)
                transformer.save_pretrained(transformer_save_path)
                logger.info(f"Saved transformer checkpoint to {transformer_save_path}")
                
                # Prune old checkpoints if needed
                if args.checkpoints_total_limit is not None:
                    transformer_checkpoints = sorted(
                        [d for d in os.listdir(timestamped_output_dir) if d.startswith("transformer-")],
                        key=lambda x: int(x.split("-")[1])
                    )
                    
                    if len(transformer_checkpoints) > args.checkpoints_total_limit:
                        for checkpoint in transformer_checkpoints[:len(transformer_checkpoints) - args.checkpoints_total_limit]:
                            shutil.rmtree(os.path.join(timestamped_output_dir, checkpoint))
                            logger.info(f"Removed old transformer checkpoint: {checkpoint}")
            
            # Run validation
            if args.validation_prompts and global_step % args.validation_steps == 0 and ds.comm.get_rank() == 0:
                images = log_validation(
                    text_encoder, 
                    tokenizer, 
                    vae, 
                    model_engine.module,  # Use the base model for validation
                    args, 
                    weight_dtype, 
                    global_step,
                    validation_prompt_encoder_hidden_states,
                    validation_prompt_encoder_text_inputs,
                    validation_prompt_negative_prompt_embeds,
                    validation_prompt_negative_text_inputs,
                    timestamped_output_dir,
                    tb_writer=tb_writer,
                )
                
                # Log validation images to wandb if enabled
                if args.report_to == "wandb":
                    for prompt_idx, validation_prompt in enumerate(args.validation_prompts):
                        prompt_images = images[prompt_idx * args.num_validation_images:(prompt_idx + 1) * args.num_validation_images]
                        wandb.log({
                            f"validation_prompt_{prompt_idx}": [
                                wandb.Image(image, caption=f"{i}: {validation_prompt}")
                                for i, image in enumerate(prompt_images)
                            ]
                        })

            logs = {
                "task_loss": loss.detach().item(),
                "l2_reg_loss": l2_loss.detach().item(),
                "total_loss": total_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            logging.info(logs)
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
    
    # Extract and save the underlying model
    if hasattr(model_engine, "module"):
        transformer = model_engine.module
    else:
        transformer = model_engine
        
    # Save in HuggingFace formats
    transformer.save_pretrained(os.path.join(timestamped_output_dir, f"transformer-final"))
    
    logger.info(f"Saved final models to {timestamped_output_dir}")

    # Clean up resources at the end
    if tb_writer is not None:
        tb_writer.close()
        logger.info("Closed TensorBoard writer")
    
    # End of training
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
