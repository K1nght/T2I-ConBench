import argparse
import logging
import math
import os
import gc
import random
import shutil
import warnings
import deepspeed as ds
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from datetime import datetime
import time
import json
import numpy as np

import datasets
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

class ExternalReplayData(Dataset):
    """
    A dataset for external data to be used as replay samples.
    Based on the ExternalData class from train_pixart_ds.py
    """
    def __init__(self,
                 root,
                 image_list_json='data_info.json',
                 resolution=512,
                 center_crop=False,
                 max_length=120,
                 tokenizer=None,
                 text_encoder=None,
                 pre_compute_text_embeddings=False,
                 ):
        self.root = root
        self.resolution = resolution
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_samples = []
        self.txt_feat_samples = []
        
        print(f"Loading external replay data from {root}")
        
        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, json_file))
            print(f"{json_file} data volume: {len(meta_data)}")
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

        print(f"Total data volume: {len(self.meta_data_clean)}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)
        return meta_data
    
    def __len__(self):
        return len(self.img_samples)
    
    def __getitem__(self, idx):
        for _ in range(5):  # Try 5 times in case of error
            try:
                example = {}
                img_path = self.img_samples[idx]
                
                # Load and process image
                instance_image = Image.open(img_path)
                instance_image = exif_transpose(instance_image)
                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
                example["instance_images"] = self.image_transforms(instance_image)
                
                # Handle text features/embeddings
                if self.txt_feat_samples and os.path.exists(self.txt_feat_samples[idx]):
                    # Use pre-computed text features
                    npz_path = self.txt_feat_samples[idx]
                    txt_info = np.load(npz_path)
                    txt_fea = torch.from_numpy(txt_info['caption_feature'])
                    attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
                    
                    example["instance_prompt_ids"] = txt_fea
                    example["instance_attention_mask"] = attention_mask
                else:
                    # Generate text features on the fly
                    txt = self.txt_samples[idx]
                    if self.pre_compute_text_embeddings and self.text_encoder is not None:
                        text_inputs = tokenize_prompt(
                            self.tokenizer, txt, tokenizer_max_length=self.max_length
                        )
                        with torch.no_grad():
                            example["instance_prompt_ids"] = encode_prompt(
                                self.text_encoder,
                                text_inputs.input_ids.to(self.text_encoder.device),
                                text_inputs.attention_mask.to(self.text_encoder.device),
                            )
                    else:
                        text_inputs = tokenize_prompt(
                            self.tokenizer, txt, tokenizer_max_length=self.max_length
                        )
                        example["instance_prompt_ids"] = text_inputs.input_ids
                        example["instance_attention_mask"] = text_inputs.attention_mask
                
                # Store the source for later loss weighting
                example["is_external_replay"] = True
                
                return example
            except Exception as e:
                print(f"Error processing external replay data {img_path}: {str(e)}")
                idx = np.random.randint(len(self))
        
        # If all retries failed
        raise RuntimeError('Could not load external replay data after multiple attempts.')

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
        replay_paths=None,  # New parameter to accept replay image paths
        replay_prompts=None,  # New parameter to accept replay prompts
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
        
        # Add replay images to instance images if provided
        if replay_paths and replay_prompts:
            self.replay_paths = replay_paths
            self.replay_prompts = replay_prompts
            self.instance_images_path.extend(replay_paths)
            self.num_instance_images = len(self.instance_images_path)
            # Create a map for image path -> prompt
            self.image_to_prompt = {}
            # Map current task images to its prompt
            for path in list(Path(instance_data_root).iterdir()):
                self.image_to_prompt[str(path)] = instance_prompt
            # Map replay images to their prompts
            for path, prompt in zip(replay_paths, replay_prompts):
                self.image_to_prompt[str(path)] = prompt
            self.has_replay = True
        else:
            self.num_instance_images = len(self.instance_images_path)
            self.instance_prompt = instance_prompt
            self.has_replay = False
            
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
        instance_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_path)
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        # Get the appropriate prompt for this image (either current task or replay)
        if self.has_replay:
            instance_prompt = self.image_to_prompt[str(instance_path)]
            # Mark as replay sample if it's from previous tasks
            example["is_replay"] = str(instance_path) not in [str(p) for p in list(Path(self.instance_data_root).iterdir())]
        else:
            instance_prompt = self.instance_prompt
            example["is_replay"] = False

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, instance_prompt, tokenizer_max_length=self.tokenizer_max_length
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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script for PixArt-Alpha with DreamBooth support.")
    
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
        "--load_transformer_path",
        type=str,
        default=None,
        help="Path to transformer model.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    
    # Training data configuration
    parser.add_argument(
        "--instance_data_dirs",
        type=str,
        default=None,
        required=True,
        help="Comma-separated list of folders containing the training data of instance images for continual learning.",
    )
    parser.add_argument(
        "--instance_prompts",
        type=str,
        default=None,
        required=True,
        help="Comma-separated list of prompts with identifiers specifying the instances (must match the order of instance_data_dirs).",
    )
    # Class data for prior preservation (continual learning)
    parser.add_argument(
        "--class_data_dirs",
        type=str,
        default=None,
        required=False,
        help="Comma-separated list of folders containing the training data of class images for continual learning prior preservation.",
    )
    parser.add_argument(
        "--class_prompts",
        type=str,
        default=None,
        help="Comma-separated list of prompts to specify class images for prior preservation (must match the order of class_data_dirs).",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--validation_prompt", 
        type=str, 
        default=None, 
        help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight", 
        type=float, 
        default=1.0, 
        help="The weight of prior preservation loss."
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=120, # Pixart-Alpha's max length 120
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
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
    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=True,
        help="Total number of training steps to perform per dataset.",
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

    # Add external replay options
    parser.add_argument(
        "--external_replay_data",
        type=str,
        default=None,
        help="Path to external data directory for replay (used in addition to task replay)",
    )
    parser.add_argument(
        "--external_replay_json",
        type=str,
        default="data_info.json",
        nargs="+",
        help="The data info json file(s). Can specify multiple JSON files by space-separating them.",
    )
    parser.add_argument(
        "--replay_loss_coefficient",
        type=float,
        default=0.5,
        help="Weight coefficient for loss on replay samples (both internal and external)",
    )
    parser.add_argument(
        "--max_external_replay_samples", 
        type=int, 
        default=100,
        help="Maximum number of external replay samples to use per task",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Parse comma-separated values into lists
    args.instance_data_dirs = [dir.strip() for dir in args.instance_data_dirs.split(",")]
    args.instance_prompts = [prompt.strip() for prompt in args.instance_prompts.split(",")]
    
    # Verify matching lengths
    if len(args.instance_data_dirs) != len(args.instance_prompts):
        raise ValueError(f"Number of instance directories ({len(args.instance_data_dirs)}) must match number of instance prompts ({len(args.instance_prompts)})")
    
    # Handle class data for prior preservation
    if args.with_prior_preservation:
        if args.class_data_dirs is None or args.class_prompts is None:
            raise ValueError("When using prior preservation in continual learning, you must specify both class_data_dirs and class_prompts")
        
        args.class_data_dirs = [dir.strip() for dir in args.class_data_dirs.split(",")]
        args.class_prompts = [prompt.strip() for prompt in args.class_prompts.split(",")]
        
        if len(args.class_data_dirs) != len(args.class_prompts):
            raise ValueError(f"Number of class directories ({len(args.class_data_dirs)}) must match number of class prompts ({len(args.class_prompts)})")
        
        if len(args.class_data_dirs) != len(args.instance_data_dirs):
            raise ValueError(f"Number of class directories ({len(args.class_data_dirs)}) must match number of instance directories ({len(args.instance_data_dirs)})")
    else:
        # Set default empty lists for clarity
        args.class_data_dirs = []
        args.class_prompts = []
    
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

def train_on_dataset(
    model_engine,
    text_encoder,
    tokenizer,
    vae,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    instance_data_dir,
    instance_prompt,
    class_data_dir,
    class_prompt,
    args,
    weight_dtype,
    device,
    timestamped_output_dir,
    start_global_step=0,
    pre_computed_encoder_hidden_states=None,
    pre_computed_encoder_text_inputs=None,
    pre_computed_class_prompt_encoder_hidden_states=None,
    pre_computed_class_prompt_encoder_text_inputs=None,
    replay_paths=None,
    replay_prompts=None,
    external_replay_dataset=None,
):
    """Train the model on a single dataset."""
    
    # Dataset and DataLoaders creation:
    main_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        class_data_root=class_data_dir if args.with_prior_preservation else None,
        class_prompt=class_prompt if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
        replay_paths=replay_paths,
        replay_prompts=replay_prompts,
    )
    
    # Create sampler for distributed training for main dataset
    main_sampler = torch.utils.data.distributed.DistributedSampler(
        main_dataset,
        num_replicas=ds.comm.get_world_size(),
        rank=ds.comm.get_rank(),
        shuffle=True,
    )

    main_dataloader = torch.utils.data.DataLoader(
        main_dataset,
        batch_size=args.train_batch_size,
        sampler=main_sampler,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
    
    # Setup external replay dataloader if available
    external_dataloader = None
    if external_replay_dataset is not None:
        # Subsample external dataset if needed
        if len(external_replay_dataset) > args.max_external_replay_samples:
            indices = random.sample(range(len(external_replay_dataset)), args.max_external_replay_samples)
            external_replay_dataset = torch.utils.data.Subset(external_replay_dataset, indices)
            if ds.comm.get_rank() == 0:
                logger.info(f"Sampled {len(indices)} examples from external replay dataset")
        
        # Create sampler for distributed training for external dataset
        external_sampler = torch.utils.data.distributed.DistributedSampler(
            external_replay_dataset,
            num_replicas=ds.comm.get_world_size(),
            rank=ds.comm.get_rank(),
            shuffle=True,
        )
        
        external_dataloader = torch.utils.data.DataLoader(
            external_replay_dataset,
            batch_size=args.train_batch_size,  # Use smaller batch size for external data
            sampler=external_sampler,
            collate_fn=lambda examples: collate_fn(examples, False),  # No prior preservation for external data
            num_workers=args.dataloader_num_workers,
        )
        
        if ds.comm.get_rank() == 0:
            logger.info(f"Created external replay dataloader with {len(external_replay_dataset)} samples")
    
    # Get max_train_steps directly from args
    max_train_steps = args.max_train_steps
    
    # Train!
    total_batch_size = args.train_batch_size * ds.comm.get_world_size() * args.gradient_accumulation_steps

    if ds.comm.get_rank() == 0:
        logger.info(f"***** Training on dataset: {instance_data_dir} with prompt: {instance_prompt} *****")
        if args.with_prior_preservation:
            logger.info(f"***** Using prior preservation with class data: {class_data_dir} and prompt: {class_prompt} *****")
        logger.info(f"  Num examples = {len(main_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        if args.replay_loss_coefficient != 1.0:
            logger.info(f"  Using replay loss coefficient = {args.replay_loss_coefficient}")

    global_step = start_global_step
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=0,
        desc=f"Training on {os.path.basename(instance_data_dir)}",
        disable=not (ds.comm.get_rank() == 0),
    )

    # Calculate number of steps per epoch for setting sampler epoch
    steps_per_epoch = math.ceil(len(main_dataloader) / args.gradient_accumulation_steps)
    
    model_engine.train()
    train_loss = 0.0
    
    # Track loss statistics separately for current task and replay
    if ds.comm.get_rank() == 0:
        task_loss_sum = 0.0
        replay_loss_sum = 0.0
        external_replay_loss_sum = 0.0
        task_sample_count = 0
        replay_sample_count = 0
        external_replay_sample_count = 0
    
    # Continue training until we reach max_train_steps
    while global_step < start_global_step + max_train_steps:
        # Set the epoch for the sampler based on current step
        current_epoch = global_step // steps_per_epoch
        main_sampler.set_epoch(current_epoch)
        if external_dataloader is not None:
            external_sampler.set_epoch(current_epoch)
        
        # Create iterators for both dataloaders
        main_iter = iter(main_dataloader)
        external_iter = iter(external_dataloader) if external_dataloader is not None else None
        
        # Continue until we finish one epoch on the main dataset
        for _ in range(len(main_dataloader)):
            # Get a batch from the main dataset
            batch = next(main_iter)
            
            # Process the main batch
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
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )

            # Prepare micro-conditions
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if getattr(model_engine.module, 'config', model_engine.module).sample_size == 128:
                resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1)
                aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1)
                resolution = resolution.to(dtype=weight_dtype, device=model_input.device)
                aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=model_input.device)
                added_cond_kwargs = {
                    "resolution": resolution, 
                    "aspect_ratio": aspect_ratio
                }
            
            # Convert inputs to model dtype
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
            

            # proceed with standard loss calculation
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute SNR-weighted loss
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    snr = snr + 1
                mse_loss_weights = (torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # If using prior preservation, add it to the total loss
            if args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                # Add the prior loss to the instance loss
                loss = loss + args.prior_loss_weight * prior_loss

            # Process external replay batch (if available)
            if external_iter is not None:
                try:
                    # Get a batch from the external dataset
                    external_batch = next(external_iter)
                    
                    # Process the external batch
                    external_pixel_values = external_batch["pixel_values"].to(device=device, dtype=weight_dtype)
                    
                    if vae is not None:
                        # Convert images to latent space
                        external_model_input = vae.encode(external_pixel_values).latent_dist.sample()
                        external_model_input = external_model_input * vae.config.scaling_factor
                    else:
                        external_model_input = external_pixel_values
                    
                    external_noise = torch.randn_like(external_model_input)
                    external_bsz = external_model_input.shape[0]
                    
                    # Sample timesteps
                    external_timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (external_bsz,), 
                        device=external_model_input.device
                    )
                    external_timesteps = external_timesteps.long()
                    
                    # Add noise
                    external_noisy_model_input = noise_scheduler.add_noise(
                        external_model_input, external_noise, external_timesteps
                    )
                    
                    # Get text embeddings
                    if args.pre_compute_text_embeddings:
                        external_encoder_hidden_states = external_batch["input_ids"].to(device=device, dtype=weight_dtype)
                    else:
                        external_encoder_hidden_states = encode_prompt(
                            text_encoder,
                            external_batch["input_ids"].to(device=device, dtype=weight_dtype),
                            external_batch["attention_mask"].to(device=device, dtype=weight_dtype),
                            text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                        )
                    
                    # Prepare micro-conditions
                    external_added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                    if getattr(model_engine.module, 'config', model_engine.module).sample_size == 128:
                        external_resolution = torch.tensor([args.resolution, args.resolution]).repeat(external_bsz, 1)
                        external_aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(external_bsz, 1)
                        external_resolution = external_resolution.to(dtype=weight_dtype, device=external_model_input.device)
                        external_aspect_ratio = external_aspect_ratio.to(dtype=weight_dtype, device=external_model_input.device)
                        external_added_cond_kwargs = {
                            "resolution": external_resolution, 
                            "aspect_ratio": external_aspect_ratio
                        }
                    
                    # Convert to model dtype
                    external_noisy_model_input = external_noisy_model_input.to(dtype=model_engine.module.dtype)
                    external_encoder_hidden_states = external_encoder_hidden_states.to(dtype=model_engine.module.dtype)
                    external_timesteps = external_timesteps.to(dtype=model_engine.module.dtype)
                    
                    # Predict noise
                    external_model_pred = model_engine(
                        external_noisy_model_input,
                        encoder_hidden_states=external_encoder_hidden_states,
                        timestep=external_timesteps,
                        added_cond_kwargs=external_added_cond_kwargs
                    ).sample.chunk(2, 1)[0]
                    
                    # Get target
                    if noise_scheduler.config.prediction_type == "epsilon":
                        external_target = external_noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        external_target = noise_scheduler.get_velocity(
                            external_model_input, external_noise, external_timesteps
                        )
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                    # Calculate external loss
                    if args.snr_gamma is None:
                        external_loss = F.mse_loss(
                            external_model_pred.float(), external_target.float(), reduction="mean"
                        )
                    else:
                        # SNR-weighted loss
                        external_snr = compute_snr(noise_scheduler, external_timesteps)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            external_snr = external_snr + 1
                        
                        external_mse_loss_weights = (
                            torch.stack([external_snr, args.snr_gamma * torch.ones_like(external_timesteps)], dim=1).min(dim=1)[0] / external_snr
                        )
                        
                        external_loss = F.mse_loss(
                            external_model_pred.float(), external_target.float(), reduction="none"
                        )
                        external_loss = external_loss.mean(dim=list(range(1, len(external_loss.shape)))) * external_mse_loss_weights
                        external_loss = external_loss.mean()
                    
                    # Apply replay coefficient to external loss
                    external_loss = args.replay_loss_coefficient * external_loss
                    
                    # Track statistics
                    if ds.comm.get_rank() == 0:
                        external_replay_loss_sum += (external_loss / args.replay_loss_coefficient).item() * external_bsz
                        external_replay_sample_count += external_bsz
                    
                    # Add external loss to main loss
                    loss = loss + external_loss
                    
                except StopIteration:
                    # If we run out of external samples, reset the iterator
                    external_iter = iter(external_dataloader)
            
            # Backpropagate with DeepSpeed engine
            model_engine.backward(loss)
            model_engine.step()
            
            # Gather loss from all processes
            if ds.comm.get_world_size() > 1:
                loss_list = [torch.zeros_like(loss) for _ in range(ds.comm.get_world_size())]
                ds.comm.all_gather(loss_list, loss)
                train_loss += sum(loss_list).item() / len(loss_list)
            else:
                train_loss += loss.item()

            # Log detailed loss breakdown every 50 steps
            if ds.comm.get_rank() == 0 and global_step % 50 == 0:
                log_message = f"Step {global_step}: "
                
                if task_sample_count > 0:
                    log_message += f"Task loss avg: {task_loss_sum/max(1, task_sample_count):.6f}, "
                    
                if replay_sample_count > 0:
                    log_message += f"Task replay loss avg: {replay_loss_sum/max(1, replay_sample_count):.6f}, "
                    log_message += f"Task replay samples: {replay_sample_count}, "
                
                if external_replay_sample_count > 0:
                    log_message += f"External replay loss avg: {external_replay_loss_sum/max(1, external_replay_sample_count):.6f}, "
                    log_message += f"External replay samples: {external_replay_sample_count}, "
                
                logger.info(log_message)
                
                # Reset counters
                task_loss_sum = 0.0
                replay_loss_sum = 0.0
                external_replay_loss_sum = 0.0
                task_sample_count = 0
                replay_sample_count = 0
                external_replay_sample_count = 0

            progress_bar.update(1)
            global_step += 1
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logging.info(logs)
            progress_bar.set_postfix(**logs)
            
            if global_step >= start_global_step + max_train_steps:
                break
                
        # Break out of the epoch loop if we've reached max_train_steps
        if global_step >= start_global_step + max_train_steps:
            break

    # Return the final step count for the next dataset
    return global_step

def main():
    args = parse_args()
    
    # Initialize DeepSpeed before anything else
    ds.init_distributed()
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        
        # Save hyperparameters as JSON for easy reference
        with open(os.path.join(timestamped_output_dir, "hyperparameters.json"), "w") as f:
            json.dump(hyperparams, f, indent=4, sort_keys=True)
    
    # Set up wandb if needed
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        if ds.comm.get_rank() == 0:  # Only initialize wandb on the main process
            wandb.init(project="pixart-dreambooth-continual", name=f"run_{timestamp}")
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
        logging.info("Pre-computing text embeddings for all datasets")

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )

            return prompt_embeds, text_inputs

        validation_prompt_negative_prompt_embeds, validation_prompt_negative_text_inputs = compute_text_embeddings("")

        # Pre-compute embeddings for all instance prompts
        pre_computed_encoder_hidden_states_list = []
        pre_computed_encoder_text_inputs_list = []
        
        for instance_prompt in args.instance_prompts:
            hidden_states, text_inputs = compute_text_embeddings(instance_prompt)
            pre_computed_encoder_hidden_states_list.append(hidden_states)
            pre_computed_encoder_text_inputs_list.append(text_inputs)

        # Pre-compute embeddings for all class prompts if using prior preservation
        pre_computed_class_prompt_encoder_hidden_states_list = []
        pre_computed_class_prompt_encoder_text_inputs_list = []
        
        if args.with_prior_preservation:
            for class_prompt in args.class_prompts:
                hidden_states, text_inputs = compute_text_embeddings(class_prompt)
                pre_computed_class_prompt_encoder_hidden_states_list.append(hidden_states)
                pre_computed_class_prompt_encoder_text_inputs_list.append(text_inputs)
        else:
            pre_computed_class_prompt_encoder_hidden_states_list = [None] * len(args.instance_prompts)
            pre_computed_class_prompt_encoder_text_inputs_list = [None] * len(args.instance_prompts)

        # Free memory after pre-computing embeddings
        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states_list = [None] * len(args.instance_prompts)
        pre_computed_encoder_text_inputs_list = [None] * len(args.instance_prompts)
        pre_computed_class_prompt_encoder_hidden_states_list = [None] * len(args.instance_prompts)
        pre_computed_class_prompt_encoder_text_inputs_list = [None] * len(args.instance_prompts)
        validation_prompt_negative_prompt_embeds = None
        validation_prompt_negative_text_inputs = None

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

    # Initialize DeepSpeed with the transformer model before using model_engine
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Initialize DeepSpeed with a dummy scheduler that will be replaced
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=1000,  # Will be replaced for each dataset
    )
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = ds.initialize(
        model=transformer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_parameters=params_to_optimize,
        config=args.deepspeed,
    )

    # Store previous task samples for replay
    replay_image_paths = []
    replay_prompts = []
    
    # Initialize external replay dataset if specified
    external_replay_dataset = None
    if args.external_replay_data:
        if ds.comm.get_rank() == 0:
            logger.info(f"Loading external replay data from {args.external_replay_data}")
        
        # Initialize external dataset
        external_replay_dataset = ExternalReplayData(
            root=args.external_replay_data,
            image_list_json=args.external_replay_json,
            resolution=args.resolution,
            center_crop=args.center_crop,
            max_length=args.tokenizer_max_length,
            tokenizer=tokenizer if not args.pre_compute_text_embeddings else None,
            text_encoder=text_encoder if not args.pre_compute_text_embeddings else None,
            pre_compute_text_embeddings=args.pre_compute_text_embeddings,
        )
        
        if ds.comm.get_rank() == 0:
            logger.info(f"Loaded {len(external_replay_dataset)} external replay samples")
    
    # Continual learning: Loop through each dataset
    global_step = 0
    for dataset_idx, (instance_data_dir, instance_prompt) in enumerate(zip(args.instance_data_dirs, args.instance_prompts)):
        if ds.comm.get_rank() == 0:
            logger.info(f"==========================================================")
            logger.info(f"Starting training on dataset {dataset_idx+1}/{len(args.instance_data_dirs)}")
            logger.info(f"Data directory: {instance_data_dir}")
            logger.info(f"Prompt: {instance_prompt}")
            if dataset_idx > 0 and replay_image_paths:
                logger.info(f"Using task replay with {len(replay_image_paths)} images from previous tasks")
            logger.info(f"==========================================================")
        
        # For datasets after the first one, we need to reinitialize the optimizer and scheduler
        if dataset_idx > 0:
            # Get parameters to optimize for the current dataset
            params_to_optimize = list(filter(lambda p: p.requires_grad, model_engine.module.parameters()))
            optimizer_cls = torch.optim.AdamW
            optimizer = optimizer_cls(
                params_to_optimize,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

            # Use max_train_steps directly for scheduler
            total_steps = args.max_train_steps * ds.comm.get_world_size()
            
            # Create a learning rate scheduler specific to this dataset
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * ds.comm.get_world_size(),
                num_training_steps=total_steps,
            )
        
            model_engine, optimizer, _, lr_scheduler = ds.initialize(
                model=model_engine.module,  # Use the underlying model
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                model_parameters=params_to_optimize,
                config=args.deepspeed,
            )
        
        # Get corresponding class data if using prior preservation
        class_data_dir = args.class_data_dirs[dataset_idx] if args.with_prior_preservation else None
        class_prompt = args.class_prompts[dataset_idx] if args.with_prior_preservation else None
        
        # Train on current dataset
        pre_computed_instance_embeddings = pre_computed_encoder_hidden_states_list[dataset_idx] if args.pre_compute_text_embeddings else None
        pre_computed_instance_inputs = pre_computed_encoder_text_inputs_list[dataset_idx] if args.pre_compute_text_embeddings else None
        pre_computed_class_embeddings = pre_computed_class_prompt_encoder_hidden_states_list[dataset_idx] if args.pre_compute_text_embeddings else None
        pre_computed_class_inputs = pre_computed_class_prompt_encoder_text_inputs_list[dataset_idx] if args.pre_compute_text_embeddings else None
        
        # Create replay options only for tasks after the first one
        replay_options = {}
        if dataset_idx > 0 and replay_image_paths:
            replay_options = {
                "replay_paths": replay_image_paths,
                "replay_prompts": replay_prompts
            }
            if ds.comm.get_rank() == 0:
                logger.info(f"Using {len(replay_image_paths)} replay samples from previous tasks")
        
        # Add external replay dataset to options
        if external_replay_dataset is not None:
            replay_options["external_replay_dataset"] = external_replay_dataset
        
        global_step = train_on_dataset(
            model_engine=model_engine,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            instance_data_dir=instance_data_dir,
            instance_prompt=instance_prompt,
            class_data_dir=class_data_dir,
            class_prompt=class_prompt,
            args=args,
            weight_dtype=weight_dtype,
            device=device,
            timestamped_output_dir=timestamped_output_dir,
            start_global_step=global_step,
            pre_computed_encoder_hidden_states=pre_computed_instance_embeddings,
            pre_computed_encoder_text_inputs=pre_computed_instance_inputs,
            pre_computed_class_prompt_encoder_hidden_states=pre_computed_class_embeddings,
            pre_computed_class_prompt_encoder_text_inputs=pre_computed_class_inputs,
            **replay_options  # Pass replay options to train_on_dataset
        )
        
        # After training on a dataset, collect replay samples
        # Sample random images from the current task to use for replay in future tasks
        if ds.comm.get_rank() == 0:
            # Get list of all images in the current task
            current_task_images = list(Path(instance_data_dir).iterdir())
            
            # Sample a few images (at least 1, up to 5 or 10% of the dataset)
            num_samples = max(1, min(5, int(len(current_task_images) * 0.1)))
            sampled_images = random.sample(current_task_images, num_samples)
            
            # Add to replay collection with corresponding prompt
            replay_image_paths.extend([str(img) for img in sampled_images])
            replay_prompts.extend([instance_prompt] * len(sampled_images))
            
            logger.info(f"Added {len(sampled_images)} images from task {dataset_idx} to replay buffer.")
            logger.info(f"Replay buffer now contains {len(replay_image_paths)} images.")
        
        # Save checkpoint after this dataset
        if ds.comm.get_rank() == 0:
            dataset_name = os.path.basename(instance_data_dir)
            save_path = os.path.join(timestamped_output_dir, f"{dataset_idx}")
            os.makedirs(save_path, exist_ok=True)
            
            # Extract and save the model
            if hasattr(model_engine, "module"):
                transformer = model_engine.module
            else:
                transformer = model_engine
                
            transformer.save_pretrained(os.path.join(save_path, "transformer"))
            logger.info(f"Saved model checkpoint after dataset {dataset_idx} to {save_path}")

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        
        # Wait for a bit to ensure memory is freed up
        time.sleep(1)
        
        # Now it's safer to call barrier
        ds.comm.barrier()
    
    # End of training
    logger.info("Continual learning completed successfully!")

if __name__ == "__main__":
    main()
