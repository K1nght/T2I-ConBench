import os
import time
import argparse
import torch
from datetime import datetime
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="运行 PixArtAlpha 推理")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="预训练模型的路径或Hugging Face模型ID"
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        required=True,
        help="transformer模型路径"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="包含多个提示词的文本文件路径，每行一个提示词"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="保存生成图像的目录"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于可复现的结果"
    )
    parser.add_argument(
        "--validation_length",
        type=int,
        default=1e6,
        help="prompt文件取用条数"
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="要生成的图像数量"
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help="用于验证的输入图像（可选）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行推理的设备 (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="是否使用半精度推理"
    )
    return parser.parse_args()

# 主程序
def main():
    args = parse_args()
    
    # 设置设备和权重类型
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

    # 检查并创建输出目录
    if not hasattr(args, 'output_dir'):
        args.output_dir = 'outputs'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"将图片保存到目录: {args.output_dir}")

    # 读取提示词
    prompts = []
    if args.prompts_file is not None and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        prompts = prompts[:args.validation_length]
        print(f"从文件 {args.prompts_file} 读取了 {len(prompts)} 个提示词")
    else:
        raise ValueError(f"文件 {args.prompts_file} 不存在")

    # 运行推理
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    images = []
    
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="生成图片"):
        # 记录开始时间
        start_time = time.time()
        
        # 为每个prompt生成num_validation_images张图片
        for img_idx in range(args.num_validation_images):
            # 生成图片
            pipeline_args = {"prompt": prompt}
            image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
            images.append(image)
            
            # 计算生成时间
            generation_time = time.time() - start_time
            print(f"图片 {i+1}/{len(prompts)} 生成完成，提示词: '{prompt}'，耗时: {generation_time:.2f}秒")
            
            # 保存图片
            # 使用提示词的前100个字符作为文件名（避免文件名过长）
            filename = f"{i}_{img_idx}_{prompt[:100]}_seed{args.seed}.png"
            image_path = os.path.join(args.output_dir, filename)
            image.save(image_path)
            print(f"图片已保存到: {image_path}")

if __name__ == "__main__":
    main()