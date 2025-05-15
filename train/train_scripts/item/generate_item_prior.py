import os
import time
import argparse
import torch
from datetime import datetime
from tqdm import tqdm

from diffusers import PixArtAlphaPipeline, Transformer2DModel


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
        "--validation_prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="用于生成图像的提示词"
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
        "--num_prior",
        type=int,
        default=1,
        help="为每个提示词生成的图像数量"
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

    # 使用单个提示词
    prompt = args.validation_prompt
    print(f"使用提示词: '{prompt}'")

    # 运行推理
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    images = []
    
    # 为单个提示词生成num_prior张图片
    for j in tqdm(range(args.num_prior), desc="生成图片", total=args.num_prior):
        # 记录开始时间
        start_time = time.time()
        
        # 生成图片
        pipeline_args = {"prompt": prompt}
        image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
        images.append(image)
        
        # 计算生成时间
        generation_time = time.time() - start_time
        print(f"图片 {j+1}/{args.num_prior} 生成完成，提示词: '{prompt}'，耗时: {generation_time:.2f}秒")
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用提示词的前50个字符作为文件名（避免文件名过长）
        filename = f"{j}_{prompt[:50]}_{timestamp}.png"
        image_path = os.path.join(args.output_dir, filename)
        image.save(image_path)
        print(f"图片已保存到: {image_path}")

if __name__ == "__main__":
    main()