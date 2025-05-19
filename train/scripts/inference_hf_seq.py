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
        help="包含多个transformer模型checkpoint的父目录，子目录应命名为0,1,2等"
    )
    parser.add_argument(
        "--prompts_files",
        nargs='+',
        type=str,
        default=None,
        help="包含多个提示词的文本文件路径列表，每个文件每行一个提示词"
    )
    parser.add_argument(
        "--validation_length",
        type=int,
        default=1e6,
        help="prompt文件取用条数"
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

    # 检查并创建输出目录
    if not hasattr(args, 'output_dir'):
        args.output_dir = 'outputs'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"将图片保存到目录: {args.output_dir}")

    # 检查transformer子目录是否存在
    transformer_dirs = []
    i = 0
    while os.path.exists(os.path.join(args.transformer_path, str(i), "transformer")):
        transformer_dirs.append(os.path.join(args.transformer_path, str(i), "transformer"))
        i += 1
    
    if not transformer_dirs:
        raise ValueError(f"在 {args.transformer_path} 中没有找到以数字命名的子目录")
    
    print(f"发现 {len(transformer_dirs)} 个transformer checkpoint目录")

    # 检查提示词文件
    prompts_list = []
    prompts_filenames = []
    for file_path in args.prompts_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
            prompts = prompts[:args.validation_length]
            prompts_list.append(prompts)
            # 从文件路径中提取文件名（不包含路径和扩展名）
            filename = os.path.splitext(os.path.basename(file_path))[0]
            prompts_filenames.append(filename)
            print(f"从文件 {file_path} 读取了 {len(prompts)} 个提示词")
        else:
            print(f"警告: 提示词文件 {file_path} 不存在")
            prompts_list.append([])
            prompts_filenames.append(f"invalid_file_{len(prompts_filenames)}")
    
    # 初始化pipeline
    pipeline = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,  # 暂时不加载transformer，稍后会根据目录加载
        torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(device)
    
    # 设置随机种子生成器
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    
    # 对每个transformer目录进行处理
    for idx, transformer_dir in enumerate(transformer_dirs):
        print(f"\n加载来自 {transformer_dir} 的transformer模型...")
        
        # 加载当前的transformer模型
        transformer = Transformer2DModel.from_pretrained(transformer_dir, torch_dtype=weight_dtype)
        transformer.to(device)
        
        # 更新pipeline的transformer
        pipeline.transformer = transformer
        
        # 为每个处理的prompts文件单独处理
        for i in range(idx + 1):
            if i < len(prompts_list) and prompts_list[i]:
                # 获取当前处理的prompts
                current_prompts = prompts_list[i]
                current_filename = prompts_filenames[i]
                
                # 创建特定transformer和prompts文件的输出目录
                output_subdir = os.path.join(args.output_dir, f"{idx}_{current_filename}")
                os.makedirs(output_subdir, exist_ok=True)
                
                print(f"使用transformer {idx} 处理prompts文件 '{current_filename}' 中的 {len(current_prompts)} 个提示词")
                print(f"每个提示词将生成 {args.num_validation_images} 张图片")
                
                # 运行推理
                for j, prompt in tqdm(enumerate(current_prompts), total=len(current_prompts), 
                                      desc=f"Transformer {idx} - {current_filename} 生成图片"):
                    # 为每个prompt创建一个子目录
                    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:100])
                    
                    # 为每个prompt生成num_validation_images张图片
                    for img_idx in range(args.num_validation_images):
                        # 记录开始时间
                        start_time = time.time()
                        
                        # 生成图片
                        pipeline_args = {"prompt": prompt}
                        image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
                        
                        # 计算生成时间
                        generation_time = time.time() - start_time
                        print(f"Prompt {j+1}/{len(current_prompts)}, 图片 {img_idx+1}/{args.num_validation_images} 生成完成，"
                              f"提示词: '{prompt}'，耗时: {generation_time:.2f}秒")
                        
                        # 保存图片
                        # 在文件名中包含图片索引
                        filename = f"{j}_{img_idx}_{safe_prompt}_seed{args.seed}.png"
                        image_path = os.path.join(output_subdir, filename)
                        image.save(image_path)
                        print(f"图片已保存到: {image_path}")

if __name__ == "__main__":
    main()