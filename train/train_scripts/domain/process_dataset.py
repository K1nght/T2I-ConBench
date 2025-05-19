from datasets import load_dataset
import os
import json
from PIL import Image
import io
from tqdm import tqdm

dataset = load_dataset(
    "T2I-ConBench/T2I-ConBench", 
    revision="refs/convert/parquet",
)

save_dir = [
    ("body", 2358),
    ("body_replay", 235),
    ("coco_500_replay", 500),
    ("cross", 1821),
    ("cross_replay", 182),
    ("nature", 2512),
    ("nature_replay", 251),
]

# 创建基础保存目录
base_save_dir = "/opt/data/private/hzhcode/T2I-ConBench-data/domain"
os.makedirs(base_save_dir, exist_ok=True)

# 计算总样本数
total_samples = sum(num for _, num in save_dir)
print(f"Total samples to process: {total_samples}")

# 创建总进度条
pbar = tqdm(total=total_samples, desc="Total Progress")

# 处理每个domain的数据
current_idx = 0

for domain, num_samples in save_dir:
    print(f"\nProcessing domain: {domain}")
    
    # 创建domain目录和图片子目录
    domain_dir = os.path.join(base_save_dir, domain)
    img_dir = os.path.join(domain_dir, "Img")
    os.makedirs(img_dir, exist_ok=True)
    
    # 准备存储prompts的列表
    prompts_list = []
    
    # 创建domain级别的进度条
    domain_pbar = tqdm(total=num_samples, desc=f"{domain}", leave=False)
    
    # 保存指定数量的样本
    for i in range(num_samples):
        sample = dataset["train"][current_idx + i]
        
        # 保存图片
        image_data = sample["image"]
        if isinstance(image_data, dict):  # 如果image是字典格式（包含bytes）
            image = Image.open(io.BytesIO(image_data["bytes"]))
        elif isinstance(image_data, (bytes, bytearray)):  # 如果是原始字节数据
            image = Image.open(io.BytesIO(image_data))
        else:  # 如果已经是PIL Image对象
            image = image_data
            
        image_path = os.path.join(img_dir, f"{i:08d}.png")
        image.save(image_path)
        
        # 存储prompt信息
        prompt_info = {
            "image_path": os.path.join("/opt/data/private/hzhcode/PixArt-alpha/data", domain, "Img", f"{i:08d}.png"),
            "prompt": sample["prompt"],
            "idx": i
        }
        prompts_list.append(prompt_info)
        
        # 更新进度条
        pbar.update(1)
        domain_pbar.update(1)
    
    # 关闭domain进度条
    domain_pbar.close()
    
    # 保存prompts到json文件
    json_path = os.path.join(domain_dir, f"{domain}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_list, f, ensure_ascii=False, indent=2)
    
    # 更新当前索引
    current_idx += num_samples
    print(f"Completed {domain}: {num_samples} samples")

# 关闭总进度条
pbar.close()

print("\nAll processing completed!")
print(f"Total processed samples: {total_samples}")
