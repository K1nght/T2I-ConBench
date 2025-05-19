from datasets import load_dataset
import os
import json
import shutil
import random
from PIL import Image
import io
from tqdm import tqdm

# 第一步：从数据集加载并保存指定文件夹的图片
print("开始从数据集加载图片...")
dataset = load_dataset("google/dreambooth")

# 指定要提取的文件夹
target_folders = ['dog', 'dog3', 'cat2', 'shiny_sneaker']

# 指定保存路径
save_dir = '/opt/data/private/hzhcode/T2I-ConBench-data'

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 遍历数据集并保存指定文件夹中的图片
for item in target_folders:
    dataset = load_dataset("google/dreambooth", name=item)
    item_save_dir = os.path.join(save_dir, "item", item)
    os.makedirs(item_save_dir, exist_ok=True)
    for split in dataset.keys():
        for idx, item in tqdm(enumerate(dataset[split]), desc=f'Processing {split}'):
            # 获取图片文件名
            filename = f"{idx:08d}.png"
            # 获取图片数据
            image = item['image']
            # 构建保存路径
            save_path = os.path.join(item_save_dir, filename)
            # 确保目标文件夹存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 保存图片
            image.save(save_path)

print("数据集图片加载完成！")

# 第二步：处理保存的图片并创建dreambooth和replay数据
print("开始处理图片并创建dreambooth数据...")

# 源文件夹和目标文件夹
prompts = {
    'dog': 'A photo of p0h1 dog.',
    'dog3': 'A photo of k5f2 dog.',
    'cat2': 'A photo of s5g3 cat.',
    'shiny_sneaker': 'A photo of b9l1 sneaker.'
}

# 创建目标文件夹
dreambooth_dir = os.path.join(save_dir, "domain", "dreambooth", "Img")
os.makedirs(dreambooth_dir, exist_ok=True)

# 处理所有图片并创建dreambooth.json
dreambooth_data = []
dreambooth_replay_data = []
global_idx = 0  # 使用全局idx计数器

for item in tqdm(target_folders, desc='Processing folders'):
    source_path = os.path.join(save_dir, "item", item)
    if not os.path.exists(source_path):
        continue
        
    # 获取文件夹中的所有图片
    images = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 随机选择一张图片用于replay
    replay_image = random.choice(images)
    
    # 复制所有图片到dreambooth文件夹
    for img in images:
        new_name = f"{global_idx:08d}.png"
        shutil.copy2(
            os.path.join(source_path, img),
            os.path.join(dreambooth_dir, new_name)
        )
        # 添加到dreambooth数据列表
        dreambooth_data.append({
            "path": os.path.join(dreambooth_dir, new_name),
            "prompt": prompts[item],
            "idx": global_idx
        })
        
        # 如果是选中的replay图片，添加到replay数据列表
        if img == replay_image:
            dreambooth_replay_data.append({
                "path": os.path.join(dreambooth_dir, new_name),
                "prompt": prompts[item],
                "idx": global_idx
            })
            
        global_idx += 1

# 保存dreambooth.json
data_info_save_dir = os.path.join(save_dir, "domain", "data_info")
os.makedirs(data_info_save_dir, exist_ok=True)
with open(os.path.join(data_info_save_dir, 'dreambooth.json'), 'w') as f:
    json.dump(dreambooth_data, f, indent=2)

# 保存dreambooth_replay.json
with open(os.path.join(data_info_save_dir, 'dreambooth_replay.json'), 'w') as f:
    json.dump(dreambooth_replay_data, f, indent=2)

print("处理完成！")
print(f"所有图片已保存到 {dreambooth_dir} 文件夹")
print("已创建 dreambooth.json 和 dreambooth_replay.json 文件")

