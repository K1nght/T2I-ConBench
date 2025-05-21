import hpsv2
import os
import argparse
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Calculate HPS-v2 scores for images')
parser.add_argument('--images_folder', type=str, required=True,
                    help='Path to the folder containing images')
parser.add_argument('--prompt_path', type=str, required=True,
                    help='Path to the prompt file')
parser.add_argument('--save_to', type=str, required=True,
                    help='Path to save the summary results')
parser.add_argument('--detailed_save_to', type=str, required=True,
                    help='Path to save the detailed scores')

args = parser.parse_args()

images_folder = args.images_folder
prompt_path = args.prompt_path
save_to = args.save_to
detailed_save_to = args.detailed_save_to

# 获取所有图片文件并按index排序
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

results = []
# 读取所有prompts
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompts = f.readlines()

# 打开详细分数文件
with open(detailed_save_to, 'w', encoding='utf-8') as f_detail:
    f_detail.write("Index\tImage\tScore\n")  # 写入表头
    
    pbar = tqdm(image_files, desc="Processing images")
    for image_file in pbar:
        partial_p= image_file.split('.')[0].split('_')[2]
        print(partial_p)
        image_path = os.path.join(images_folder, image_file)

        prompt = None 
        for p in prompts:
            if p.startswith(partial_p):
                prompt = p 
                break 
        if prompt is None:
            print(f"{image_path} not find {partial_p}")
            continue
        
        try:    
            result = hpsv2.score(image_path, prompt.strip(), hps_version="v2.1")
            score = float(result[0])
            results.append(score)
            
            # 更新进度条显示
            pbar.set_postfix_str(f"Image: {image_file}, Score: {score:.4f}")
            
            # 写入每张图片的得分（制表符分隔）
            f_detail.write(f"{prompt.strip()}\t{image_file}\t{score:.4f}\n")
                
        except Exception as e:
            error_msg = f"Error processing image {image_file}: {e}"
            tqdm.write(error_msg)
            f_detail.write(f"{prompt.strip()}\t{image_file}\tError: {e}\n")
            continue

# 计算平均分数
mean_hpsv2 = sum(results)/len(results)
print(f"\nMean HPS-v2 Score: {mean_hpsv2:.4f}")

# 保存汇总结果到原始save_to文件
with open(save_to, 'a', encoding='utf-8') as f:
    f.write(f"Images Folder: {images_folder}\n")
    f.write(f"Prompt Path: {prompt_path}\n")
    f.write(f"Mean HPS-v2 Score: {mean_hpsv2:.4f}\n")
    f.write(f"Detailed scores saved to: {detailed_save_to}\n")