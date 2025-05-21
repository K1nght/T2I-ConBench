from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Calculate FID score for generated images')
    parser.add_argument('--generations_path', type=str, required=True, help='Path to generated images')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save FID score')
    
    args = parser.parse_args()
    
    fid, _ = calculate_fid(
        args.generations_path,
        get_coco_fid_stats()
    )
    
    # Append result to txt file
    with open(args.output_file, 'a') as f:
        f.write(f"{args.generations_path}: {float(fid):.6f}\n")

if __name__ == '__main__':
    main()