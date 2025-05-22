import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import json
import os
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="Answer questions based on generated images")
    parser.add_argument("--image_dir", type=str, default="/eva/ip/inference2", help="Directory containing test/generated images")
    parser.add_argument("--ref_image_dir", type=str, default="/eva/ip/raw/datasets", help="Directory containing all reference images (for p0h1_dog, k5f2_dog, s5g3_cat, b9l1_sneaker)")
    parser.add_argument("--ref_p0h1_dog", type=str, default=None, help="Reference image path for p0h1_dog (relative to ref_image_dir)")
    parser.add_argument("--ref_k5f2_dog", type=str, default=None, help="Reference image path for k5f2_dog (relative to ref_image_dir)")
    parser.add_argument("--ref_s5g3_cat", type=str, default=None, help="Reference image path for s5g3_cat (relative to ref_image_dir)")
    parser.add_argument("--ref_b9l1_sneaker", type=str, default=None, help="Reference image path for b9l1_sneaker (relative to ref_image_dir)")
    parser.add_argument("--model_name", type=str, default="/Qwen/Qwen2.5-VL-7B-Instruct", help="Name of the VLM model to use")
    parser.add_argument("--output_file", type=str, default="/eva/ip/inference2/results.json", help="Path to save the output results")
    parser.add_argument("--txt_output", type=str, default="/eva/ip/inference2/results.txt", help="Path to save the output results")
    args = parser.parse_args()
    # Set default reference image paths if not provided, using ref_image_dir
    if args.ref_p0h1_dog is None:
        args.ref_p0h1_dog = os.path.join(args.ref_image_dir, "dog/00.jpg")
    if args.ref_k5f2_dog is None:
        args.ref_k5f2_dog = os.path.join(args.ref_image_dir, "dog3/00.jpg")
    if args.ref_s5g3_cat is None:
        args.ref_s5g3_cat = os.path.join(args.ref_image_dir, "cat2/00.jpg")
    if args.ref_b9l1_sneaker is None:
        args.ref_b9l1_sneaker = os.path.join(args.ref_image_dir, "shiny_sneaker/00.jpg")
    return args

def extract_animals(args):
    # Dynamically parse folder names
    animal_keywords = {}
    for folder in os.listdir(args.image_dir):
        folder_path = os.path.join(args.image_dir, folder)
        if os.path.isdir(folder_path):
            parts = folder.split('_')
            if len(parts) < 4:
                continue
            # Format: train_round_special_mark_normal_category
            train_round = parts[0]
            special_mark = parts[1]
            normal_category = parts[2]
            special_category = f"{special_mark}_{normal_category}"
            
            if normal_category not in animal_keywords:
                animal_keywords[normal_category] = []
            if special_category not in animal_keywords:
                animal_keywords[special_category] = []
            animal_keywords[normal_category].append(folder)
            animal_keywords[special_category].append(folder)
    return animal_keywords

def check_existence(file_path, keyword):
    # Check if the keyword exists in the file name
    return keyword in os.path.basename(file_path).lower()

def qwen2_5_VL_7B_answer(model, processor, image_path, question, reference_image_path, args):
    # print(image_path)
    # print(reference_image_path)

    def load_and_resize_image(path):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        new_width = width // 2
        new_height = height // 2
        resized_images = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_images

    reference_image = load_and_resize_image(reference_image_path)
    input_image = load_and_resize_image(image_path)

    messages = [
        {
            "role": "User",
            "content": [
                {
                    "type": "image",
                    "image": reference_image,
                },
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) : ] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    answer = output_text[0].split("%")[0].strip()
    answer = float(answer)
    # print(output_text)
    # print(answer)

    reference_image.close()
    input_image.close()

    return answer

def answer_questions(model, processor, args):
    count = 0
    total_count = 0
    results = {}
    folder_path = args.image_dir

    # Mapping from training round to normal categories
    round_to_normal_categories = {
        "0": ["p0h1_dog"],
        "1": ["p0h1_dog", "k5f2_dog"],
        "2": ["p0h1_dog", "k5f2_dog", "s5g3_cat"],
        "3": ["p0h1_dog", "k5f2_dog", "s5g3_cat", "b9l1_sneaker"]
    }

    # Mapping of questions and reference images
    question_map = {
        "p0h1_dog": {
            "question": "What is the probability that the second image has the same dog as the first image?Please just give the probability.",
            "reference_image": args.ref_p0h1_dog
        },
        "k5f2_dog": {
            "question": "What is the probability that the second image has the same dog as the first image?Please just give the probability.",
            "reference_image": args.ref_k5f2_dog
        },
        "s5g3_cat": {
            "question": "What is the probability that the second image has the same cat as the first image?Please just give the probability.",
            "reference_image": args.ref_s5g3_cat
        },
        "b9l1_sneaker": {
            "question": "What is the probability that the second image has the same shiny_sneaker as the first image?Please just give the probability.",
            "reference_image": args.ref_b9l1_sneaker
        }
    }

    # Traverse all subfolders
    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        if not os.path.isdir(folder_path_full):
            continue

        # Parse folder name
        parts = folder.split('_')
        if len(parts) < 3:
            continue
        train_round = parts[0]
        special_mark = parts[1]
        normal_category = parts[2]
        special_category = f"{special_mark}_{normal_category}"

        # Get the normal categories to be queried in the current training round
        if train_round not in round_to_normal_categories:
            continue
        normal_categories = round_to_normal_categories[train_round]

        folder_results = {
            "normal_scores":[],
            "special_scores":[]
        }

        # Traverse all images in the folder
        for file in os.listdir(folder_path_full):
            file_path = os.path.join(folder_path_full, file)
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_results = {
                "file_name": file,
                "normal_scores": {},
                "special_score": None
            }

            # Determine if it is a normal or special category
            if check_existence(file_path, normal_category) and not check_existence(file_path, special_mark):
                # Normal category, query all normal categories for this round
                for category in normal_categories:
                    if category in question_map:
                        question = question_map[category]["question"]
                        reference_image = question_map[category]["reference_image"]
                        correct_answer = 1 if check_existence(file_path, category) else 0

                        # Answer the question
                        model_answer = qwen2_5_VL_7B_answer(model, processor, file_path, question, reference_image, args)
                        score = 1 - abs(model_answer - correct_answer)

                        image_results["normal_scores"][category] = {
                            "Question": question,
                            "Correct Answer": correct_answer,
                            "Generated Answer": model_answer,
                            "Image Path": file_path,
                            "reference_image": reference_image,
                            "Category": category,
                            "Score" : score
                        }

                        # Print result
                        print(f"file: {file_path}")
                        print(f"question: {question}")
                        print(f"model_answer: {model_answer}")
                        print(f"correct_answer: {correct_answer}")
                        print(f"Category: {category}")
                        print(f"score : {score}")
                        print("-" * 30)

            else:
                # Special category, query if it is a special dog
                if special_category in question_map:
                    question = question_map[special_category]["question"]
                    reference_image = question_map[special_category]["reference_image"]
                    correct_answer = 1 if check_existence(file_path, special_mark) else 0

                    # Answer the question
                    model_answer = qwen2_5_VL_7B_answer(model, processor, file_path, question, reference_image, args)
                    score = 1 - abs(model_answer - correct_answer)

                    # Record result
                    image_results["special_score"]={
                        "Question": question,
                        "Correct Answer": correct_answer,
                        "Generated Answer": model_answer,
                        "Image Path": file_path,
                        "reference_image": reference_image,
                        "Category": special_category,
                        "Score" : score
                    }

                    # Print result
                    print(f"file: {file_path}")
                    print(f"question: {question}")
                    print(f"model_answer: {model_answer}")
                    print(f"correct_answer: {correct_answer}")
                    print(f"score : {score}")
                    print("-" * 30)

                    # Count correct answers
                    count += 1 - abs(model_answer - correct_answer)
                    total_count += 1

            if image_results["normal_scores"]:        
                folder_results["normal_scores"].append(image_results["normal_scores"])
            if image_results["special_score"]:
                folder_results["special_scores"].append(image_results["special_score"])

        # Record subfolder results
        results[folder] = folder_results

    # Calculate the average score for each normal and special category in each folder
    normal_categories = ["p0h1_dog", "k5f2_dog", "s5g3_cat", "b9l1_sneaker"]
    for folder, data in results.items():
        normal_scores = data["normal_scores"]
        if normal_scores:
            # Average score for each normal category
            normal_category_averages = {}
            for category in normal_categories:
                scores = []
                for image_scores in normal_scores:
                    if category in image_scores:
                        scores.append(image_scores[category]["Score"])
                if scores:
                    avg = sum(scores) / len(scores)
                    normal_category_averages[category] = avg
            data["normal_category_averages"] = normal_category_averages

            # Overall average score for normal categories
            all_normal_scores = []
            for image_scores in normal_scores:
                for category in image_scores:
                    all_normal_scores.append(image_scores[category]["Score"])
            if all_normal_scores:
                avg_normal_score = sum(all_normal_scores) / len(all_normal_scores)
                data["avg_normal_score"] = avg_normal_score

        special_scores = data["special_scores"]
        if special_scores:
            # Average score for special categories
            all_special_scores = [score["Score"] for score in special_scores]
            if all_special_scores:
                avg_special_score = sum(all_special_scores) / len(all_special_scores)
                data["avg_special_score"] = avg_special_score

    # Save results to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Save to {args.output_file} file")

    # Save results to TXT file
    with open(args.txt_output, 'w') as f:
        # Output details for each folder
        for folder, data in results.items():
            f.write(f"Folder: {folder}\n")
            # Output average score for each normal category
            if "normal_category_averages" in data:
                f.write("Normal Category Averages:\n")
                for category, avg in data["normal_category_averages"].items():
                    f.write(f"  {category} Average: {avg:.2f}\n")
            
            # Output overall average score for normal categories
            if "avg_normal_score" in data:
                f.write(f"Average Normal Score: {data['avg_normal_score']:.2f}\n")
            
            # Output average score for special categories
            if "avg_special_score" in data:
                f.write(f"Average Special Score: {data['avg_special_score']:.2f}\n")
            
            f.write("-" * 50 + "\n")

    print(f"Save to {args.txt_output} file")

def main():
    args = get_args()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_name)
    answer_questions(model, processor, args)

if __name__ == "__main__":
    main()
