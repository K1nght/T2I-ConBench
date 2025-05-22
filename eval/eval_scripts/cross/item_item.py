# coding:utf-8  
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import json
import os
from PIL import Image
import glob
import re

def get_args():
    parser = argparse.ArgumentParser(description="Answer questions based on generated images")
    parser.add_argument("--image_dir", type=str, default="/eva/cross/cross_new_test_v2", help="Directory containing generated/test images")
    parser.add_argument("--ref_image_dir", type=str, default="/eva/cross_final/raw/datasets", help="Directory containing reference images")
    parser.add_argument("--qa_file", type=str, default="/eva/cross/cross_new_test_qa.txt", help="Path to the QA data file")
    parser.add_argument("--model_name", type=str, default="/Qwen/Qwen2.5-VL-7B-Instruct", help="Name of the VLM model to use")
    parser.add_argument("--output_file", type=str, default="/eva/results2.json", help="Path to save the output results")
    parser.add_argument("--txt_output", type=str, default="/results/eva/results2.json", help="Path to save the output results")
    args = parser.parse_args()
    return args

# Extract special animals from each prompt

def extract_animals(text, animal_list):
    """
    Extract special animal names from a single prompt
    :param text: input text
    :param animal_list: list of special animal names
    :return: list of extracted special animal names
    """
    pattern = r'\b(' + '|'.join(re.escape(animal) for animal in animal_list) + r')\b'
    matches = re.findall(pattern, text, re.IGNORECASE)  # Ignore case when matching
    return list(set(matches))  # Remove duplicates, keep original case


def is_special_animal_question(question, animal_list):
    """
    Determine whether the question is related to special animals
    :param question: input question
    :param animal_list: list of special animal names
    :return: whether it is a special animal related question (bool)
    """
    pattern = r'\b(' + '|'.join(re.escape(animal) for animal in animal_list) + r')\b'
    return bool(re.search(pattern, question, re.IGNORECASE))


def load_and_resize_image(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    new_width = width // 2
    new_height = height // 2
    resized_images = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_images


def main():
    args = get_args()
    print(f"Reference image directory: {args.ref_image_dir}")
    print(f"Test/generated image directory: {args.image_dir}")

    # Build animal_images using the reference directory
    animal_images = {
        "s5g3 cat": os.path.join(args.ref_image_dir, "cat2/00.jpg"),
        "k5f2 dog": os.path.join(args.ref_image_dir, "dog3/03.jpg"),
        "b9l1 sneaker": os.path.join(args.ref_image_dir, "shiny_sneaker/00.jpg"),
        "p0h1 dog": os.path.join(args.ref_image_dir, "dog/00.jpg")
    }

    with open(args.qa_file, 'r') as f:
        qa_pairs = f.read().splitlines()
    
    results = []
    correct_count = 0

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    for i in range(0, len(qa_pairs), 2):
        print(f"Processing question pair {i // 2 + 1}...")
        prompt_line = qa_pairs[i]
        qa_line = qa_pairs[i+1]

        prompt = prompt_line.split(': ')[1]
        qa_data = json.loads(qa_line)
        question = qa_data['question']
        correct_answer = qa_data['Correct Answer']

        # Search for the corresponding generated image path
        search_path = os.path.join(args.image_dir, f"*{prompt[:100]}*.png")
        image_paths = glob.glob(search_path)
        if not image_paths:
            print(f"No matching image found for prompt: {prompt}")
            continue
        image_path = image_paths[0]
        print(f"Matched generated image: {image_path}")

        # Determine if it is a special animal question
        if is_special_animal_question(question, animal_images.keys()):
            # Extract special animals from question
            animals = extract_animals(question, animal_images.keys())
            if not animals:
                print(f"No special animal found in question: {question}")
                continue
            animal = animals[0]
            animal_image = animal_images[animal]
            reference_image = load_and_resize_image(animal_image)
            print(f"Identified special animal: {animal}, Image path: {animal_image}")
            print("modified_question")
            if animal=="p0h1 dog" or animal=="k5f2 dog":
                modified_question = "The image of {} is image 1,  please identify the breed of this dog. For image 2, is there a dog of the same breed and similar appearance in image one? Please just answer yes or no.".format(animal)
            elif animal=="s5g3 cat":
                modified_question = "The image of {} is image 1,  please identify the breed of this cat. For image 2, is there a cat of the same breed and similar appearance in image one? Please just answer yes or no.".format(animal)
            else:
                modified_question = "The image of {} is image 1,  please identify the style of this sneaker. For image 2, is there a sneaker of the same style and similar appearance in image one? Please just answer yes or no.".format(animal)
        else:
            modified_question = question
            animal_image = None
            reference_image = None

        # Build message template
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        if reference_image:
            messages[0]["content"].append(
                {
                    "type": "image",
                    "image": reference_image,
                }
            )
        messages[0]["content"].append(
            {
                "type": "image",
                "image": image_path,
            }
        )
        messages[0]["content"].append(
            {
                "type": "text",
                "text": modified_question,
            }
        )

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
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer = output_text[0].split(".")[0].strip()
        print(output_text)
        print(answer)

        results.append({
            "Image path": image_path,
            "Question": question,
            "Correct Answer": correct_answer,
            "Generated Answer": answer
        })

        if answer.lower() == correct_answer.lower():
            correct_count += 1

    total_questions = len(qa_pairs) // 2
    accuracy = correct_count / total_questions
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(args.txt_output, 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%")
    
    print(f"save to {args.output_file} file")
    print(f"Save to {args.txt_output} file")

if __name__ == "__main__":
    main()
