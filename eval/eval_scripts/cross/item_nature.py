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
    parser.add_argument("--ref_image_dir", type=str, default="/eva/cross_final/raw/datasets", help="Directory containing reference images (for animal_images and natural_images)")
    parser.add_argument("--ref_nature_dir", type=str, default="/eva/cross_final/image", help="Directory containing reference natural images (for natural_images)")
    parser.add_argument("--qa_file", type=str, default="/eva/cross/cross_new_test_qa.txt", help="Path to the QA data file")
    parser.add_argument("--model_name", type=str, default="/Qwen/Qwen2.5-VL-7B-Instruct", help="Name of the VLM model to use")
    parser.add_argument("--output_file", type=str, default="/results/eva/results2.json", help="Path to save the output results")
    parser.add_argument("--txt_output", type=str, default="/results/eva/results2.json", help="Path to save the output results")
    args = parser.parse_args()
    return args

# Extract special animal names from a single prompt
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
    print(f"Reference animal image directory: {args.ref_image_dir}")
    print(f"Reference natural image directory: {args.ref_nature_dir}")
    print(f"Test/generated image directory: {args.image_dir}")

    animal_images = {
        "s5g3 cat": os.path.join(args.ref_image_dir, "cat2/00.jpg"),
        "k5f2 dog": os.path.join(args.ref_image_dir, "dog3/00.jpg"),
        "b9l1 sneaker": os.path.join(args.ref_image_dir, "shiny_sneaker/00.jpg"),
        "p0h1 dog": os.path.join(args.ref_image_dir, "dog/00.jpg")
    }
    natural_images = {
        "Squid": os.path.join(args.ref_nature_dir, "A squid floats motionless, relying on its color-sh-2.png"),
        "Squids": os.path.join(args.ref_nature_dir, "A squid floats motionless, relying on its color-sh-2.png"),
        "Quokka": os.path.join(args.ref_nature_dir, "A quokka digs softly into the earth, uncovering a -1.png"),
        "Quokkas": os.path.join(args.ref_nature_dir, "A quokka digs softly into the earth, uncovering a -1.png"),
        "Gerenuk": os.path.join(args.ref_nature_dir, "Gerenuk pauses near a large rock, its dark eyes sc-2.png"),
        "Gerenuks": os.path.join(args.ref_nature_dir, "Gerenuk pauses near a large rock, its dark eyes sc-2.png"),
        "Macaw": os.path.join(args.ref_nature_dir, "Two Spix's Macaws call to each other, their cries -0.png"),
        "Macaws": os.path.join(args.ref_nature_dir, "Two Spix's Macaws call to each other, their cries -0.png"),
        "Pomelo": os.path.join(args.ref_nature_dir, "A Pomelo hangs from a sturdy tree branch, swaying -3.png"),
        "Pomelos": os.path.join(args.ref_nature_dir, "A Pomelo hangs from a sturdy tree branch, swaying -3.png")
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

        # Extract special animals and natural items from prompt
        animals = extract_animals(prompt, animal_images.keys())
        natures = extract_animals(prompt, natural_images.keys())
        if not animals and not natures:
            print(f"No special animal and natural items found in prompt: {prompt}")
            continue
        if animals:
            animal = animals[0]
            animal_image = animal_images[animal]
            print(f"Identified special animal: {animal}, Image path: {animal_image}")
        if natures:
            nature = natures[0]
            natural_image = natural_images[nature]
            print(f"Identified special natural items: {nature}, Image path: {natural_image}")

        # Search for the corresponding generated image path
        search_path = os.path.join(args.image_dir, f"*{prompt[:100]}*.png")
        image_paths = glob.glob(search_path)
        if not image_paths:
            print(f"No matching image found for prompt: {prompt}")
            continue
        image_path = image_paths[0]
        print(f"Matched generated image: {image_path}")
        print(question)

        # Determine if it is a special animal or natural item question
        if is_special_animal_question(question, animal_images.keys()):
            if animal=="p0h1 dog" or animal=="k5f2 dog":
                modified_question = "The image of {} is image 1,  please identify the breed of this dog. For image 2, is there a dog of the same breed and similar appearance in image one? Please just answer yes or no.".format(animal)
            elif animal=="s5g3 cat":
                modified_question = "The image of {} is image 1,  please identify the breed of this cat. For image 2, is there a cat of the same breed and similar appearance in image one? Please just answer yes or no.".format(animal)
            else:
                modified_question = "The image of {} is image 1,  please identify the style of this sneaker. For image 2, is there a sneaker of the same style and similar appearance in image one? Please just answer yes or no.".format(animal)
        elif is_special_animal_question(question, natural_images.keys()):
            if nature == "Macaw" or nature == "Macaws":
                modified_question =  "The image of Spix’s macaw is image 1, have over 30 percent blue feathers. For image 2, ".format(nature) + question
            elif nature == "Squid" or nature == "Squids":
                modified_question =  "The image of Squid is Image 1. Note that Squids have a distinct elongated body and tentacles, and should not be confused with Octopuses, which have a more rounded body and eight arms without distinct tentacles. For image 2, ".format(nature) + question
            else:
                modified_question = "Only animals that are very similar to the one in image 1 will be considered {}. For image 2, ".format(nature) + question
        else:
            modified_question = question
        # print(question)
        # print(modified_question)

        reference_image = load_and_resize_image(animal_image)

        # Build message template
        if nature=="Pomelo" or nature=="Pomelos":
                messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": animal_image,
                        },
                        {
                            "type": "image",
                            "image": os.path.join(args.ref_nature_dir, "image.png"),
                        },
                        {
                            "type": "image",
                            "image": image_path,
                        },
                    {"type": "text", "text": "The image of the pomelo is image 1. Image 2 is a part of the pomelo. For image 3,".format(animal) + " " + question},
                    ],
                }
            ]

        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": reference_image,
                        },
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": modified_question,
                        },
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
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer = output_text[0].split(".")[0].strip()
        # print(output_text)
        print(answer)

        results.append({
            "Question": question,
            "Correct Answer": correct_answer,
            "Generated Answer": answer,
            "image_path": image_path
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
