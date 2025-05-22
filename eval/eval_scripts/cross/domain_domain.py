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
    parser.add_argument("--image_dir", type=str, default="cross/cross_new_test_v2", help="Directory containing test/generated images")
    parser.add_argument("--ref_nature_dir", type=str, default="image", help="Directory containing reference images")
    parser.add_argument("--qa_file", type=str, default="cross/cross_new_test_qa.txt", help="Path to the QA data file")
    parser.add_argument("--model_name", type=str, default="../Qwen/Qwen2.5-VL-7B-Instruct", help="Name of the VLM model to use")
    parser.add_argument("--output_file", type=str, default="../results/eva/results2.json", help="Path to save the output results")
    parser.add_argument("--txt_output", type=str, default="../results/eva/results2.json", help="Path to save the output results")
    args = parser.parse_args()
    return args

# Extract animals from each prompt

def extract_animals(text, animal_list):
    """
    Extract animal names from a single prompt
    :param text: input text
    :param animal_list: list of animal names
    :return: list of extracted animal names
    """
    pattern = r'\b(' + '|'.join(re.escape(animal) for animal in animal_list) + r')\b'
    matches = re.findall(pattern, text, re.IGNORECASE)  # Case-insensitive matching
    return list(set(match.capitalize() for match in matches))  # Remove duplicates and keep the first letter capitalized

def main():
    args = get_args()
    print(f"Reference image directory: {args.ref_nature_dir}")
    print(f"Test/generated image directory: {args.image_dir}")
    
    # Animal list and their corresponding image paths (using reference image directory)
    animal_images = {
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

    # Animal image directory, points to reference images
    # Use argparse to pass in the reference image directory
    image_dir = args.ref_nature_dir
    print(f"Image directory: {image_dir}")

    for i in range(0, len(qa_pairs), 2):
        print(i)
        prompt_line = qa_pairs[i]
        qa_line = qa_pairs[i+1]

        prompt = prompt_line.split(': ')[1]
        animal = extract_animals(prompt, animal_images.keys())[0]
        print(f" Animal: {animal} ")
        animal_image = animal_images[animal]
        print(f"  Animal: {animal}, Image: {animal_image}")
        qa_data = json.loads(qa_line)
        question = qa_data['question']
        print(question)
        Question = "Only animals that are very similar to the one in image 1 will be considered {}. For image 2, ".format(animal) + question
        print(Question)
        correct_answer = qa_data['Correct Answer']
        
        search_path = os.path.join(args.image_dir, f"*{prompt[:100]}*.png")
        #print(search_path)
        #print(glob.glob(search_path))
        image_path = glob.glob(search_path)[0]
        print(image_path)
        
        if animal=="Pomelo" or animal=="Pomelos":
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
                            "image": os.path.join(args.ref_nature_dir,"image.png"),
                        },
                        {   
                            "type": "image",
                            "image": image_path,
                        },
                    {"type": "text", "text": "The image of the pomelo is image 1. Image 2 is a part of the pomelo. For image 3,".format(animal) + " " + question},
                    ],
                }
            ]

        # for blue feathers of macaws 0428
        
        elif (animal=="Macaws" or animal=="Macaw"):
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
                            "image": image_path,
                        },
                    {"type": "text", "text": "The image of Spix’s macaw is image 1, have over 30 percent blue feathers. For image 2, ".format(animal) + question},
                    ],
                }
            ]

        # For image 1, it is the image of Macaw. For image 2, does the bird have over 30% blue feathers? Please just answer yes or no.
        # for blue feathers of macaws 0428

        elif (animal=="Squid" or animal=="Squids"):
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
                            "image": image_path,
                        },
                    {"type": "text", "text": "The image of Squid is Image 1. Note that Squids have a distinct elongated body and tentacles, and should not be confused with Octopuses, which have a more rounded body and eight arms without distinct tentacles. For image 2, ".format(animal) + question},
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
                            "image": animal_image,
                        },
                        {   
                            "type": "image",
                            "image": image_path,
                        },
                    {"type": "text", "text": Question},
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
