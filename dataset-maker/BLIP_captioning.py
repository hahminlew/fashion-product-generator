import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from PIL import Image
from tqdm import tqdm
import os
from argparse import ArgumentParser
import json


def blip_image_captioning():
    # Check if CUDA is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Blip Image Captioning model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    # Load dataset information
    dataset_info = os.path.join(args.dataset_dir, 'dataset.json')
    with open(dataset_info, "r") as f:
        dataset = json.load(f)

    # Prepare the dataset paths
    dataset_path = [os.path.join(args.dataset_dir, "img", filename) for filename in dataset_path if filename.endswith(('.jpg', '.png', '.jpeg'))]

    for img_path in tqdm(dataset_path):
        img_name = img_path.split("/")[-1]
        img_caption = dataset[img_name]["caption"]
        
        raw_image = Image.open(img_path).convert('RGB')

        # Conditional or unconditional image captioning
        if args.use_condition:
            text = args.text_condition
            inputs = processor(raw_image, text, return_tensors="pt").to(device)

            out = model.generate(**inputs)
            blip_caption = processor.decode(out[0], skip_special_tokens=True)

            final_img_caption = img_caption + ", " + blip_caption

            dataset[img_name]["caption"] = final_img_caption

        else:
            inputs = processor(raw_image, return_tensors="pt").to(device)

            out = model.generate(**inputs)
            blip_caption = processor.decode(out[0], skip_special_tokens=True)

            final_img_caption = img_caption + ", " + blip_caption

            dataset[img_name]["caption"] = final_img_caption

    # Save updated dataset information as a JSON file
    with open(os.path.join(args.dataset_dir, 'dataset_BLIP.json'), "w") as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default="./kream", type=str, 
                        help="Path to save images and captions")
    parser.add_argument("--use_condition", action='store_true',
                        help='Use conditional image captioning or not')
    parser.add_argument("--text_condition", default="a photography of", type=str, 
                        help="Context of conditional image captioning")
    args = parser.parse_args()

    blip_image_captioning()