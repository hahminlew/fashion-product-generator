from datasets import Dataset

from PIL import Image
from tqdm import tqdm
import os
from argparse import ArgumentParser
import json


def gen():
    # Load dataset information
    dataset_info = os.path.join(args.dataset_dir, 'dataset_BLIP.json')
    with open(dataset_info, "r") as f:
        dataset = json.load(f)

    # Prepare the dataset paths
    dataset_path = [os.path.join(args.dataset_dir, "img", filename) for filename in dataset if filename.endswith(('.jpg', '.png', '.jpeg'))]

    for img_path in tqdm(dataset_path):
        img_name = img_path.split("/")[-1]

        img_caption = dataset[img_name]["caption"]
        raw_image = Image.open(img_path).convert('RGB')

        yield {"image": raw_image, "text": img_caption}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default="./kream", type=str, 
                        help="Path to save images and captions")
    parser.add_argument("--hub_directory", default="", type=str, 
                        help="Path to hub directory to push datasets")
    args = parser.parse_args()

    dataset = Dataset.from_generator(gen)

    dataset.push_to_hub(args.hub_directory)