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

        break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default="./kream", type=str, 
                        help="Path to save images and captions")
    args = parser.parse_args()

    ds = Dataset.from_generator(gen)

    print(ds)
    print(ds[0])