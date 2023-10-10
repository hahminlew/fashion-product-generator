from urllib.request import urlretrieve

from tqdm import tqdm
import os
from argparse import ArgumentParser
import json


def download_kream():
    print("Start downloading KREAM Product Dataset...")

    # Load dataset information
    dataset_info = os.path.join(args.save_dir, 'dataset.json')
    with open(dataset_info, "r") as f:
        dataset = json.load(f)

    for img_name in tqdm(dataset):
        try:   
            # Path to save images
            img_path = os.path.join(args.save_dir, "img", img_name)
            
            # Download the image and save it as a file
            urlretrieve(dataset[img_name]['url'], img_path)

            break

        except Exception:
            pass

    print(f"All procedure successfully finished. Please check the results in {args.save_dir}.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default="./kream", type=str, help="Path to save images and captions")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_dir, "img"), exist_ok=True)

    download_kream()