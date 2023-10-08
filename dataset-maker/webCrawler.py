from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.request import urlretrieve

import time
import re
from tqdm import tqdm
import os
from argparse import ArgumentParser
import json


def custom_webcrawling():
    '''
    -----------------------------------------------------------------------------------
                                     Descriptions
    -----------------------------------------------------------------------------------
    1. Inspect your desired website (command key: F12) and find iterable img locations.
    2. write a code here and rename your custom_webcrawling function if you want.
    '''
    # Start the Chrome web driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    # Initialize to save the dataset info in json format
    dataset = {

    }

    # Initialize a target url here
    url = ""
    print(f"Start custom_crawling for {url}...")

    # Navigate to the web page
    driver.get(url)

    # This part is freely editable.
    '''
    Write a code here.
    '''
    # Close the web browser
    driver.quit()

    # Save dataset info as json file
    with open(os.path.join(args.save_dir, 'dataset.json'), "w") as f:
            json.dump(dataset, f)
    
    print(f"All procedure successfully finished. Please check the results in {args.save_dir}.")


def kream_webcrawling():
    # Start the Chrome web driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Initialize a target url: 49-Outer, 50-Top, 51-Bottom
    category_name = ["outer", "top", "bottom"]
    category_dict = {
        category_name[0]: 49,
        category_name[1]: 50,
        category_name[2]: 51
    }

    # Initialize to save the dataset info in json format
    dataset = {

    }

    for category in category_name:
        url = f"https://kream.co.kr/search?tab={category_dict[category]}"
        
        print(f"{category}: Start kream_crawling for {url}...")

        # Navigate to the web page
        driver.get(url)

        # Scroll to load more data
        scroll_number = 1 # Variable to define the number of scrolls

        for _ in tqdm(range(scroll_number)):
            # Select a specific div element by class name
            div_elements = driver.find_elements(By.CLASS_NAME, "product_card")

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            time.sleep(1)

        print(f"Final {category} div_elements num:", len(div_elements))

        for i, div_element in enumerate(tqdm(div_elements)):
            # Find the target tag within the selected div element
            picture_element = div_element.find_element(By.TAG_NAME, "picture")
            name_element = div_element.find_element(By.CLASS_NAME, "name")

            # Find the image tag (e.g., img tag) inside the picture tag
            img_element = picture_element.find_element(By.TAG_NAME, "img")

            # Get the src attribute of the image
            img_src = img_element.get_attribute("src")
            name_text = name_element.text

            # Use regular expression to search and remove Korean characters, square brackets, 
            # and product serial numbers surrounded by round brackets
            filtered_name = re.sub(r'\[.*?\]|\(.*?\)|[가-힣]+', '', name_text) # Unicode range for Korean characters, square brackets, and serial numbers

            caption = f"{category}, " + filtered_name

            img_name = f"{category}_%05d.png" % i

            try:   
                # Path to save images
                img_path = os.path.join(args.save_dir, "img", img_name)
                
                # Download the image and save it as a file
                urlretrieve(img_src, img_path)

                # Save dataset info in dataset dictionary
                dataset[img_name] = {
                    "url": img_src,
                    "type": category,
                    "caption": caption
                }

            except Exception:
                pass

    # Close the web browser
    driver.quit()

    # Save dataset info as json file
    with open(os.path.join(args.save_dir, 'dataset.json'), "w") as f:
        json.dump(dataset, f)
    
    print(f"All procedure successfully finished. Please check the results in {args.save_dir}.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default="./kream", type=str, help="Path to save images and captions")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_dir, "img"), exist_ok=True)

    kream_webcrawling()