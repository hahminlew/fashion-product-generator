from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from tqdm import tqdm

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

    # Initialize a target url here
    url = ""

    # Navigate to the web page
    driver.get(url)

    # This part is freely editable.
    '''
    Write a code here.
    '''
    # Close the web browser
    driver.quit()
    return

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

    for category in category_name:
        url = f"https://kream.co.kr/search?tab={category_dict[category]}"

    # Navigate to the web page
    driver.get(url)

    # Scroll to load more data
    scroll_number = 1 # Variable to define the number of scrolls

    for _ in tqdm(range(scroll_number)):
        # Select a specific div element by class name
        div_elements = driver.find_elements(By.CLASS_NAME, "product_card")

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        time.sleep(1)

    print("final div_elements num: ", len(div_elements))

    for i, div_element in enumerate(div_elements):
        # Find the target tag within the selected div element
        picture_element = div_element.find_element(By.TAG_NAME, "picture")
        name_element = div_element.find_element(By.CLASS_NAME, "name")

        # Find the image tag (e.g., img tag) inside the picture tag
        img_element = picture_element.find_element(By.TAG_NAME, "img")

        # Get the src attribute of the image
        img_src = img_element.get_attribute("src")
        name_text = name_element.text

        # Use regular expression to search and remove Korean characters and square brackets
        pattern = re.compile(r'[가-힣\[\]]+')  # Unicode range for Korean characters and square brackets
        filtered_name = pattern.sub('', name_text)
        # Filtering product serial numbers
        filtered_name = re.sub(r'\([^)]*\)', '', filtered_name)

        caption = f"{category}, " + filtered_name

        # Print the src attribute of the image
        print("Image Attributes:", img_src, caption)

    # Close the web browser
    driver.quit()

if __name__ == "__main__":
    kream_webcrawling()