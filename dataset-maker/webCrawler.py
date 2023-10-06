from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re

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
    category_name = ['outer', 'top', 'bottom']
    category_dict = {
        category_name[0]: 49,
        category_name[1]: 50,
        category_name[2]: 51
    }

    for category in category_name:
        url = f"https://kream.co.kr/search?tab={category_dict[category]}"

    url = "https://kream.co.kr/search?tab=54"
    # Navigate to the web page
    driver.get(url)

    # Scroll to load more data
    scroll_count = 0 # Variable to track the number of scrolls

    span_element = driver.find_element(By.CLASS_NAME, "title")
    total_product_num = span_element.text # Variable to crawl total product numbers
    pattern = re.compile(r'[가-힣\,]+')
    total_product_num = pattern.sub('', total_product_num)
    total_product_num = int(total_product_num)

    total_scroll_count = total_product_num // 50 + 1

    data_collected = set()

    while scroll_count <= total_scroll_count:
        # Select a specific div element by class name
        div_elements = driver.find_elements(By.CLASS_NAME, "product_card")

        for i, div_element in enumerate(div_elements):
            # Find the target tag within the selected div element
            picture_element = div_element.find_element(By.TAG_NAME, "picture")
            name_element = div_element.find_element(By.CLASS_NAME, "name")

            # Find the image tag (e.g., img tag) inside the picture tag
            img_element = picture_element.find_element(By.TAG_NAME, "img")

            # Get the src attribute of the image
            img_src = img_element.get_attribute("src")
            name_src = name_element.text

            # Use regular expression to search and remove Korean characters and square brackets
            pattern = re.compile(r'[가-힣\[\]]+')  # Unicode range for Korean characters and square brackets
            filtered_name_src = pattern.sub('', name_src)

            # Print the src attribute of the image
            # print("Image Attributes:", img_src, filtered_name_src)

            if filtered_name_src not in data_collected:
                data_collected.add(filtered_name_src)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Scroll down to load more data
        time.sleep(2)  # Delay between scrolls (adjust as needed, in seconds)

        scroll_count += 1

        print(len(div_elements))

    print('final div_elements num: ', len(div_elements), '| final data num: ', len(data_collected))

    # Close the web browser
    driver.quit()

    # for i, div_element in enumerate(div_elements):
    #     # Find the target tag within the selected div element
    #     picture_element = div_element.find_element(By.TAG_NAME, "picture")
    #     name_element = div_element.find_element(By.CLASS_NAME, "name")

    #     # Find the image tag (e.g., img tag) inside the picture tag
    #     img_element = picture_element.find_element(By.TAG_NAME, "img")

    #     # Get the src attribute of the image
    #     img_src = img_element.get_attribute("src")
    #     name_src = name_element.text

    #     # Use regular expression to search and remove Korean characters and square brackets
    #     pattern = re.compile(r'[가-힣\[\]]+')  # Unicode range for Korean characters and square brackets
    #     filtered_name_src = pattern.sub('', name_src)

    #     # Print the src attribute of the image
    #     print("Image Attributes:", img_src, filtered_name_src)

if __name__ == "__main__":
    kream_webcrawling()