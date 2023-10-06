from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

url = "https://kream.co.kr/search"

# 웹 페이지 열기
driver.get(url)

# 웹 페이지의 내용을 출력
print(driver.page_source)

# 웹 드라이버 종료
driver.quit()


# headers = {
#     "User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
# }
# resp = requests.get(url, headers=headers)
# soup = BeautifulSoup(resp.text, 'lxml')

# search = soup.find_all('div', attrs={'data-v-a443911e': True, 'class': 'search_content'})

# print(search)

# search = soup
# print(len(search))
# print(search)