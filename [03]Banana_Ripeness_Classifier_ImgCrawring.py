# Basic image crawling

import os
import requests
from bs4 import BeautifulSoup

# 검색어 및 폴더 설정
search_keywords = ["black banana", "green banana", "yellow banana"]
save_folders = ["overripeimg", "unripeimg", "fitripeimg"]

# 이미지 다운로드 함수 정의
def download_images(keyword, save_folder):
    # 이미지 검색 결과 페이지 URL 설정
    search_url = f"https://www.bing.com/images/search?q={keyword}"
    
    # 웹 페이지 가져오기
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 이미지 링크 찾기 및 다운로드
    img_tags = soup.find_all("img")
    for img_tag in img_tags:
        img_url = img_tag.get("src")
        if img_url and img_url.startswith("https"):
            img_data = requests.get(img_url).content
            with open(os.path.join(save_folder, f"{keyword}_{len(os.listdir(save_folder)) + 1}.jpg"), "wb") as img_file:
                img_file.write(img_data)

# 이미지 다운로드 수행
for keyword, folder in zip(search_keywords, save_folders):
    os.makedirs(folder, exist_ok=True)
    download_images(keyword, folder)

print("Image download completed.")


# Optional image crawling 
import os
import requests
import cv2
from bs4 import BeautifulSoup

# 검색어 및 폴더 설정을 위한 함수
def set_keywords_and_folders():
    search_keywords = {
        "black banana": "overripeimg",
        "green banana": "unripeimg",
        "yellow banana": "fitripeimg",
      
    }
    return search_keywords

# 이미지 다운로드 및 리사이징 함수 정의
def download_images(keyword, save_folder, max_count=50):
    search_url = f"https://www.pixabay.com/images/search?q={keyword}&qft=+filterui:filetype-jpg"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    img_tags = soup.find_all("img")
    
    count = 0
    for img_tag in img_tags:
        if count >= max_count:
            break
        img_url = img_tag.get("src")
        if img_url and img_url.startswith("https"):
            img_data = requests.get(img_url).content
            filename = f"{save_folder.split('img')[0]}_{count + 1}.jpg"
            filepath = os.path.join(save_folder, filename)
            
            with open(filepath, "wb") as img_file:
                img_file.write(img_data)
            
            # 이미지 리사이징
            img = cv2.imread(filepath)
            h, w, _ = img.shape
            if h > 200 or w > 200:
                img = cv2.resize(img, (200, 200))
                cv2.imwrite(filepath, img)
                
            count += 1

# 검색어 및 폴더 설정
search_keywords = set_keywords_and_folders()

# 이미지 다운로드 수행
for keyword, folder in search_keywords.items():
    os.makedirs(folder, exist_ok=True)
    download_images(keyword, folder)

print("Image download and resizing completed.")
