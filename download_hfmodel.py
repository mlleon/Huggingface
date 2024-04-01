import os
import re
import requests
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from requests.exceptions import RequestException, HTTPError

# 配置日志记录
logging.basicConfig(filename='scraper.log', level=logging.ERROR)


def get_file_urls(model_name):
    """
    获取文件URL列表
    """
    original_text = "https://hf-mirror.com/ByteDance/AnimateDiff-Lightning/"
    new_text = f"https://hf-mirror.com/{model_name}/"
    link_base = re.sub(re.escape(original_text), new_text, original_text)
    repo_url = urljoin(link_base, 'tree/main')  # 构造模型存储库链接

    file_url_list = []
    try:
        response = requests.get(repo_url)
        response.raise_for_status()  # 检查HTTP状态码
        html_doc = response.text

        soup = BeautifulSoup(html_doc, 'html.parser')
        file_links = soup.find_all('a')
        for link in file_links:
            filename_span = link.find('span', class_='truncate group-hover:underline')
            if filename_span:
                filename = filename_span.text
                file_url = urljoin(link_base, f"resolve/main/{filename}?download=true")
                print(file_url)
                file_url_list.append(file_url)
    except (HTTPError, RequestException) as e:
        logging.error(f"Request Error:{e}")
        return []

    return file_url_list


def create_directory(directory):
    """
    创建目录
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory: {e}")
        return False
    return True


def write_urls_to_file(file_urls, output_file):
    """
    将URL写入文件
    """
    try:
        with open(output_file, 'a') as f:
            for file_url in file_urls:
                f.write(f"{file_url}\n")
    except OSError as e:
        logging.error(f"Failed to write to file:{e}")
        return False
    return True


def main(model_name):
    """
    主函数
    """
    file_urls = get_file_urls(model_name)
    if not file_urls:
        return

    model_name_with_underscore = model_name.replace("-", "_")

    # 提取目录名
    directory_name = os.path.dirname(model_name_with_underscore)
    output_directory = os.path.join(".", directory_name)

    if not create_directory(output_directory):
        return

    output_file = os.path.join(output_directory, f"{os.path.basename(model_name_with_underscore)}.txt")
    output_file_directory = os.path.join(output_directory, f"{os.path.basename(model_name_with_underscore)}")

    if not create_directory(output_file_directory):
        return

    # 检查文件是否为空
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print("File is not empty. Skipping writing to the TXT file.")
        return

    if write_urls_to_file(file_urls, output_file):
        print("The URL has been written to the file.")


if __name__ == "__main__":
    main("cardiffnlp/twitter-roberta-base-sentiment-latest")
