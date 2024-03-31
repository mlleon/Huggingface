import os
import re
import requests
import logging
from bs4 import BeautifulSoup
from requests.exceptions import RequestException, HTTPError

# Configure logging
logging.basicConfig(filename='scraper.log', level=logging.ERROR)


def get_file_urls(model_name):
    original_text = "https://hf-mirror.com/ByteDance/AnimateDiff-Lightning/"
    new_text = f"https://hf-mirror.com/{model_name}/"
    link_base = re.sub(re.escape(original_text), new_text, original_text)
    repo_url = f"{link_base}tree/main"  # Construct model repository link

    file_url_list = []
    try:
        response = requests.get(repo_url)
        response.raise_for_status()  # Check HTTP status code
        html_doc = response.text

        soup = BeautifulSoup(html_doc, 'html.parser')
        file_links = soup.find_all('a')
        for link in file_links:
            filename_span = link.find('span', class_='truncate group-hover:underline')
            if filename_span:
                filename = filename_span.text
                file_url = f"{link_base}resolve/main/{filename}?download=true"
                print(file_url)
                file_url_list.append(file_url)
    except HTTPError as e:
        logging.error(f"HTTP Error: {e}")
        return []
    except RequestException as e:
        logging.error(f"Request Error: {e}")
        return []

    model_name_with_underscore = model_name.replace("-", "_")

    # Extract directory name from model_name
    directory_name = os.path.dirname(model_name_with_underscore)
    output_directory = f"./{directory_name}"

    # Create the directory if it doesn't exist
    try:
        os.makedirs(output_directory, exist_ok=True)  # Use exist_ok to handle directory existence
    except OSError as e:
        logging.error(f"Failed to create directory: {e}")
        return []

    output_file = f"{output_directory}/{os.path.basename(model_name_with_underscore)}.txt"

    # Check if the file is empty
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print("File is not empty. Skipping writing to the TXT file.")
        return file_url_list

    # Write to file
    try:
        with open(output_file, 'a') as f:
            for file_url in file_url_list:
                f.write(f"{file_url}\n")
    except OSError as e:
        logging.error(f"Failed to write to file: {e}")
        return []

    return file_url_list


# Call the function and specify the output file path
file_urls = get_file_urls("cardiffnlp/twitter-roberta-base-sentiment-latest")
