import pandas as pd
import os
import requests
from tqdm import tqdm

INPUT_FOLDER = "data/clean_csv"
OUTPUT_FOLDER = "data/images"

TIMEOUT = 10


def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        return False
    return False


def process_csv(file_path):
    pass


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith("_clean.csv")]

    for file in files:
        path = os.path.join(INPUT_FOLDER, file)
        process_csv(path)


if __name__ == "__main__":
    main()