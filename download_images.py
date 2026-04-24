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
    df = pd.read_csv(file_path)

    if "image_url" not in df.columns:
        print("No image_url column, skipping:", file_path)
        return

    species_name = os.path.basename(file_path).replace("_clean.csv", "")
    species_folder = os.path.join(OUTPUT_FOLDER, species_name)

    os.makedirs(species_folder, exist_ok=True)

    print(f"\nDownloading images for: {species_name}")

    success = 0

    for i, url in tqdm(enumerate(df["image_url"]), total=len(df)):
        filename = f"{i}.jpg"
        save_path = os.path.join(species_folder, filename)

        if download_image(url, save_path):
            success += 1

    print(f"Downloaded {success}/{len(df)} images")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith("_clean.csv")]

    for file in files:
        path = os.path.join(INPUT_FOLDER, file)
        process_csv(path)


if __name__ == "__main__":
    main()
