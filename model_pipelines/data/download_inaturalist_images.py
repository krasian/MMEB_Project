"""
Download iNaturalist images from URLs in cleaned CSVs.

Reads each CSV in INPUT_CSV, downloads the image at `image_url` for each
row, saves it to a per-species folder, and writes out a new CSV where
`image_url` has been replaced with the local image path (renamed to
`image_path`).

This is a one-off data-prep utility. It is not imported by any pipeline
module — run it directly:

    python -m data.download_inaturalist_images
"""
import os
import pandas as pd
import requests


# ---- Configuration ---------------------------------------------------------
# Adjust these paths to wherever your raw CSVs live.
DATA_RAW_DIR = r"D:\test\data\raw"

INPUT_CSV = [
    "crow_clean.csv",
    "flamingo_clean.csv",
    "robin_clean.csv",
    "thrush_clean.csv",
    "toucan_clean.csv",
]

OUTPUT_CSV = [
    "updated_crow_data.csv",
    "updated_flamingo_data.csv",
    "updated_robin_data.csv",
    "updated_thrush_data.csv",
    "updated_toucan_data.csv",
]

IMAGE_FOLDER = [
    "downloaded_crow_images",
    "downloaded_flamingo_images",
    "downloaded_robin_images",
    "downloaded_thrush_images",
    "downloaded_toucan_images",
]


def download_image(url: str, filename: str):
    """Download an image from a URL and save it locally. Returns path or None."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        print(f"Failed to download: {url}")
        return None
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def main():
    # Make sure each output image folder exists.
    for folder in IMAGE_FOLDER:
        os.makedirs(os.path.join(DATA_RAW_DIR, folder), exist_ok=True)

    for n, csv_in in enumerate(INPUT_CSV):
        df = pd.read_csv(os.path.join(DATA_RAW_DIR, csv_in))
        new_image_paths = []

        for index, row in df.iterrows():
            image_url = row["image_url"]
            if pd.isna(image_url):
                new_image_paths.append(None)
                continue

            file_extension = ".jpg"
            filename = f"{row['id']}{file_extension}"
            filepath = os.path.join(DATA_RAW_DIR, IMAGE_FOLDER[n], filename)

            saved_path = download_image(image_url, filepath)
            new_image_paths.append(saved_path)

            print(f"Processed row {index + 1}/{len(df)}")

        df["image_url"] = new_image_paths
        df = df.rename(columns={"image_url": "image_path"})
        df.to_csv(OUTPUT_CSV[n], index=False)

        print("Done!")
        print(f"New CSV saved as: {OUTPUT_CSV[n]}")
        print(f"Images saved in: {IMAGE_FOLDER[n]}")


if __name__ == "__main__":
    main()
