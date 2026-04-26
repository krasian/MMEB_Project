import os
import pandas as pd
import requests

# Input CSV file
INPUT_CSV = [r"D:\test\data\raw\crow_clean.csv ",r"D:\test\data\raw\flamingo_clean.csv",
             r"D:\test\data\raw\robin_clean.csv", r"D:\test\data\raw\thrush_clean.csv",
             r"D:\test\data\raw\toucan_clean.csv",
     ]

# Output CSV file
OUTPUT_CSV = ["updated_crow_data.csv","updated_flamingo_data.csv",
              "updated_robin_data.csv","updated_thrush_data.csv","updated_toucan_data.csv"]

# Folder where images will be stored
IMAGE_FOLDER = ["downloaded_crow_images","downloaded_flamingo_images",
               "downloaded_robin_images","downloaded_thrush_images","downloaded_toucan_images"]


for i in IMAGE_FOLDER:
    os.makedirs(fr"D:\test\data\raw\{i}", exist_ok=True)

# Load CSV into a DataFrame
df_list = []
for i in INPUT_CSV:
    df_list.append(pd.read_csv(i))


def download_image(url, filename):
    """
    Downloads an image from a URL and saves it locally.

    Args:
        url (str): The image URL
        filename (str): Local file path to save the image

    Returns:
        str: Path to saved image OR None if failed
    """
    try:
        response = requests.get(url, timeout=10)

        # Check if request was successful
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        else:
            print(f"Failed to download: {url}")
            return None

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


n = -1
for df in df_list:
    new_image_paths = []
    n += 1
    for index, row in df.iterrows():
        image_url = row['image_url']

        # Skip if URL is missing
        if pd.isna(image_url):
            new_image_paths.append(None)
            continue

        # Create a unique filename using row ID
        file_extension = ".jpg"  # default extension
        filename = f"{row['id']}{file_extension}"

        filepath = os.path.join(fr"D:\test\data\raw\{IMAGE_FOLDER[n]}", filename)


        saved_path = download_image(image_url, filepath)


        new_image_paths.append(saved_path)

        print(f"Processed row {index + 1}/{len(df)}")

    df['image_url'] = new_image_paths
    df = df.rename(columns={'image_url': 'image_path'})

    df.to_csv(OUTPUT_CSV[n], index=False)

    print("Done!")
    print(f"New CSV saved as: {OUTPUT_CSV[n]}")
    print(f"Images saved in: {IMAGE_FOLDER}")