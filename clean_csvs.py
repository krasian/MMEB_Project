import pandas as pd
import os

INPUT_FOLDER = "data/raw_csv"

files = os.listdir(INPUT_FOLDER)

for file in files:
    if file.endswith(".csv"):
        path = os.path.join(INPUT_FOLDER, file)
        df = pd.read_csv(file)
        print(df.head())

df = df.dropna(subset=["image_url"])

if "image_url" in df.columns:
    df = df.dropna(subset=["image_url"])

df = df.drop_duplicates(subset=["image_url"])

if any(sp in name for sp in TRAIN_SPECIES):
    if "place_country_name" in df.columns:
        df = df[df["place_country_name"] == "Netherlands"]

# seperate train and test species
TRAIN_SPECIES = ["blackbird", "blue_tit"]
TEST_SPECIES = ["robin", "crow"]

name = file.lower()

if any(sp in name for sp in TRAIN_SPECIES):
    print("train")

elif any(sp in name for sp in TEST_SPECIES):
    print("test")





