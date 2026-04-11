import pandas as pd
import os

INPUT_FOLDER = "data/raw_csv"

files = os.listdir(INPUT_FOLDER)

for file in files:
    if file.endswith(".csv"):
        path = os.path.join(INPUT_FOLDER, file)
        df = pd.read_csv(file)
        print(df.head())


# seperate train and test species
TRAIN_SPECIES = ["blackbird", "blue_tit", "great_tit", "sparrow"]
TEST_SPECIES = ["robin", "crow", "starling", "thrush","flamingo","toucan"]

name = file.lower()

if any(sp in name for sp in TRAIN_SPECIES):
    print("train")

elif any(sp in name for sp in TEST_SPECIES):
    print("test")

# clean function
def clean_df(df, country=None):

    df = df.dropna(subset=["image_url"])
    df = df.drop_duplicates(subset=["image_url"])

    if country and "place_country_name" in df.columns:
        df = df[df["place_country_name"] == country]

    return df


OUTPUT_FOLDER = "data/clean_csv"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

out_name = file.replace(".csv", "_clean.csv")
out_path = os.path.join(OUTPUT_FOLDER, out_name)
df.to_csv(out_path, index = False)

MAX_SAMPLES = 2000

if len(df) > MAX_SAMPLES:
    df = df.sample(MAX_SAMPLES, random_state =42)


print(f"\nProcessing: {file}")
print("Final rows:", len(df))
print("Saved:", out_path)











